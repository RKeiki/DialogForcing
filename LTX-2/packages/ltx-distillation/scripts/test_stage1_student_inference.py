#!/usr/bin/env python3
"""
Run single-GPU inference with a distilled Stage-1 student checkpoint.

Example:
    python scripts/test_stage1_student_inference.py \
        --checkpoint /path/to/outputs/stage1_bidirectional_dmd/checkpoint_000050/model.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
for package_src in (
    REPO_ROOT / "packages" / "ltx-distillation" / "src",
    REPO_ROOT / "packages" / "ltx-core" / "src",
    REPO_ROOT / "packages" / "ltx-pipelines" / "src",
    REPO_ROOT / "packages" / "ltx-causal" / "src",
):
    package_src_str = str(package_src)
    if package_src_str not in sys.path:
        sys.path.insert(0, package_src_str)

import torch
from omegaconf import OmegaConf

from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.video_vae import TilingConfig, decode_video as vae_decode_video, get_video_chunks_number
from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
from ltx_pipelines.utils.helpers import cleanup_memory
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_ledger import ModelLedger


def resolve_config_path(checkpoint_path: Path, config_path: Optional[str]) -> Path:
    if config_path is not None:
        return Path(config_path)

    candidate = checkpoint_path.parent.parent / "config.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not infer config.yaml from checkpoint path {checkpoint_path}. "
        "Please pass --config-path explicitly."
    )


def load_prompts(prompt_path: str, num_prompts: int) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts[:num_prompts]


def build_denoising_sigmas(config, device: torch.device) -> torch.Tensor:
    num_inference_steps = int(getattr(config, "num_inference_steps", 40))
    full_sigmas = LTX2Scheduler().execute(steps=num_inference_steps)
    denoising_sigmas = []
    for timestep in config.denoising_step_list:
        target_sigma = timestep / 1000.0
        idx = (full_sigmas - target_sigma).abs().argmin().item()
        denoising_sigmas.append(full_sigmas[idx])
    return torch.stack(denoising_sigmas).to(device)


def compute_latent_shapes(
    *,
    num_frames: int,
    video_height: int,
    video_width: int,
    batch_size: int = 1,
    latent_channels: int = 128,
    vae_temporal_compression: int = 8,
    vae_spatial_compression: int = 32,
    video_fps: float = 24.0,
    audio_sample_rate: int = 16000,
    audio_hop_length: int = 160,
    audio_latent_downsample: int = 4,
) -> Tuple[list[int], list[int]]:
    if (num_frames - 1) % vae_temporal_compression != 0:
        raise ValueError(
            f"num_frames must be 1 + {vae_temporal_compression}*k, got {num_frames}"
        )

    latent_frames = 1 + (num_frames - 1) // vae_temporal_compression
    latent_h = video_height // vae_spatial_compression
    latent_w = video_width // vae_spatial_compression

    video_duration = float(num_frames) / float(video_fps)
    audio_latent_fps = (
        float(audio_sample_rate) / float(audio_hop_length) / float(audio_latent_downsample)
    )
    audio_frames = round(video_duration * audio_latent_fps)

    video_shape = [batch_size, latent_frames, latent_channels, latent_h, latent_w]
    audio_shape = [batch_size, audio_frames, latent_channels]
    return video_shape, audio_shape


def add_noise(original: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    if sigma.dim() == 1:
        sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
    elif sigma.dim() == 2:
        sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
    sigma = sigma.to(dtype=original.dtype)
    return ((1 - sigma) * original + sigma * noise).to(dtype=original.dtype)


@torch.inference_mode()
def decode_and_save_sample(
    *,
    model_ledger: ModelLedger,
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    save_path: Path,
    fps: int,
    audio_sample_rate: int,
) -> Tuple[bool, Optional[Path]]:
    if video_latent.dim() != 5 or video_latent.shape[2] != 128:
        raise ValueError(f"Expected video latent shape [B, F, 128, H, W], got {tuple(video_latent.shape)}")

    video_latent_for_decode = video_latent.permute(0, 2, 1, 3, 4).contiguous()
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(
        num_frames=8 * (video_latent_for_decode.shape[2] - 1) + 1,
        tiling_config=tiling_config,
    )

    audio_waveform = None
    try:
        audio_decoder = model_ledger.audio_decoder()
        vocoder = model_ledger.vocoder()
        if audio_latent.dim() != 3:
            raise ValueError(f"Expected audio latent shape [B, T, C], got {tuple(audio_latent.shape)}")
        z_channels = getattr(audio_decoder, "z_channels", None)
        if z_channels is None:
            raise ValueError("Audio decoder is missing z_channels")
        batch_size, audio_frames, packed_channels = audio_latent.shape
        if packed_channels % z_channels != 0:
            raise ValueError(
                f"Audio latent packed dimension {packed_channels} is not divisible by z_channels={z_channels}"
            )
        latent_mel = packed_channels // z_channels
        audio_latent_for_decode = audio_latent.reshape(
            batch_size,
            audio_frames,
            z_channels,
            latent_mel,
        ).permute(0, 2, 1, 3).contiguous()
        audio_waveform = vae_decode_audio(audio_latent_for_decode, audio_decoder, vocoder).cpu().float()
        del audio_decoder, vocoder
        cleanup_memory()
    except Exception as exc:
        print(f"[Warn] Audio decode failed for {save_path.name}: {exc}")
        audio_waveform = None

    wav_path = None
    wrote_audio = audio_waveform is not None
    try:
        video_decoder = model_ledger.video_decoder()
        decoded_video = vae_decode_video(
            video_latent_for_decode,
            video_decoder,
            tiling_config=tiling_config,
        )
        encode_video(
            video=decoded_video,
            fps=fps,
            audio=audio_waveform if audio_waveform is not None else None,
            audio_sample_rate=audio_sample_rate if audio_waveform is not None else None,
            output_path=str(save_path),
            video_chunks_number=video_chunks_number,
        )
        del decoded_video, video_decoder
        cleanup_memory()
    except Exception as exc:
        print(f"[Warn] encode_video failed for {save_path.name}: {exc}")
        if audio_waveform is not None:
            try:
                import torchaudio

                wav_path = save_path.with_suffix(".wav")
                wav_to_save = audio_waveform
                if wav_to_save.dim() == 1:
                    wav_to_save = wav_to_save.unsqueeze(0)
                torchaudio.save(str(wav_path), wav_to_save, audio_sample_rate)
            except Exception as wav_exc:
                print(f"[Warn] Saving wav failed for {save_path.name}: {wav_exc}")
        raise

    return wrote_audio, wav_path


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_xxxxx/model.pt")
    parser.add_argument("--config-path", type=str, default=None, help="Optional run config.yaml path")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt-path", type=str, default=None, help="Override prompt file path")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--audio-sample-rate", type=int, default=24000)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = resolve_config_path(checkpoint_path, args.config_path).resolve()
    config = OmegaConf.load(config_path)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            torch.cuda.set_device(device.index)
            device = torch.device("cuda", device.index)
    dtype = torch.bfloat16 if getattr(config, "mixed_precision", True) else torch.float32

    prompt_path = args.prompt_path or config.data_path
    all_prompts = load_prompts(prompt_path, args.start_index + args.num_prompts)
    prompts = all_prompts[args.start_index: args.start_index + args.num_prompts]
    if not prompts:
        raise ValueError("No prompts selected after applying --start-index/--num-prompts")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = checkpoint_path.parent / f"inference_first_{len(prompts):03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] checkpoint={checkpoint_path}")
    print(f"[Load] config={config_path}")
    print(f"[Load] prompts={prompt_path} | selected={len(prompts)}")
    print(f"[Save] output_dir={output_dir}")

    model_ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=config.checkpoint_path,
        gemma_root_path=config.gemma_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    generator_state = checkpoint.get("generator", checkpoint)

    denoising_sigmas = build_denoising_sigmas(config, device)

    video_shape, audio_shape = compute_latent_shapes(
        num_frames=int(config.num_frames),
        video_height=int(config.video_height),
        video_width=int(config.video_width),
        batch_size=1,
    )

    prompt_manifest_path = output_dir / "prompts.txt"
    with open(prompt_manifest_path, "w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts, start=args.start_index):
            f.write(f"{idx}\t{prompt}\n")

    total_start = time.perf_counter()
    rng_devices = []
    if device.type == "cuda":
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    for offset, prompt in enumerate(prompts):
        prompt_idx = args.start_index + offset
        sample_seed = args.seed + prompt_idx
        save_path = output_dir / f"sample_{prompt_idx:03d}.mp4"
        prompt_txt_path = output_dir / f"sample_{prompt_idx:03d}.txt"
        with open(prompt_txt_path, "w", encoding="utf-8") as f:
            f.write(prompt + "\n")

        with torch.no_grad():
            text_encoder = create_text_encoder_wrapper(
                checkpoint_path=config.checkpoint_path,
                gemma_path=config.gemma_path,
                device=device,
                dtype=dtype,
            )
            text_encoder.eval()
            conditional_dict = text_encoder(text_prompts=[prompt])
            del text_encoder
            cleanup_memory()

            generator = create_ltx2_wrapper(
                checkpoint_path=config.checkpoint_path,
                gemma_path=config.gemma_path,
                device=device,
                dtype=dtype,
                video_height=int(config.video_height),
                video_width=int(config.video_width),
            )
            incompatible = generator.load_state_dict(generator_state, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            if offset == 0:
                print(f"[Load] generator state loaded | missing={len(missing)} unexpected={len(unexpected)}")
            generator.eval()

            pipeline = BidirectionalAVInferencePipeline(
                generator=generator,
                add_noise_fn=add_noise,
                denoising_sigmas=denoising_sigmas,
            )
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(sample_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(sample_seed)
                step_start = time.perf_counter()
                video_latent, audio_latent = pipeline.generate(
                    video_shape=tuple(video_shape),
                    audio_shape=tuple(audio_shape),
                    conditional_dict=conditional_dict,
                )
                elapsed = time.perf_counter() - step_start

        del conditional_dict
        del pipeline, generator
        if device.type == "cuda":
            cleanup_memory()

        wrote_audio, wav_path = decode_and_save_sample(
            model_ledger=model_ledger,
            video_latent=video_latent,
            audio_latent=audio_latent,
            save_path=save_path,
            fps=args.fps,
            audio_sample_rate=args.audio_sample_rate,
        )
        print(
            f"[{offset + 1:03d}/{len(prompts):03d}] prompt_idx={prompt_idx} "
            f"time={elapsed:.2f}s video={save_path.name} "
            f"audio={'embedded' if wrote_audio else (wav_path.name if wav_path else 'none')}"
        )

        del video_latent, audio_latent
        if device.type == "cuda":
            cleanup_memory()

    total_elapsed = time.perf_counter() - total_start
    print(f"[Done] generated {len(prompts)} samples in {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
