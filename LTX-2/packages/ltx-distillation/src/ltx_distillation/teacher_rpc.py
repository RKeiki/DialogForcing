from __future__ import annotations

import pickle
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ltx_distillation.distributed_topology import DistributedTopology


_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.bool: 5,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}
_OPTIONAL_TENSOR_MAX_NDIM = 8
_OPTIONAL_TENSOR_HEADER_SIZE = 3 + _OPTIONAL_TENSOR_MAX_NDIM

_REAL_SCORE_REQUEST_KEYS = (
    "noisy_video",
    "noisy_audio",
    "video_sigma",
    "audio_sigma",
    "conditional_video_context",
    "conditional_audio_context",
    "conditional_attention_mask",
    "unconditional_video_context",
    "unconditional_audio_context",
    "unconditional_attention_mask",
)
_REAL_SCORE_RESPONSE_KEYS = (
    "pred_real_cond_video",
    "pred_real_cond_audio",
    "pred_real_uncond_video",
    "pred_real_uncond_audio",
)
_CONDITIONING_RESPONSE_KEYS = (
    "conditional_video_context",
    "conditional_audio_context",
    "conditional_attention_mask",
    "unconditional_video_context",
    "unconditional_audio_context",
    "unconditional_attention_mask",
)


def _optional_tensor_header(tensor: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    header = torch.full(
        (_OPTIONAL_TENSOR_HEADER_SIZE,),
        fill_value=-1,
        device=device,
        dtype=torch.int64,
    )
    if tensor is None:
        header[0] = 0
        return header

    if tensor.dim() > _OPTIONAL_TENSOR_MAX_NDIM:
        raise ValueError(
            f"Teacher RPC supports tensors with ndim <= {_OPTIONAL_TENSOR_MAX_NDIM}, "
            f"got ndim={tensor.dim()}"
        )
    dtype_code = _DTYPE_TO_CODE.get(tensor.dtype)
    if dtype_code is None:
        raise TypeError(f"Unsupported tensor dtype for teacher RPC: {tensor.dtype}")
    header[0] = 1
    header[1] = tensor.dim()
    header[2] = dtype_code
    if tensor.dim() > 0:
        header[3: 3 + tensor.dim()] = torch.tensor(
            list(tensor.shape),
            device=device,
            dtype=torch.int64,
        )
    return header


def _send_optional_tensor(
    tensor: Optional[torch.Tensor],
    *,
    dst: int,
    device: torch.device,
    group=None,
) -> None:
    header = _optional_tensor_header(tensor, device=device)
    dist.send(header, dst=dst, group=group)
    if tensor is not None:
        if tensor.device != device:
            tensor = tensor.to(device=device)
        dist.send(tensor.contiguous(), dst=dst, group=group)


def _recv_optional_tensor(
    *,
    src: int,
    device: torch.device,
    group=None,
) -> Optional[torch.Tensor]:
    header = torch.empty((_OPTIONAL_TENSOR_HEADER_SIZE,), device=device, dtype=torch.int64)
    dist.recv(header, src=src, group=group)
    if int(header[0].cpu().item()) == 0:
        return None

    ndim = int(header[1].cpu().item())
    dtype_code = int(header[2].cpu().item())
    dtype = _CODE_TO_DTYPE[dtype_code]
    shape = tuple(int(dim) for dim in header[3: 3 + ndim].cpu().tolist())
    tensor = torch.empty(shape, device=device, dtype=dtype)
    dist.recv(tensor, src=src, group=group)
    return tensor


def _send_object(obj, *, dst: int, device: torch.device, group=None) -> None:
    payload = pickle.dumps(obj)
    payload_len = torch.tensor([len(payload)], device=device, dtype=torch.int64)
    dist.send(payload_len, dst=dst, group=group)
    if payload:
        payload_tensor = torch.tensor(list(payload), device=device, dtype=torch.uint8)
        dist.send(payload_tensor, dst=dst, group=group)


def _recv_object(*, src: int, device: torch.device, group=None):
    payload_len = torch.empty(1, device=device, dtype=torch.int64)
    dist.recv(payload_len, src=src, group=group)
    size = int(payload_len.cpu().item())
    if size == 0:
        return None
    payload_tensor = torch.empty(size, device=device, dtype=torch.uint8)
    dist.recv(payload_tensor, src=src, group=group)
    return pickle.loads(payload_tensor.cpu().numpy().tobytes())


def _split_dict_by_batch(
    tensor_dict: Dict[str, Optional[torch.Tensor]],
    batch_sizes: List[int],
) -> List[Dict[str, Optional[torch.Tensor]]]:
    outputs = [{key: None for key in tensor_dict} for _ in batch_sizes]
    start = 0
    for idx, batch_size in enumerate(batch_sizes):
        end = start + batch_size
        for key, tensor in tensor_dict.items():
            outputs[idx][key] = None if tensor is None else tensor[start:end]
        start = end
    return outputs


def _cat_optional_tensors(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        return None
    if len(present) != len(tensors):
        raise ValueError("Teacher RPC received a mixed optional tensor batch")
    return torch.cat(present, dim=0)


def _merge_real_score_requests(requests: List[Dict[str, Optional[torch.Tensor]]]) -> Dict[str, Optional[torch.Tensor]]:
    merged: Dict[str, Optional[torch.Tensor]] = {}
    for key in _REAL_SCORE_REQUEST_KEYS:
        merged[key] = _cat_optional_tensors([request[key] for request in requests])
    return merged


class RemoteTeacherClient:
    def __init__(self, topology: DistributedTopology, device: torch.device):
        if not topology.is_worker:
            raise ValueError("RemoteTeacherClient can only be constructed on worker ranks")
        if topology.teacher_peer_rank is None:
            raise ValueError("Worker rank is missing a paired teacher rank")
        self.topology = topology
        self.device = device

    def request_conditioning(
        self,
        *,
        text_prompts: List[str],
        negative_prompt: str,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        dst = self.topology.teacher_peer_rank
        cpu_device = torch.device("cpu")
        control_pg = self.topology.control_process_group
        _send_object(
            {
                "text_prompts": list(text_prompts),
                "negative_prompt": negative_prompt,
            },
            dst=dst,
            device=cpu_device,
            group=control_pg,
        )
        response = {
            key: _recv_optional_tensor(src=dst, device=cpu_device, group=control_pg)
            for key in _CONDITIONING_RESPONSE_KEYS
        }
        conditional_dict = {
            "video_context": response["conditional_video_context"].to(device=self.device),
            "audio_context": response["conditional_audio_context"].to(device=self.device),
            "attention_mask": None if response["conditional_attention_mask"] is None else response["conditional_attention_mask"].to(device=self.device),
        }
        unconditional_dict = {
            "video_context": response["unconditional_video_context"].to(device=self.device),
            "audio_context": response["unconditional_audio_context"].to(device=self.device),
            "attention_mask": None if response["unconditional_attention_mask"] is None else response["unconditional_attention_mask"].to(device=self.device),
        }
        return conditional_dict, unconditional_dict

    def run_real_score(
        self,
        *,
        noisy_video: torch.Tensor,
        noisy_audio: Optional[torch.Tensor],
        video_sigma: torch.Tensor,
        audio_sigma: Optional[torch.Tensor],
        conditional_dict: Dict[str, torch.Tensor],
        unconditional_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        dst = self.topology.teacher_peer_rank
        request = {
            "noisy_video": noisy_video,
            "noisy_audio": noisy_audio,
            "video_sigma": video_sigma,
            "audio_sigma": audio_sigma,
            "conditional_video_context": conditional_dict["video_context"],
            "conditional_audio_context": conditional_dict["audio_context"],
            "conditional_attention_mask": conditional_dict.get("attention_mask"),
            "unconditional_video_context": unconditional_dict["video_context"],
            "unconditional_audio_context": unconditional_dict["audio_context"],
            "unconditional_attention_mask": unconditional_dict.get("attention_mask"),
        }

        for key in _REAL_SCORE_REQUEST_KEYS:
            _send_optional_tensor(request[key], dst=dst, device=self.device)

        response = {
            key: _recv_optional_tensor(src=dst, device=self.device)
            for key in _REAL_SCORE_RESPONSE_KEYS
        }

        return (
            response["pred_real_cond_video"],
            response["pred_real_cond_audio"],
            response["pred_real_uncond_video"],
            response["pred_real_uncond_audio"],
        )


class RemoteTeacherService:
    def __init__(
        self,
        *,
        topology: DistributedTopology,
        device: torch.device,
        real_score_module,
        text_encoder_module,
    ):
        if not topology.is_teacher:
            raise ValueError("RemoteTeacherService can only be constructed on teacher ranks")
        self.topology = topology
        self.device = device
        self.real_score = real_score_module
        self.text_encoder = text_encoder_module

    @torch.no_grad()
    def _run_conditioning_round(self) -> None:
        source_ranks = self.topology.teacher_service_source_ranks
        cpu_device = torch.device("cpu")
        control_pg = self.topology.control_process_group
        prompt_requests = [
            _recv_object(src=src, device=cpu_device, group=control_pg)
            for src in source_ranks
        ]

        merged_prompts: List[str] = []
        merged_negative_prompts: List[str] = []
        batch_sizes: List[int] = []
        for request in prompt_requests:
            prompts = list(request["text_prompts"])
            negative_prompt = str(request["negative_prompt"])
            batch_sizes.append(len(prompts))
            merged_prompts.extend(prompts)
            merged_negative_prompts.extend([negative_prompt] * len(prompts))

        conditional_dict = self.text_encoder(text_prompts=merged_prompts)
        unconditional_dict = self.text_encoder(text_prompts=merged_negative_prompts)
        response = {
            "conditional_video_context": conditional_dict["video_context"],
            "conditional_audio_context": conditional_dict["audio_context"],
            "conditional_attention_mask": conditional_dict.get("attention_mask"),
            "unconditional_video_context": unconditional_dict["video_context"],
            "unconditional_audio_context": unconditional_dict["audio_context"],
            "unconditional_attention_mask": unconditional_dict.get("attention_mask"),
        }
        split_responses = _split_dict_by_batch(response, batch_sizes)

        for src, payload in zip(source_ranks, split_responses):
            for key in _CONDITIONING_RESPONSE_KEYS:
                _send_optional_tensor(payload[key], dst=src, device=cpu_device, group=control_pg)

    @torch.no_grad()
    def _run_real_score_round(self) -> None:
        requests = []
        batch_sizes = []
        source_ranks = self.topology.teacher_service_source_ranks
        for src in source_ranks:
            request = {
                key: _recv_optional_tensor(src=src, device=self.device)
                for key in _REAL_SCORE_REQUEST_KEYS
            }
            requests.append(request)
            batch_sizes.append(int(request["noisy_video"].shape[0]))

        merged = _merge_real_score_requests(requests)
        conditional_dict = {
            "video_context": merged["conditional_video_context"],
            "audio_context": merged["conditional_audio_context"],
            "attention_mask": merged["conditional_attention_mask"],
        }
        unconditional_dict = {
            "video_context": merged["unconditional_video_context"],
            "audio_context": merged["unconditional_audio_context"],
            "attention_mask": merged["unconditional_attention_mask"],
        }

        pred_real_cond_video, pred_real_cond_audio = self.real_score(
            noisy_image_or_video=merged["noisy_video"],
            conditional_dict=conditional_dict,
            timestep=merged["video_sigma"],
            noisy_audio=merged["noisy_audio"],
            audio_timestep=merged["audio_sigma"],
        )
        pred_real_uncond_video, pred_real_uncond_audio = self.real_score(
            noisy_image_or_video=merged["noisy_video"],
            conditional_dict=unconditional_dict,
            timestep=merged["video_sigma"],
            noisy_audio=merged["noisy_audio"],
            audio_timestep=merged["audio_sigma"],
        )

        response = {
            "pred_real_cond_video": pred_real_cond_video,
            "pred_real_cond_audio": pred_real_cond_audio,
            "pred_real_uncond_video": pred_real_uncond_video,
            "pred_real_uncond_audio": pred_real_uncond_audio,
        }
        split_responses = _split_dict_by_batch(response, batch_sizes)

        for src, payload in zip(source_ranks, split_responses):
            for key in _REAL_SCORE_RESPONSE_KEYS:
                _send_optional_tensor(payload[key], dst=src, device=self.device)

    @torch.no_grad()
    def run_step(self, *, train_generator: bool) -> None:
        self._run_conditioning_round()
        if train_generator:
            self._run_real_score_round()
