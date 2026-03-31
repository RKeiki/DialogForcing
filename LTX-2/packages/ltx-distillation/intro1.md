Stage 1 的 distillation 不是“拿 teacher 输出直接监督 student”的普通回归，而是标准 DMD 思路：

  1. 先让 student 生成自己的样本。
  2. 在这些样本上比较 fake_score 和冻结 teacher real_score 的输出差。
  3. 用这个差构造 generator 的分布匹配梯度。
  4. 同时单独训练 fake_score 去拟合“student 样本分布上的去噪/flow”。

  1. 启动与 Stage1 配置
  [train_stage1_bidirectional_dmd.sh](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/scripts/
  train_stage1_bidirectional_dmd.sh):91 只是做分布式环境整理，然后 torchrun -m
  ltx_distillation.train_distillation。

  Stage1 配置在 [stage1_bidirectional_dmd.yaml](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-
  distillation/configs/stage1_bidirectional_dmd.yaml):4，关键点是：

  - 目标是把 LTX-2 从 1000-step 蒸馏成一个 few-step bidirectional 模型。
  - few-step 调度是 denoising_step_list = [1000, 757, 522, 0]，见 [stage1_bidirectional_dmd.yaml](/home/
    liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/configs/stage1_bidirectional_dmd.yaml):30
  - generator、real_score、fake_score 都是 bidirectional_av，见 [stage1_bidirectional_dmd.yaml](/home/liujingqi/
    wanAR/OmniForcing/LTX-2/packages/ltx-distillation/configs/stage1_bidirectional_dmd.yaml):41
  - teacher real_score 冻结，student generator 和 critic fake_score 训练，见 [stage1_bidirectional_dmd.yaml](/
    home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/configs/stage1_bidirectional_dmd.yaml):20
  - Stage1 用的是 backward_simulation: true，也就是只吃文本 prompt，不吃真实 ODE latent 数据，见
    [stage1_bidirectional_dmd.yaml](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/configs/
    stage1_bidirectional_dmd.yaml):73
  - dfake_gen_update_ratio: 5，即 generator 5 步更新一次，critic 每步都更新，见 [stage1_bidirectional_dmd.yaml]
    (/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/configs/stage1_bidirectional_dmd.yaml):62

  2. 模型角色
  LTX2DMD 里明确有三套 diffusion wrapper，见 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-
  distillation/src/ltx_distillation/dmd.py):46：

  - generator: student
  - real_score: 冻结 teacher
  - fake_score: critic

  三者都是从同一个 LTX-2 checkpoint 初始化出来的，随后按配置决定谁参与训练，见 [dmd.py](/home/liujingqi/wanAR/
  OmniForcing/LTX-2/packages/ltx-distillation/src/ltx_distillation/dmd.py):482 和 [dmd.py](/home/liujingqi/
  wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/ltx_distillation/dmd.py):513。

  另外，few-step 的 sigma 不是直接拿 t/1000，而是先用 LTX2 scheduler 生成一条更细的 40-step sigma 轨迹，再把
  [1000,757,522,0] 映射到最近的 sigma 上，保证和 ODE 轨迹一致，见 [dmd.py](/home/liujingqi/wanAR/OmniForcing/
  LTX-2/packages/ltx-distillation/src/ltx_distillation/dmd.py):178。加噪公式是 flow-matching 形式：
  x_t = (1 - sigma) * x_0 + sigma * eps，见 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-
  distillation/src/ltx_distillation/dmd.py):263。

  3. 每个训练 step 的主流程
  Trainer 的主循环在 [train_distillation.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/
  src/ltx_distillation/train_distillation.py):948，每步核心逻辑在 [train_distillation.py](/home/liujingqi/wanAR/
  OmniForcing/LTX-2/packages/ltx-distillation/src/ltx_distillation/train_distillation.py):491。

  Stage1 每步大致是：

  1. 从 prompt 文件读一批文本，因为 backward_simulation=true，所以没有真实视频/音频 latent，见
     [train_distillation.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/
     ltx_distillation/train_distillation.py):324 和 [train_distillation.py](/home/liujingqi/wanAR/OmniForcing/
     LTX-2/packages/ltx-distillation/src/ltx_distillation/train_distillation.py):534
  2. 编码条件文本，同时缓存一份 negative prompt 的 unconditional embedding 供 CFG 用，见 [train_distillation.py]
     (/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/ltx_distillation/
     train_distillation.py):549
  3. 如果当前 step 满足 step % 5 == 0，更新 generator；critic 则每步都更新，见 [train_distillation.py](/home/
     liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/ltx_distillation/train_distillation.py):510

  这里 latent shape 由 num_frames=121, 512x768 算出来；视频 latent 是 16 帧，音频 latent 大约 126 帧，逻辑在
  [train_distillation.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/
  ltx_distillation/train_distillation.py):33。

  4. Generator 的 distillation 流程
  入口在 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/ltx_distillation/
  dmd.py):1907。

  先看“student 样本怎么来”：

  - _run_generator() 会先做 backward simulation，见 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/
    ltx-distillation/src/ltx_distillation/dmd.py):1764
  - backward simulation 调的是 BidirectionalAVTrajectoryPipeline.inference_with_trajectory()，它从纯噪声开始，按
    当前 few-step generator 一步步预测 x0，再 re-noise 到下一个 sigma，保存整条轨迹，见
    [bidirectional_pipeline.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/
    ltx_distillation/inference/bidirectional_pipeline.py):43
  - 所以 Stage1 训练输入不是 teacher 轨迹，而是“当前 student 自己 rollout 出来的轨迹状态”

  然后：

  - 从这条 few-step 轨迹里随机抽一个 step 的 noisy latent 作为 student 的训练输入，Stage1 bidirectional 模式下整
    个样本的视频帧和音频帧共享同一个 step，见 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-
    distillation/src/ltx_distillation/dmd.py):1856
  - generator 对该 noisy latent 预测 clean x0，见 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-
    distillation/src/ltx_distillation/dmd.py):1897

  接着进入真正的 DMD loss，见 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/
  ltx_distillation/dmd.py):1592：

  - 把 generator 预测出的 x0 当成当前“生成样本”
  - 再随机采一个完整训练时刻 t in [0.02T, 0.98T]，转成 sigma，给这个生成样本重新加噪，见 [dmd.py](/home/
    liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/ltx_distillation/dmd.py):1619
  - 用 fake_score 预测一次，用 real_score 分别在 cond/uncond 下预测两次，再做 CFG 融合，见 [dmd.py](/home/
    liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/ltx_distillation/dmd.py):1412
  - 定义 DMD 梯度：
    grad = pred_fake - pred_real_cfg
    见 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/ltx_distillation/
    dmd.py):1449
  - 再按 teacher 残差尺度做归一化，见 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-
    distillation/src/ltx_distillation/dmd.py):1453
  - 最终 loss 不是直接回归 teacher 输出，而是构造一个 surrogate：
    本质上等价于对 grad 做 0.5 * ||grad||^2 的优化，从而让 generator 朝 teacher 分布方向移动，见 [dmd.py](/home/
    liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/ltx_distillation/dmd.py):1685

  这就是 Stage1 distillation 最核心的点：
  不是 student(x_t) ≈ teacher(x_t) 的直接蒸馏，而是用 fake_score - real_score 近似 KL gradient，逼 student 的生
  成分布向 teacher 靠拢。

  5. Critic 的训练流程
  入口在 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/ltx_distillation/
  dmd.py):1952。

  流程是：

  1. 先 no_grad 跑一次当前 generator，得到 generated sample，见 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-
     2/packages/ltx-distillation/src/ltx_distillation/dmd.py):1966
  2. 再随机采一个 critic timestep，给 generated sample 加噪，见 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-
     2/packages/ltx-distillation/src/ltx_distillation/dmd.py):1980
  3. fake_score 去预测去噪结果，见 [dmd.py](/home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/
     src/ltx_distillation/dmd.py):2023
  4. 用 flow_pred = (x_t - x0_pred) / sigma 做 velocity / flow matching loss，而不是简单 x0 MSE，见 [dmd.py](/
     home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/src/ltx_distillation/dmd.py):2032

  所以 critic 学到的是：
  “在 student 当前生成分布上，如何把 noisy sample 去噪/估计 flow”。

  6. 一句话概括
  Stage1 的算法可以概括成：

  - 用当前 bidirectional student 自己做 few-step rollout，制造训练时会遇到的 noisy state。
  - student 从这些 state 预测 x0。
  - teacher real_score 提供目标分布方向，critic fake_score 估计 student 分布方向。
  - 用两者差值构造 DMD gradient 更新 student。
  - 同时训练 critic 在 student 样本上做 flow matching。
  - 交替优化后，得到一个 few-step bidirectional DMD 模型，后续再拿它当 teacher/critic 去做 causal 阶段。