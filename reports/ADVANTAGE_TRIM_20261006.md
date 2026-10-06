# 原始 |adv| 前 50% 样本筛选（2026-10-06）

## 状态

已实现并排队，当前训练尚未启用筛选。冻结任务为
`runs/v6_cluster8_adv_top50_v1`，队列 ID `01791268835472576477-69492c095c9842cc82c3140465bc1c6c`。
当前训练段相对 51000 / 全局 77554 完成后，从完整断点迁移；在记录时的
全局 76604，按当前普通/长历史混合耗时，预计约
12.2 小时到达边界，另需八卡编译和验收。

## 实现

- 配置 `ppo.adv_keep_fraction=0.5`；省略或 `1.0` 保持完整样本路径。
- 完整轨迹先计算同座位 GAE、return 及全局 advantage 归一化；排序使用
  归一化前的原始 |adv|，正负两侧同等参与。
- 在全部八个 rank 的有效 learner 决策中，精确保留 floor(N / 2)。
  未解决的 GAE 尾部和非 learner 决策不参与排序；并列时按 rank/本地顺序处理。
- 四次 256 桶直方图归约找到全局阈值，不搬运或聚集大型 observation。
- 选中样本打乱并前置；跨卡都为空的尾部 minibatch 跳过模型和梯度计算。
  局部不足处 mask 补齐，各卡采用正确的全局样本数量归一化梯度。
- policy、value、可选 belief loss 及自适应 vf 诊断均使用筛选结果。
  EV 使用完整 rollout；EMA 每个已完成 rollout 只推进一次，包括 KL 停止轮。
- 保留模型/Adam/坐标年龄/环境/RNG/vf/EMA，保持每轮 4194304 个新决策、
  gamma 1、lambda 0.95、lr 1e-5、每 100 步保存和总目标 200000。

## 已验证

23 项 CPU 工程测试通过（132.44 秒），另有 2 项单卡实验配置兼容测试通过
（16.08 秒）。包含八个 CPU 设备的集体计算、单一全局 batch 梯度对照、
不均匀有效掩码、空 rank、KL/非有限数更新拒绝、空样本、奇数样本、并列值、
完整 EV、真实跳过反向调用，以及完整续训状态迁移约束。
这些 CPU 设备不是实际八卡训练的证据。

生产规模筛选检查使用 4194304 个随机原始 advantage：4004988 个有效样本中
恰好保留 2002494 个；保留样本最小 |adv| 为 0.674685359，丢弃样本最大
|adv| 为 0.674684942。CPU 热执行 0.255 秒，仅说明工程开销，不能当作 CUDA 耗时。

## 待验证

提交任务在实际远端八卡完成 8 轮后才允许接管，要求真实 nranks=8/NCCL、
精确全局 50% 样本数、选中样本全部完成更新、有限值和合法动作、raw/EMA
副本同步及显存余量。原续训任务在验收通过前保留，失败不自动重试。

筛选使 PPO 的更新样本分布偏向较大 advantage，没有补偿这种选择偏差。
目前没有真实 CUDA 加速或实力提升的结论；后续通过分历史耗时、完整 EV/KL
及持续的每 500 步 DouZero BEST 全角色期望得分评测判断实际效果。

## 证据

- `/mnt/pfs/guoyuchong/guoyuchong/ddz/runs/advantage_trim_audit_20261006/TESTS_VERIFICATION.json`
- `/mnt/pfs/guoyuchong/guoyuchong/ddz/runs/advantage_trim_audit_20261006/production_scale_selection.json`
- `/mnt/pfs/guoyuchong/guoyuchong/ddz/runs/v6_cluster8_adv_top50_v1/READY.json`、`REQUEST.json`、`SUBMISSION.json`
- 实际八卡验收将在 `/mnt/pfs/guoyuchong/guoyuchong/ddz/runs/v6_cluster8_adv_top50_v1/engineering_READY.json` 和 `engineering.log` 发布。
