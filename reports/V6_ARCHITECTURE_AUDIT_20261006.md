# V6 架构与启动链路审计（2026-10-06）

## 当前安排

模型为 8,014,192 参数；本次修复围绕计算、断点恢复、队列接管和消费者。
当前 V5 八卡训练继续。修订版冻结目录为
`runs/v6_cluster8_attention_8m_v4`，FIFO 队列项为
`01791261485793277820-a18907712e714ca886d64606d966aa10`。
它在 V5 relative 50600 / global 77154 的完整保存边界后执行。
V3 未启动，已转入 interrupted 并保留 SUPERSEDED 凭证；V1/V2 保持历史状态。

本轮用户随后要求暂停 QOJ，上线进程已执行停止，启动入口加入 OFFLINE 检查。
共享工作区之后被另一任务重新启动，AGENTS 又记录用户要求持续对战；
该跨任务状态冲突已向用户提出澄清。热更新修复源码已完成，实际上线状态
须查看最新用户指令及运行凭证，不能将早先停止凭证当作当前状态。

## 修复与加固

| 路径 | 问题/缺口 | 本次处理 |
|---|---|---|
| cuDNN 注意力 | 用 Q/K 等长推断历史自注意力；15 个事件加 null 恰好与 16 个状态 token 等长 | 明确 query 语义，cross 保留所有状态 query；history 使用实际前缀长度 |
| 完整环境恢复 | 重置 65,536 副新牌只为构造模板后丢弃；缺少严格 schema 校验 | 直接恢复已有状态，检查全字段 shape/dtype、历史长度、座位及 8 份 RNG |
| Adam 年龄 | 保存年龄映射可能被广播或包含未来年龄 | 验证键集合、标量/逐坐标形状、整数类型和年龄范围；冷恢复也验证 |
| 迁移配置 | 部分固定字段清单可能遗漏未训练的 auxiliary/value 开关 | 除批准的 attention/depth 字段外全部固定；共享 Efficient 模型 schema |
| CUDA 启动 | 8 轮短样本 rollout 未必覆盖满历史反向路径 | 实际 GPU 提前执行 4096×88、1024×192 的完整 8M 前向/反向 |
| 保存发布 | 迁移大文件直接写入 | full/raw/EMA 写入、fsync、原子替换，config 完成后发布 latest 标记 |
| 训练接管 | 批量取消旧后续项后才检查数量；来源身份缺少绑定 | 先验证唯一后续项及 step/SHA，再取消；来源 proof 绑定确切 queue id |
| 续训验收 | 只检查末尾 step；遗漏中间轮次、EMA 和非有限数值 | 检查连续新增轮次、全部数值、policy/EMA 副本，拒绝零新增轮次 |
| 热更新 | 准备与 serving 架构切换重叠；报告回调失败可能影响决策 | 固定准备对象，提交前检查 signature；持久化后的报告失败单独记录 |
| 脚本入口 | scripts 的 package import 依赖外部 PYTHONPATH | prepare/engineering 入口自行加入 repo 路径 |

参数模板初始化使用 4 个历史位置，参数形状与完整历史相同，降低启动成本。
新增输出投影继续零初始化；首次输出更新后，新 Q/V 内部参数也获得梯度。

## 已完成的证据

- **58 项 CPU 测试通过。** 覆盖 warm start、完整合法动作、历史 0/2/89/192、
  隐藏 padding、时序、PPO/EMA，以及损坏断点、队列 claim、热更新和停用保护。
- **真实完整断点。** relative 49900 / global 76454；保留 65,536 个环境、
  8×2 RNG、Adam step 5,174,079、EMA 10,900 次 rollout / decay 0.999。
  权重、动量、年龄、arena、EMA 和全环境序列化后冷恢复一致；保存年龄校验路径也通过。
- **128 个实际局面。** raw/EMA 的 CPU FP32 和 A100 CUDA BF16 的 body KL、
  完整 greedy 动作差异、选中完整动作 logp 差异、value 差异均为 0。
- **两种实际 CUDA 反向尺寸。** 历史 88 / batch 4096、历史 192 / batch 1024
  均有限且梯度非零；进程峰值 26,399,646,976 bytes，约 24.59 GiB。
  工程局部损失的梯度范数不能解释为 PPO 对局质量。
- **10 种 CUDA mask 对照。** 2/4/6/8/10 heads 的 history/self 和 state/cross；
  标准注意力对照最大 BF16 差异 0.015625，包含 Q=K=16 的情况。
- **8 个 CPU 工程 rank。** 增长后的真实模型执行 PPO、Adam、EMA 和冷恢复；
  更新与单个全局 batch 对照通过，副本完全一致。CPU 测试不作为 NCCL 验收。

### 本地失败尝试与编译警告

第一次反向测试限额为本地 allocator 的 30%，单个临时 buffer 请求
25,473,057,216 bytes，触及额度；失败日志保留，控制器恢复空转。
保留约 49 GiB 的其它停止进程显存后，显式第二次工程测试将限额调至 37.5%，
两种完整 batch 均通过。

两条 GEMM autotuning 数值警告已检查：JAX 0.4.38 对应的 XLA 固定版本
将不匹配候选标记为 DISQUALIFIED。最终执行的前向保持、mask 对照和反向 finite
检查通过，编译精度开关保持原配置。
[固定版本的剔除代码](https://github.com/openxla/xla/blob/20a482597b7dd3067b26ca382b88084ee5a21cf7/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc#L1076)。

## 正式验收与边界

V4 READY 冻结代码、config、CPU/tests、完整恢复和本地 CUDA 证明。
队列项仍要对边界时真正的最新完整断点重新核验 CPU FP32 和 CUDA BF16，
执行两种 CUDA 反向签名，再完成 8 轮真实八卡 PPO。
nranks=8、NCCL、合法动作、finite、policy/EMA 同步、EMA 轮数及显存余量全部
通过，才取消旧 V5 后续项并更新生产指针。失败保留证据和 V5 续训项。

本次没有 V6 对局实力或真实八卡吞吐结果。上述显存是模型反向的单卡工程峰值，
完整 rollout+Adam 峰值仍由远端验收。本地 A100 已恢复单一控制器和空闲占满，
每 500 步 DouZero BEST 评测继续。

证据目录为 `runs/v6_audit_20261006_v1/`，公开 JSON 索引为
`reports/V6_ARCHITECTURE_AUDIT_20261006.json`。私有手牌仅保存在被 git 忽略的运行目录。
