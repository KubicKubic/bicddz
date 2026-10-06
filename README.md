# 斗地主：JAX / PPO / 完整公开历史 Transformer

## 当前发布与八卡续训

`models/release.json` 标明此次发布的 raw、EMA 和完整续训 checkpoint 的 step、
SHA-256、EMA rollout 计数及原始训练配置。三个 `.msgpack` 使用 Git LFS；克隆后
先运行 `git lfs install --local`，再运行 `git lfs pull`。上线默认使用 raw policy，EMA 单独保留。

### QOJ 自动跟随 checkpoint

`scripts/qoj_match.sh prepare` 默认启用 `ddz/qoj_checkpoint_watch.py`，每 5 秒
读取 `runs/production_current.json` 指向的训练目录。训练原子发布 `latest.json`
后，后台加载对应 raw policy，检查参数结构、形状、dtype、有限值和模型/GAE
兼容性，并预热叫分与出牌前向。游戏线程在下一次请求或决策前切换权重；
JIT 将权重作为输入，不重启客户端、不重新匹配、不读取大型完整续训文件。
加载失败保持原模型，记录 `MODEL_UPDATE_REJECTED`；成功记录
`MODEL_PREPARED` / `MODEL_ACTIVATED` 及前后 step、SHA-256、加载耗时。

`runs/qoj_match_v1/active_model.json` 保存实际启用的 checkpoint；重启后验证
并恢复它，损坏时回退冻结发布版本。`deployment.json`、实时面板、每条决策
记录都会标明实际模型 step。切换训练目录后继续按 global step 单调更新。
`DDZ_CHECKPOINT_POINTER` 可指定其他训练指针。已注册的 V5→V6 架构升级在后台
构建、验证并预热新的 Policy，游戏线程再原子替换，重启后仍可恢复 V6。
Git 的 `models/` 发布包仍是版本固定的发布制品；实时更新直接读取训练快照。

八卡生产训练从 raw 47600 / global 74154 的完整断点立即切换。
`configs/cluster8_env2x_batch2x_v1.json` 是用户要求的新配置：65536 环境、
全局 minibatch 32768，每卡分别 8192 环境和 4096 minibatch，每轮 4194304 个
新决策。horizon 64、总目标 200000 global updates、lr 1e-5、gamma 1、lambda
0.95、同座位 GAE、自适应 value 系数、每 100 步保存均保留。
`ddz.resize_distributed` 只在已验证的八卡队列边界迁移；保留原有每个 rank 的
环境、Adam、RNG、EMA 权重及 rollout 计数，新增环境使用独立随机流。先执行
8 轮实际八卡健康验证，再长期续训。八卡配置、入队凭证及状态详见
[本次操作记录](reports/RELEASE_AND_SCALE_20261005.md)。
用户要求的立即切换状态和验收记录见
[立即切换记录](reports/RESIZE_IMMEDIATE_20261005.json)。

## V6：约 8M 的注意力扩容

参数量 **8,014,192**。状态交互从 4 层增至 6 层，注意力总宽度从
192 增至 320；公开历史编码从 3 层增至 4 层，注意力总宽度增至 256。
新增层插入前部：状态执行顺序 `0,4,1,5,2,3`，历史执行顺序 `0,3,1,2`。
旧残差通道保持 192，旧 FFN、动作头和完整单步动作空间保持当前尺寸。
继承层通过独立注意力头组增宽，保留旧投影、每头 32 维及温度；所有新增
输出投影初始化为零。公开事件、位置、变长历史及完整注意力继续使用。

配置为 `configs/cluster8_attention_v6_8m.json`。迁移保留旧权重和 Adam
坐标、环境、RNG、value 系数、EMA 权重与 rollout 计数；新参数使用自己的
Adam 年龄。PPO 和 200000 global updates 总目标继续使用原配置。
切换任务先执行 CPU 功能核验、真实 CUDA 功能核验和
8 轮八卡 NCCL/PPO/显存检查，通过才更新生产指针；失败保留原 V5 后续任务。

本地 A100 控制器已支持先完成 V5 评测，再自动跟随已验证的 V6 目录，继续
每 500 步、65536 副牌 / 393216 局的全角色 DouZero BEST 评测和空闲占满。
QOJ 客户端提前部署支持 V6 的代码，模型架构切换时后台预热并继续当前比赛。
工程测试不等于实力提升；最新记录见
[扩容报告](reports/V6_ATTENTION_SCALING_20261006.md)。

后续审计修订冻结在 `runs/v6_cluster8_attention_8m_v4`：58 项 CPU 工程测试、
真实完整断点迁移与冷恢复、两种生产 CUDA 反向尺寸均已通过。128 个实际局面
的 raw/EMA CPU FP32、CUDA BF16 输出均保持一致；本地反向显存峰值约 24.6 GiB。
正式八卡验收通过后才接管生产，详见
[架构审计报告](reports/V6_ARCHITECTURE_AUDIT_20261006.md)。

用户随后要求立即切换。新的冻结任务为
`runs/v6_cluster8_attention_8m_now_v3`，从完整断点相对 50000 / 全局 76554
迁移；原 50600 边界任务已取消并保留记录。未保存的 59 轮日志单独保留，
恢复状态不计入这些轮次。实际八卡验收与接管结果见
[提前切换报告](reports/V6_IMMEDIATE_SWITCH_20261006.md)。
首次候选因一轮 KL 提前停止而未获接管；修订版为新增 Adam 坐标加入
1024 次优化器更新的 warmup，保存该设置并检查恢复一致性，原参数保留成熟
Adam 年龄。迁移还等待实际旧训练子进程退出后再检查显存和启动新任务。

## V5：更深的注意力与动作交互

入口为 `ddz.train_v5`、`configs/a100_v5.json` 和 `ddz.api_v5`，运行目录为
`runs/a100_interaction_complete_move_v5`。模型 **5,097,200 参数**，有 **3 层完整历史
self attention**，以及 **4 层当前状态处理**，每层含历史 cross attention 和状态
self attention。保留当前牌局全部出牌/空过事件、完整公开快照、时序嵌入、变长
掩码、自身角色、底牌信息；192 事件容量覆盖合法牌局的完整出牌历史。

合法主体直接查询 15 个经过历史与状态交互的牌点 token；显式输入出牌后的剩余
手牌。带牌偏好通过轻量条件网络依赖已选带牌和假设下一选择后的剩余手牌，
不再只变化合法掩码。value 也读取合法动作集合的特征摘要。主体/带牌组成完整
合法动作树，沿用单次模型前向、单次公开出牌、单条 PPO transition、joint logp。
为控制计算量，历史 FFN 宽度为 512、当前状态 FFN 宽度为 1536，将更多参数放在
16 个当前状态 token 上。其余批量与 PPO 协议保持 V4 配置。

从 V4 step 39811 迁移全部原有参数；新增残差输出零初始化，扩展 FFN 保留旧子矩阵。
Adam moments、优化器计数、余弦学习率进度、1024 个环境、RNG、熵系数进度及自适应
value 权重均继续；V5 新增参数/FFN 扩展坐标的 Adam bias correction 用自己的
新增年龄，原有坐标保留 V4 全部优化器计数。V5 自身更新编号重新从 0 记，
保存仍每 100 步。首次运行的 Adam 年龄审计保留在
`runs/a100_interaction_complete_move_v5_initial_adam_audit/`。单元测试覆盖
FP32 函数保持、梯度活性、完整动作概率归一化、条件带牌交互、精确续训、API 与全角色评测。
标准注意力后端的 BF16 迁移中 128 个实际状态的 greedy 出牌全部一致。生产选择
cuDNN full attention；与 FP32 注意力参考的最大绝对误差小于 0.005，补齐位置
污染不影响输出，反向梯度有限。不同 BF16 实现存在舍入差异：cuDNN 后端在这
128 个状态中有 2 个 greedy 动作变化，不能称为逐位等价迁移。
训练按整批实际最大历史长度选择独立编译的 88/192 事件分支，长历史使用 minibatch
1024、短历史保持 2048，避免 XLA 为普通批次预留长历史反向图内存，不截断历史。
A100 同配置测量和实际 BF16 迁移核验见 `runs/interaction_v5_matched_benchmark/`。
生产后端基准为 **22,863 新决策/秒**，rollout 0.741 秒、train 2.125 秒；普通注意力
后端为 11,664/秒，提升约 96%。比 2.36M 参数 V4 的 33,234/秒慢约 31%，这是
更深的历史注意力与新增动作交互的计算成本。计时不含编译、周期诊断、评估与保存。
更大容量不等于更强棋力，需要后续独立评测确认。

500-step 天梯入口为 `scripts/watch_interaction_v5_all_roles_ladder.sh`，每组 288 副
共同牌、3456 局，对战最近三代与 step 0，包含自然叫分和确保三个角色的固定叫分
评测。每个完整版本的对战完成后更新 PNG，避免等待整个积压队列才绘图。

### V5 可选随机探索

训练配置 `ppo.random_action_prob` 即参数 p，范围 `[0, 1]`；省略或为 0 时
关闭，保留原有动作采样、RNG 和条件熵计算路径。当前生产配置未开启。
`configs/a100_v5_explore_p001.json` 与生产配置仅相差 `p=0.01`。

每个完整动作只抽一次分支：以 p 的概率走随机分支，以 1-p 的概率按模型
概率采样。随机分支均匀选择合法主体，再依次均匀选择合法带牌；这不等于
所有完整组合均匀采样。叫分阶段同样在合法叫分中随机。压牌、空过、带牌
可完成性及规范顺序均使用已有精确规则，仍只产生一次环境提交。
整手牌的实际概率为 `q=(1-p)*pi+p*u`，使用 logaddexp 计算。
rollout、PPO ratio/KL/clip 和自适应 value 梯度诊断全部使用 q，p 在一个
配置内固定。启用时熵项采用 `E_qold[-(q/qold)*log(q)]` 的重要性采样估计，
保留权重梯度；只对采样的 `-log(q)` 求导会丢失正确的熵梯度。该估计存在
采样方差。训练目标是混合策略 q，去掉额外探索后的实力需要独立评测。
API、周期评估、全角色天梯和 DouZero 对照仍使用原有 greedy 策略。

从当前完整 V5 checkpoint 新建独立分支（命令仅供后续运行）：

```bash
PY=../.venv-gnn-jax/bin/python
"$PY" -m ddz.train_v5 \
  --config configs/a100_v5_explore_p001.json \
  --out runs/v5_explore_p001 \
  --fork-from-v5 runs/a100_interaction_complete_move_v5/latest.msgpack \
  --require-a100
```

`--fork-from-v5` 读取完整 checkpoint，保留参数、Adam moments/年龄修正、
学习率/熵进度、环境、自适应 value 权重、RNG 和原有 step 编号，只允许修改 p。
新目录在第一轮更新前保存继承状态，并记录源 checkpoint SHA-256 和 p 的变化。
不能与 `--resume` 或 V4 迁移选项同时使用。对照分支可使用原配置以同一份
冻结 checkpoint 启动 p=0；公平比较应先复制一次源 latest.msgpack 给两个分支。
恢复探索分支使用同配置与 `--resume`。`--stop-after` 是绝对 step 编号。
配置 p 写入 checkpoint，并逐步写入 metrics.jsonl；resume 不允许静默改变 p。

对应测试（CPU，不启动实验）：

```bash
JAX_PLATFORMS=cpu "$PY" -m pytest -q tests/test_v5_exploration.py
```

覆盖完整动作概率归一化、随机分布独立核验、整手牌分支采样频率、p=0/1、
叫分/压牌/空过、多带牌、混合概率与熵梯度、完整牌局合法性、PPO/诊断、
checkpoint 精确恢复，以及真实 CLI 的新建、分支和续训流程。

## 当前强基准：DouZero ResNet 2.0 best

`configs/douzero_reference.json` 指向 EdwardPooh/douzero-resnet-2.0 发布者说明的
最强已发布 `Douzero_Resnet/baseline/best`，commit
`85afd773abd01c411f543d6ade5b99a4fde327d2`。三个角色权重与模型/观察编码源文件
保留在 `runs/douzero_resnet_2_0_reference/`，严格载入并核验 SHA-256。
V4 复测入口为 `ddz.compare_douzero_resnet`，V5 为 `ddz.compare_douzero_v5_resnet`。
旧原版 ADP/WP 结果保留为历史证据。

V4 step 39811 的强基准复测：1024 副共同牌、6144 局、三角色互补交换控制，
无非法动作；镜像队伍胜率 **44.78% [43.41%, 46.14%]**，三角色等权期望积分
**−0.2769 [−0.3623, −0.1933]**。固定叫分 3，不测叫分实力；积分使用项目的
线性炸弹/春天规则。报告为
`runs/a100_compact_complete_move_v4/resnet2_best_step39811_all_roles_1024_v1/report.html`，
同目录保留 PNG、全部分片与逐副结果。这个结果属于 V4，不能用来声称 V5 更强。

## 此前 V4：吞吐优先的完整动作压缩

入口为 `ddz.train_v4`、`configs/a100_v4.json` 和 `ddz.api_v4`，运行目录为
`runs/a100_compact_complete_move_v4`。保留完整公开事件、变长历史全注意力、
时序嵌入、角色和底牌信息。主体加条件带牌构成压缩的合法动作树，覆盖原有全部
完整合法动作；一次模型前向给出主体和带牌参数，一手牌只有一条 transition 和
一个 joint logp。条件带牌选择只做掩码计算，不追加模型前向或公开事件。

主体描述编码在批次内共享，以 48 维的轻量 cross attention 查询合法主体集合；
不再对每个带牌组合运行三层候选解码器。新增策略支路零初始化，从 V2 最终策略
复制全部原有权重，保持初始 actor、带牌偏好和 value 输出。约 236 万参数。
整手牌合法性在设备上一次验证、一次提交；采样和 PPO 使用 `lax.scan/jit`，
无需每步将手牌取回 CPU 枚举候选。

1024 环境 × 64 步 = 65536 **新决策**；随机混合 minibatch 2048、PPO 一轮，
gamma=1、lambda=0.95、基础学习率 1e-4，每 10 步自适应 value 权重。
每 100 步保存与评估，继续记录 value explained variance。
A100 基准稳定轮次为约 3.32 万新决策/秒，数据见
`runs/compact_v4_benchmark/benchmark.json`；编译、诊断、评估和保存开销另行计量。

500-step 全角色综合天梯由 `scripts/watch_compact_v4_all_roles_ladder.sh` 持续更新，
使用每组 288 副共同牌、3456 场对局，对战最近三代及初始版本，口径与旧天梯一致。
V4 的 step 0 是迁移后的 V2 强策略，不能与 V3 的初始锚点评级直接相减。
144 副共同牌、1728 场直接对战确认初始版本与 V2 的逐副配对净分全部相同；
step 100 的自然叫分净分为 -0.41/局（95% 区间 -0.83 至 -0.01），因此吞吐提升
不应解读为续训棋力已提升。具体结果与含周期工作开销的吞吐见
`runs/a100_compact_complete_move_v4/performance_summary.json`。

## 此前 V3：完整合法出牌集合

训练入口为 `ddz.train_v3`，配置为 `configs/a100_v3.json`；推理入口为
`ddz.api_v3`。每次决策先枚举当前手牌中全部合法的**完整出牌解释**，跟牌时按上手牌的
牌型、长度、点数以及炸弹、王炸规则过滤。出牌主体与全部带牌一次选定，不再逐张选择
带牌。候选表有 28,238 种物理上可能的完整牌型解释；每个状态只取合法子集。批量计算
按实际最大候选数补齐到 2 的幂，空位编码为 pass 并在注意力和策略概率中屏蔽，绝不
截断合法动作。

模型保留 V2 的当前牌局完整出牌与空过时序、变长全注意力历史、公开状态和自身位置
编码。每个候选有整手牌、主体、带牌、牌型、点数、长度的编码；状态 token 跨注意力
查询候选集合，候选 token 再跨注意力读取当前状态。默认 3,321,412 参数，低于 4M
上限。rank 级动作映射到实体牌时，同点数优先使用已亮出的底牌牌 ID，因而优先打出
底牌花色。训练使用 `lambda=0.95`、`vf_coef=0.1`，每 100 次更新保存和评估，并记录
value explained variance。

V3 全角色综合天梯入口为 `ddz.score_ladder_graph_all_roles_v3`。每 500 次更新
取一个快照，分别与最近三个快照及 step 0 对战。每组 288 副共同牌；每副牌以
三个起始座位为焦点，每个座位都对调“单座位模型”和“其余两座位模型”，自然叫分
六局的原始积分期望值构成主曲线。另以固定叫分的六局确保地主、地主下家、门板
各角色都被单独测量；地主积分除以 2 后与两个农民角色等权合成出牌诊断。
每组共 3456 场，配对比分、共同牌 bootstrap 区间、PNG、HTML 和 CSV 独立保存
于 `runs/a100_candidate_set_v3/score_ladder_all_roles_500_v1`。旧版
`ddz.score_ladder_graph_v3` 的 576 场固定叫分队伍积分曲线保留作历史记录。
持续回填命令为 `bash scripts/watch_candidate_v3_all_roles_ladder.sh`；评测进程
固定使用后台 CPU 核，不占用训练用的 A100，且每 12 副牌缓存一次对局结果。

V3 与 V2 的直接比较使用 `bash scripts/compare_candidate_v3_800_vs_v2_final.sh`，
默认对战 V3 step 800 和 V2 最终 step 1572。可用 `DDZ_V3_COMPARE_STEP=900`
选择别的 V3 快照，用 `DDZ_V2_COMPARE_STEP=400` 选择早期 V2，或用
`DDZ_COMPARE_DEALS=288` 在同一目录追加共同牌样本。
自然叫分综合净分是主比较；固定叫分的三个角色分别报告，原始牌局与 95% 区间
保存在 `vs_v2_final_<step>_all_roles_v1` 等目录。V3 初始权重复制了 V2 最终版
中形状相同的参数，因此与 V2 早期快照的新增环境决策量相等不等于独立训练预算相等。

```bash
cd /mnt/pfs/guoyuchong/guoyuchong/ddz
../.venv-gnn-jax/bin/python -m ddz.train_v3 \
  --config configs/a100_v3.json --out runs/a100_candidate_set_v3 \
  --require-a100 --init-from-v2 runs/a100_complete_move_v2/policy_0001572.msgpack
```

运行中可用 `--resume --ppo-epochs 3` 增加同一批数据的 PPO 训练轮数；实际轮数与
KL 提前停更会逐步写入 `metrics.jsonl`。采样端按候选数分成 64 局小组的实测稳定
rollout 为约 14–15 秒，统一批量约 11–12 秒，故默认使用统一批量。

## 此前 V2：完整动作与变长出牌历史

V2 训练入口是 `ddz.train_v2`，配置为 `configs/a100_v2.json`，运行目录为
`runs/a100_complete_move_v2`。V1 代码和第 9510 步检查点保留，下面其余章节描述
V1 的训练流程和历史结果。

V2 每次决策提交一手完整牌，包括全部翼牌；一手牌只产生一条 PPO transition。
当前牌局的每次出牌和空过各占一个历史事件，按发生顺序保留。每条事件包含行动者、
完整出牌点数、牌型及主体信息、动作后的三家剩余张数、地主、叫分、炸弹数、空过数、
上手牌、各家出牌手数、下个行动者等全部规则相关公开状态。叫牌、亮底牌和重发牌
以当前状态向量表示，不占用出牌时间序列；重发牌会清空旧牌局的序列。

模型用 `hist_len` 屏蔽填充位置，支持 0 至 192 条有效事件的变长历史；每个有效事件
在历史自注意力层中可读取其他所有有效事件，再由当前手牌 token 跨注意力读取。
时序位置有可学习嵌入。当前玩家位置明确编码为地主、地主下家、地主上家（门板）
三类。历史行动者和其他座位字段按当前玩家视角旋转；对手暗牌仍不可见。

V2 的 A100 正式配置使用 512 环境 × 64 步、PPO minibatch 2048、`lambda=0.95`，
每 100 次更新保存和评估，记录 value explained variance。API 推理入口是
`python -m ddz.api_v2`，每次从公开日志重建当前牌局的出牌序列，一次模型前向完成
整手牌的主体与翼牌选择。

按 [api.md](api.md) 实现 QOJ 斗地主。默认模型 **1,986,763 参数**，仅以每局结束时服务器定义的原始积分 `deltas` 学习；所有非终局奖励严格为零。无奖励塑形、辅助 reward、奖励裁剪、奖励缩放、模仿或辅助预测损失。PPO 的 entropy 是标准策略正则项，不是环境奖励。

## 规则与动作

- 54 张牌、随机首叫、每家一次叫分、叫 3 立即结束叫分；三次流局后第四副最后一人必要时强制叫分。
- 完整 14 类牌型，包含歧义飞机的显式 `choice`、重复单翼、四带二可带对子、翼牌禁止炸弹和王炸、顺子主体禁止 2 和王。
- 炸弹/王炸及春天/反春各令倍数 **加一**。地主得分为农民相反数的两倍，不把胜负当成积分代理。
- 309 个主体动作（含 pass）+ 4 个叫分动作 + 15 个翼牌动作。先选主体，翼牌按点数不降序逐个选择，带对时点数严格递增。可完成性掩码保证没有死路，穷举覆盖所有合法解释，不依赖 `/hints`，没有候选截断。
- 翼牌内部决策不轮换玩家、不发布半手牌、不触发额外奖励。整个动作完成才写公开历史。
- 三座位收益向量保持绝对座位顺序计算 return，actor 取自己的分量。不能照搬双人棋类轮流翻转 advantage 符号的实现。

## 模型与信息边界

15 个带点数位置编码的手牌 token 和一个全局状态 token，3 层 cross-attention + self-attention + FFN，宽 192，6 heads，FFN 640，bf16 计算、float32 参数/输出。

类别 embedding 使用与查表等价的 one-hot GEMM，避免大批量重复历史和 padding 在反向传播中争用少量 embedding 行的原子加法。

每层可读取整局公开事件记忆：192 个位置，保存公开叫分、流局、底牌揭示、每手出牌与 pass 的座位、牌数和牌型解释。事件按发生顺序写入，并加入可学习的时序位置嵌入；`hist_len` 掩码支持从零到 192 个有效事件的变长历史。192 大于单局合法公开事件上界；不丢弃旧记录。PPO 保存原始记忆并用更新后的参数重编码，没有 stale recurrent hidden state，也不需要跨 minibatch 拼接隐状态。训练可用 `--train-memory-limit 96` 跳过被掩码遮住的填充位置；任何 minibatch 出现更长的历史都会自动使用完整 192 位置路径。

输入包括自己手牌、公开底牌、三个座位已出牌计数、未知牌计数、剩余手牌张数、叫分、地主相对位置、当前需压牌型、pass 数、炸弹数、各座位出牌手数、流局次数和当前内部带牌状态。服务器 log 可重建 table 和完整顺序；不会读取对手手牌、未揭示底牌或 fairness 中的洗牌序列。测试对未知牌分配做扰动，验证 actor 和 critic 输入保持相同。时间/版本/托管控制由 API executor 处理，不作为理想自博弈规则的输入；目标为每局期望原始积分，不以六局比赛的名次做额外奖励。

Policy 和 value 有各自的两层 MLP 分支。Value 同时预测三座位的原始期望积分，并精确约束三者和为零；三个 head target 都只来自相同真实积分 return。

## 并行 PPO

默认 2,048 个并行环境 × 128 步 = **262,144 决策/更新**，minibatch 2,048，1 epoch；环境、rollout、return、minibatch 更新均由 JAX `vmap` / `lax.scan` / `jit` 执行，Python 不逐局采样。参数预算小并使用大样本批次以降低不完全信息的方差。

`gamma=1, lambda=0.95`：不折扣终局积分，用 GAE 的 0.95 衰减控制优势估计跨度；rollout 截断处使用 value bootstrap。优势做全批量标准化，不对奖励/return 做标准化。AdamW、学习率 warmup、梯度范数裁剪、PPO clip 与 KL 停更保护均已实现。已有主运行在 step 6700 前使用 `lambda=1`，从该完整检查点续训时记录 `ppo.lambda: 1 → 0.95` 的配置迁移；日志逐步记录 `gae_lambda`。

每次更新还记录 `value_explained_variance = 1 - Var(target - value) / Var(target)`，在 rollout 采集时的三座位 value 预测和对应 GAE lambda-return target 上计算，与三座位 value loss 口径一致。`acting_value_explained_variance` 单独统计当时出牌座位的预测。target 方差为零时记 0；负数表示预测残差的方差大于 target 自身方差。

每 500 次更新的积分天梯用 `python -m ddz.score_ladder_graph --run-dir runs/my_run --output runs/my_run/score_ladder_graph_500_v3 --reuse-baseline-dir runs/my_run/score_ladder_500_v2 --watch` 持续更新。每个快照都与前面最近三个 500-step 版本及 step 0 锚点对战。每组使用共同的 144 副牌，每副牌交换地主和双农民角色，共 288 场；配对结果以候选模型控制座位的原始队伍净积分期望值（积分/局）衡量，包含倍数。主曲线对全部配对净分做等权最小二乘投影，step 0 定为 0。整副牌共同重采样给出曲线和相邻变化的 95% 区间。旧版只对战 step 0 的积分曲线与胜负 Elo 曲线独立保留。

每 10 更新，随机取一个 minibatch，分别反传纯 policy loss 与未加权 value loss，记录完整参数和共享主干的梯度 L2 norm。至少训练 20 更新且连续 5 更新满足有限数值、KL 阈值和实际更新比例后，向 `policy_grad_l2/value_grad_l2` 调整 `vf_coef`。几何平滑，每次最多变化 2 倍，限制 `[0.001,10]`。记录加权 value norm、下一系数和预测比值，避免把 loss 数值接近误当成梯度接近。系数和稳定性计数均随断点保存。

## 运行

使用本工作区已有环境：

```bash
cd /mnt/pfs/guoyuchong/guoyuchong/ddz
PY=../.venv-gnn-jax/bin/python
JAX_PLATFORMS=cpu "$PY" -m pytest -q
"$PY" -m ddz.train --smoke --out runs/my_smoke --require-a100
"$PY" -m ddz.benchmark --envs 256 --horizon 16 --minibatch 512
scripts/train_a100.sh --out runs/my_run
scripts/train_a100.sh --out runs/my_run --resume
```

正式训练入口强制验证只使用一张 A100。默认 GPU 内存比例 0.35，给暂停且保留进程状态的围棋训练留出显存。需要改变时在启动前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION`。

输出：`metrics.jsonl`、`eval.jsonl`、`status.json`、`latest.json`、`latest.msgpack`（参数、optimizer、环境、随机数、迭代、vf runtime）、周期性 `policy_*.msgpack`、`config.json`。完整状态原子替换；同一目录有排他写锁，必须显式 `--resume` 才能接着跑。SIGTERM/SIGINT 会在当前更新后保存并停止。默认每 100 次更新保存、每 100 次与固定出牌启发式对手评估；初期评估不是强棋力证明。已有运行允许单次只改变保存间隔或 PPO `lambda`，其他冻结配置仍须一致。

续训时可用 `--minibatch N` 调整 PPO minibatch；运行时记录有效值，并使学习率按原先每次更新的进度连续变化。实测 4096/8192 对当前 A100 吞吐提升很小，当前训练仍使用 2048。`--train-memory-limit` 也会随完整检查点保存并在下次续训自动恢复。

```bash
kill -TERM "$(cat runs/my_run/pid)"  # 保存当前更新后退出
```

训练超参数和代码在 `runs/a100_2m_rawscore_v1/launch.json` 中有启动记录和源码哈希。围棋暂停记录见 `runs/go_pause.json`；本次使用 SIGSTOP 保留全部内存状态，恢复命令是 `kill -CONT 642518`，恢复前应先停止斗地主训练以避免 GPU 资源竞争。

## API bot

先完成模型编译再入队；每次完整状态都从公开 log 重建记忆，因此重启、流局、409 重算都不会重复追加历史。物理牌 id 从自己实际手牌映射，包含合法 `choice`。支持叫分、出牌、pass、自动托管退出、排队心跳、版本冲突、超时后的状态重新读取和比赛换局。不调用聊天接口。

```bash
export DDZ_API_KEY='your-key'
../.venv-gnn-jax/bin/python -m ddz.api \
  --base https://YOUR-SITE/api/v1/doudizhu \
  --config runs/my_run/config.json \
  --checkpoint runs/my_run/policy_0000100.msgpack \
  --mode single --once
```

基础 CLI 从环境读取密钥；长期上线 worker 从仓库外的私有文件读取。
Fortune 已连接 QOJ 真实计分比赛，持续 tmux 运行。使用以下可迁移入口：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-cpu.txt
git lfs install --local
git lfs pull
# 将自己的 key 写入 ~/.config/ddz/qoj_api_key，权限设为 0600。
DDZ_USERNAME=YOUR_USERNAME scripts/qoj_match.sh prepare
scripts/qoj_match.sh start
scripts/qoj_match.sh attach
```

`prepare` 验证发布文件哈希，并冻结代码、worker 和权重到
`runs/qoj_match_v1/releases/`。同一账户使用排他锁，已存在 tmux 会话时 `start`
拒绝另开客户端。`scripts/qoj_match.sh status` 和该目录的 `dashboard.html` 可查看
当前状态、模型完整输入与 value。需要代理时，在运行目录的私有 `network.env`
中设置并 export 对应环境变量；该文件不进入 Git。模型先在 CPU 完成编译才入队；
看门狗恢复退出或卡住的 worker，401 与账户锁冲突停止并留下日志。

输入只包含自身手牌和公开信息，绝对座位循环重编号后全部输入张量一致。
同座位 GAE 的其他座位展示分数由自身 value 和确定的团队积分比例推导，
不是通过对手手牌预测；地主未确定时隐藏其他座位 value。

## 测试范围

独立 NumPy 牌型 oracle 对照全部主体和随机牌型；小手牌的完整子集枚举与层次动作集合一致；256 局批量随机完整牌局检查合法性、牌数守恒、终局积分和历史容量；叫分/流局/春天/反春/炸弹累计、非法动作不改变状态；训练/API 观测逐字段一致；隐藏信息不可见；409、托管、HTTP JSON 与 Bearer header；三人 return 与截断/跨局边界；记忆影响、padding 不影响；两个梯度分支非零、参数更新与完整 checkpoint 恢复后采样逐字一致。另有实际 A100 bf16 PPO smoke 与批量对战评估。

## V5 样本效率实验

隔离入口 `ddz.train_efficiency` 从一个完整、冻结的 V5 checkpoint 分叉，保留策略参数、已有 Adam moments、坐标年龄、环境和 RNG。新增辅助参数使用自己的 Adam 年龄。原 `ddz.train_v5` 训练配置不变。

实验开关：

| 配置 | 默认 | 用途 |
| --- | --- | --- |
| `ppo.epochs` | 1 | 多轮复用 rollout；当前 minibatch KL 超限时拒绝更新，并跳过后续反传 |
| `ppo.gae_clock` | `public` | `own` 按同一座位的实际观测递推；未获得同座位 bootstrap 的尾部样本排除 |
| `model.belief_head` / `ppo.belief_coef` | false / 0 | 对手剩余牌数分类辅助监督；隐藏手牌仅作标签，不进入网络输入 |
| `pool_probability` / `opponent_pool` | 0 / 空 | 每局随机引入冻结历史对手；其动作不参与 learner PPO 策略损失 |
| `ppo.optional_actor_only` | false | 排除仅有一个完整合法动作的 actor 样本，保留 value/辅助监督 |
| `model.team_value` | false | 地主已确定后约束两个农民的同观测 value 相等，保持零和 |
| `ppo.random_action_prob` | 0 | 使用已有完整动作混合探索及精确行为概率 |

学习率按新采集 rollout 数推进，避免增加 epochs 顺带改变学习率。各配置保留 `gamma=1, lambda=.95`、自适应 value 系数和每 100 updates 保存。所有历史仍有完整注意力和变长支持，出牌仍是完整动作的一次决策。

`own` GAE 中 gamma 按间隔的公开 transition 数计，lambda 按该玩家的下一次决策计；因此它同时改变 bootstrap 的信息来源和有效信用分配长度。对照结果应归因于这项完整改动，不能仅归因于其中一个因素。

```bash
../.venv-gnn-jax/bin/python -m ddz.train_efficiency \
  --config path/to/variant.json --source path/to/frozen/latest.msgpack \
  --out runs/isolated_variant --steps 1024 --require-a100
# 续跑同一配置、同一源 checkpoint 与同一冻结对手池
../.venv-gnn-jax/bin/python -m ddz.train_efficiency \
  --config path/to/variant.json --source path/to/frozen/latest.msgpack \
  --out runs/isolated_variant --steps 2048 --require-a100 --resume
```

`ddz.evaluate_efficiency` 用全局牌局编号生成共同独立牌，分别评估自然叫分及强制叫分后的地主、下家、门板六个互补对局。置信区间按整副牌重采样，不能把同副牌的六场当成六个独立样本。自然叫分指标按叫分起始座位统计；强制角色指标对地主分数除以 2 后对三个角色等权。

消融流程、源码哈希、训练曲线、强 DouZero 结果和确认结果保留在 `runs/v5_efficiency_campaign_v1/`。开关实现与工程测试通过不等于棋力收益成立；以独立牌局、匹配训练预算和两条训练 RNG 流的确认结果为准。未确认的方案保持为实验选项。

## V5 八卡续训

八卡入口为 `ddz.train_distributed_v5`，通过持久集群队列运行。8 个独立 rollout 副本共享一份策略和 Adam，梯度用 NCCL 同步；优势归一化、有效样本计数、KL 停更和自适应 value 梯度均按全局数据计算。包含空有效 mask 的副本；任一副本出现非有限数值时全部停止更新。

| 项目 | 单卡 | 八卡 |
| --- | ---: | ---: |
| 总并行环境 | 1024 | 8192 |
| 每个环境 rollout 长度 | 64 | 64 |
| 每轮新决策 | 65,536 | 524,288 |
| 常规全局 minibatch | 2048 | 16,384 |
| PPO epochs | 1 | 1 |
| 模型参数 | 5,097,200 | 5,097,200 |

历史超过常规长度时仍使用完整 192 个事件容量，并将单卡训练 minibatch 降至 1024，即全局 8192；没有截断当前牌局事件。迁移保留已有权重、Adam moments/坐标年龄和 value 系数；rank 0 保留原环境与 RNG，新增七个副本使用独立随机流。八卡完整检查点保存全部环境和随机数，可恢复相同 rollout 状态。

冻结输入、三项八副本 CPU 数值对照、八卡实跑/NCCL 证据保存在 `runs/v5_cluster8_v1/`。提交入口只用 `../Q/tools/cluster/submit_scheduler_task.sh '<command>'`，不写旧 STOP/RUN 桥接命令。每个生产队列项执行最多 1000 updates，成功保存后才将下一段追加到 FIFO 队尾；失败项保留，不自动重试。每 100 updates 保存，CPU 天梯每 500 updates 对战最近三个版本与起点，使用 1536 副共同牌、自然叫分和强制角色两种口径。

`scripts/finalize_efficiency_campaign.py --root runs/v5_efficiency_campaign_v1 --cluster-root runs/v5_cluster8_v1` 自动生成最终 HTML/PNG，并按冻结门槛选择验证过的同座位 GAE 或保存的原 V5 起点，再提交八卡生产续训；不会增加棋力对照组。当前训练位置可读 `runs/production_current.json` 和八卡生产目录的 `training/status.json`。
