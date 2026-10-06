# 仓库整理与最新 checkpoint 同步（2026-10-06）

## 分支

本地只保留 `main`，远端只有 `origin/main`。已删除旧发布前备份分支
`local-pre-publish-74054`（`40ebbf5875870e6806149f5f78f44547bf0e63d4`）。
该分支有独立旧提交，因此先生成并验证完整 Git bundle，再移除分支引用。
归档：`/mnt/pfs/guoyuchong/guoyuchong/ddz/runs/repo_cleanup_20261006/local-pre-publish-74054.bundle`；SHA-256 `d7ec7348020f13974a8c98035f680b2c3e30059f08738f97e9eb46ba3953ea33`。
原三个 LFS 对象仍在本地，不执行 LFS prune 或重写 main 历史。
恢复命令记录在 `runs/repo_cleanup_20261006/branch_archive.json`。

## 最新完整模型发布

- V6，8,014,192 参数，相对 50600 / 全局 **77154 step**。
- raw policy、EMA policy 和完整续训 checkpoint 已从同一原子完整断点导出。
  实际核对 raw/EMA 快照参数与完整断点一致，然后计算全部发布文件 SHA-256。
- EMA decay 0.999、rollout 更新计数 11600、optimizer step 5341622。
- `models/release.json` 标明配置、三个权重/状态文件、完整 SHA-256 和来源。
- `models/*.msgpack` 使用 Git LFS；发布三个新 LFS 对象及关联 main 提交。
- 之前的 V6 76562 发布包保留在
  `runs/model_exports/previous_models_before_v6_77154_cleanup_20261006`，原 Git/LFS
  历史亦保留。导出结果移入 models，不保留一份额外的新全量导出副本。

## 代码与评测兼容

一并提交已审阅的 QOJ tmux 布局、启动/附着 helper 和持续在线说明；两份
shell 脚本语法检查通过。未启动或重启 QOJ 服务。

核对运行依赖时发现，待启用的裁剪方案会切换到新的 V6 训练目录，现有本地
评测器仅支持 V5→V6 架构切换，会拒绝这次单字段采样变更。现增加注册检查：
必须只有 `ppo.adv_keep_fraction=0.5` 变更，原目录、完整状态保留 receipt 和
相对迁移步数一致；要求真实已验收 nranks=8/NCCL 和 8,014,192 参数。
旧的裁剪验收 schema 缺少 parameters 字段时，使用实际八卡 trainer status
核对参数量，仅在已注册裁剪方案中接受这条兼容路径。

13 项针对性测试通过（1.14 秒）：裁剪跟随、重启不重复初始点、原架构跟随
和缺失/错误证明、其他超参变更、保留状态失配拒绝。测试证明绑定源码 SHA。
控制器在旧评测已完成、GPU 空转时切换；原进程和自有 idle 退出后才启动新
冻结控制器 `/mnt/pfs/guoyuchong/guoyuchong/ddz/runs/local_a100_v6_trim_follow_500_65536_v1`，保留全部 100 个已完成评测点和结果引用。

当前唯一 GPU owner 记录在 `runs/local_a100_current.json`，空转利用率实测
100%。八卡训练、待运行裁剪任务的冻结依赖保持原样。

## 证据

- `runs/repo_cleanup_20261006/branch_archive.json`
- `runs/repo_cleanup_20261006/model_publication.json`
- `runs/repo_cleanup_20261006/controller_VERIFICATION.json`
- `/mnt/pfs/guoyuchong/guoyuchong/ddz/runs/local_a100_v6_trim_follow_500_65536_v1/migration_receipt.json`
- `models/release.json`
