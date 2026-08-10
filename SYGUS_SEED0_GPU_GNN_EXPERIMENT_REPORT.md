# SMTgazer2：SyGuS Seed 0 自训练 GNN 与 GPU 利用率实验报告

> 实验状态：成功完成  
> 实验日期：2026-08-08（UTC）  
> Run ID：`20260808_004232_39745`  
> 数据集：SyGuS  
> GNN seed：0  
> SMTgazer portfolio seed：0

## 1. 摘要

本轮实验完成了从 SyGuS SMT2 查询建图、自主训练 Sibyl GNN、批量提取图嵌入，到 SMTgazer portfolio 训练与测试结果生成的完整流水线。实验不再把 Sibyl 的发布 checkpoint 当作固定外部接口，而是在本机 GPU 上重新训练 GNN，并把新训练模型产生的 201 维特征交给 SMTgazer。

本轮有三个主要结论：

1. **工程流程成功。** 完整流水线运行结束，日志出现 `[OK] Experiment completed`；GNN checkpoint、训练指标、GPU 指标和 SMTgazer seed 0 结果均已生成。
2. **GPU 优化目标基本达成。** GNN 训练阶段 GPU 平均利用率为 82.17%，GPU 活跃样本占比为 99.85%；批量特征提取阶段 GPU 平均利用率为 76.59%，峰值达到 100%。
3. **算法效果可用，但尚不能证明优于历史方案。** 新调度在 79,929 个测试实例上未解数为 0，PAR2 为 0.031061，比最佳单求解器 cvc5 改善约 14.25%；但相较现存历史 seed 1–9 结果的平均 PAR2 0.027250，本次结果高约 13.99%。由于测试覆盖、seed 和特征模型不同，这一历史比较只能作为参考，不能当作严格 A/B 结论。

总体判断是：**本轮在工程和 GPU 利用率方面成功，在最终算法质量方面取得了有效结果，但还没有证明自主训练 GNN 优于发布模型。**

## 2. 实验目标

原始流程主要使用 Sibyl 已发布的 GNN checkpoint，对 SMT 查询进行单条特征提取，再将特征交给 SMTgazer。新需求包括：

- 不再只使用发布 checkpoint，而是在本机 GPU 上训练 Sibyl GNN；
- 提高 GNN 训练和特征提取阶段的 GPU 利用率；
- 记录 GPU 利用率、显存、功耗、温度、时钟和估算能耗；
- 使用 SyGuS、seed 0 完成端到端验证；
- 输出可复现的训练指标和 SMTgazer 最终调度结果。

## 3. 完整实验流水线

本轮最终运行链如下：

```text
环境与 GPU 检查
    ↓
SyGuS SMT2 → PySMT 图构建/图缓存
    ↓
Tesla V100 上训练 Sibyl GAT
    ↓
保存最佳 checkpoint
    ↓
GPU mini-batch 提取 201 维图嵌入
    ↓
SMTgazer X-means 聚类
    ↓
SMAC/LightGBM portfolio 优化（seed 0）
    ↓
生成并校验 SyGuS 测试调度结果
    ↓
按阶段汇总 GPU 指标
```

执行命令为：

```bash
cd /root/autodl-tmp/SMTgazer2

bash ./run_smtgazer.sh run \
  --dataset SyGuS \
  --stage full \
  --train-sibyl \
  --gnn-epochs 25 \
  --gnn-batch-size 8 \
  --gnn-workers 4 \
  --graph-workers 6 \
  --feature-batch-size 8 \
  --gnn-seed 0 \
  --seeds 0 \
  --torch-device cuda:0 \
  --lightgbm-backend cuda \
  --gpu-sample-interval 1 \
  --overwrite
```

其中 `--overwrite` 替换了 canonical SyGuS 特征和 seed 0 输出，因此旧的 seed 0 结果已不可用于严格同 seed 对比。

## 4. 我们做了哪些修改

### 4.1 统一训练与实验入口

顶层脚本 [`run_smtgazer.sh`](run_smtgazer.sh) 增加了以下能力：

- `--train-sibyl`：在生成特征前训练新的 Sibyl GNN；
- GNN epoch、batch size、DataLoader worker、图构建 worker 和 GNN seed 参数；
- 以 1 秒间隔持续采集 GPU 指标；
- 使用阶段标记区分 `sibyl_graph_build`、`sibyl_gnn_train`、`sibyl_feature_extract` 和 `portfolio`；
- 实验退出时自动生成 `gpu-summary.json`。

### 4.2 图缓存

新增：

- [`sibyl/graph_cache.py`](sibyl/graph_cache.py)：定义图缓存路径、校验和 PyG Dataset；
- [`sibyl/prepare_graph_cache.py`](sibyl/prepare_graph_cache.py)：并行将 SMT2 查询转换为可复用的 NPZ 图缓存。

图缓存避免每个 epoch 重复解析 SMT2 文件，并允许 DataLoader worker 并行加载数据，减少 GPU 等待 CPU 的时间。

### 4.3 GPU mini-batch GNN 训练

新增 [`sibyl/train_gnn.py`](sibyl/train_gnn.py)，主要特性包括：

- 真正的图 mini-batch 训练，batch size 为 8；
- 4 个 DataLoader worker、pin memory、persistent worker 和预取；
- CUDA AMP 混合精度训练；
- 固定随机 seed；
- 训练/验证吞吐、损失、Top-1、求解率、PAR2、显存峰值等指标；
- 保存结构化 checkpoint，包括模型配置与训练元数据；
- 按验证 ranking loss 保存最佳 checkpoint。

同时修改 [`sibyl/src/networks/gnn.py`](sibyl/src/networks/gnn.py)，保留 batch 维度，解除原实现只能使用 `batch_size=1` 的限制。

### 4.4 批量 GPU 特征提取

修改：

- [`sibyl/extract_embedding.py`](sibyl/extract_embedding.py)：兼容新旧 checkpoint，并支持从图缓存进行批量 GPU embedding；
- [`SMTgazer/machfea/mach_run_inference.py`](SMTgazer/machfea/mach_run_inference.py)：以 batch=8、workers=4 生成训练和测试特征。

### 4.5 实验编排与 GPU 汇总

修改 [`SMTgazer/run_experiment.py`](SMTgazer/run_experiment.py)，把建图、GNN 训练、新 checkpoint、特征提取和 portfolio 串成完整流程。

新增 [`SMTgazer/summarize_gpu_metrics.py`](SMTgazer/summarize_gpu_metrics.py)，计算：

- GPU 利用率平均值、P50、P95、最大值；
- GPU 活跃样本比例；
- 显存平均值和峰值；
- 功耗平均值和峰值；
- 估算能耗；
- 峰值温度与平均 SM 时钟。

## 5. 首次失败与修复

首次运行 `20260807_192209_10885` 只完成了 CPU 建图：

- 目标图数量：99,909；
- 成功：99,899；
- 失败：10；
- GPU 利用率：0%，因为训练尚未开始；
- 程序在图构建结束后退出。

失败原因是 PySMT 默认复用进程级全局 `Environment`。并行 worker 会连续解析多个独立 SyGuS 文件，而常见符号 `inv-f` 在不同文件中具有不同函数签名，导致后一个文件被误判为符号类型重定义。

修复位置为 [`sibyl/src/data_handlers/graph-builder.py`](sibyl/src/data_handlers/graph-builder.py)：每个 SMT2 查询现在创建独立的 PySMT `Environment`，并让 `SmtLibParser` 与 `ASTBuilder` 使用同一个独立环境。

修复后：

- 10 个失败样例在同一 Python 进程中连续构建两轮，共 20 次，全部成功；
- 第二次完整运行复用已有 99,899 个缓存，只补建缺失的 10 个；
- 图缓存阶段最终 `failed=0`。

## 6. 实验环境

| 项目 | 配置 |
|---|---|
| GPU | Tesla V100-PCIE-32GB |
| GPU 显存 | 32,768 MiB |
| NVIDIA Driver | 580.105.08 |
| 驱动支持 CUDA | 13.0 |
| Python | 3.10.20 |
| PyTorch | 2.2.0+cu121 |
| PyTorch Geometric | 2.5.3 |
| CUDA 训练运行时 | 12.1 |
| LightGBM | 4.5.0，CUDA backend |
| pySMT | 0.9.6 |
| NumPy | 1.26.4 |
| GNN device | `cuda:0` |

说明：驱动显示支持 CUDA 13.0，而 PyTorch/编译工具链使用 CUDA 12.1；NVIDIA 驱动向后兼容，本轮 smoke test、LightGBM CUDA 和 PyTorch CUDA 均通过。

## 7. 数据与训练配置

### 7.1 数据覆盖

| 数据项 | 训练集 | 测试集 |
|---|---:|---:|
| SMTgazer 可求解实例 | 19,996 | 79,981 |
| GNN 图/特征实例 | 19,980 | 79,929 |
| 缺少图缓存的可求解实例 | 16 | 52 |
| 全求解器超时而跳过 | 4 | 19 |

GNN 标签文件包含 19,980 个训练实例和 79,929 个测试实例，总图缓存数为 99,909。SMTgazer 标签比 GNN 标签多 68 个可求解实例，因此本轮启用了 partial feature 校验，最终评价覆盖 79,929 个测试实例。

图缓存结果：

- 总数：99,909；
- 本轮新建：10；
- 复用：99,899；
- 失败：0；
- 缓存大小：约 569 MiB；
- 本轮扫描/补建耗时：23.53 秒。

### 7.2 GNN 配置

| 参数 | 数值 |
|---|---:|
| 训练 seed | 0 |
| Epoch | 25 |
| Batch size | 8 |
| DataLoader workers | 4 |
| AMP | 开启 |
| 训练样本 | 17,982 |
| 验证样本 | 1,998 |
| Message-passing passes | 2 |
| 输入维度 | 67 |
| 输出维度 | 5 个候选求解器 |
| Attention layers/heads 配置 | 5 |
| Pooling | attention |
| Jumping Knowledge | cat，开启 |
| 初始学习率 | 0.001 |
| Dropout | 0 |

模型 checkpoint SHA-256：

```text
d0d2ca484cbbd1ba3bb4223e77b5dfc1ace069bae85bbf05641a62959e561a6f
```

## 8. 运行完成情况

| 阶段 | 状态 | 主要输出 |
|---|---|---|
| 环境和 GPU smoke test | 成功 | PyTorch、LightGBM CUDA、SMAC、Sibyl embedding 均通过 |
| 图缓存 | 成功 | 99,909/99,909，failed=0 |
| GNN 训练 | 成功 | `SyGuS_gnn_seed0.pt` |
| 特征提取 | 成功 | train 19,980；test 79,929 |
| SMTgazer portfolio 训练 | 成功 | `train_result_SyGuS_4_3_0.json` |
| SMTgazer 测试结果 | 成功 | 79,929 条调度结果 |
| GPU 指标汇总 | 成功 | `gpu-summary.json` |

完整日志最后包含：

```text
[OK] test result .../test_result_SyGuS_0_3.json
[OK] experiment stage completed
[OK] Experiment completed
[OK] GPU metric summary: .../gpu-summary.json
```

## 9. GPU 利用率与资源指标

### 9.1 整体指标

| 指标 | 结果 |
|---|---:|
| 采样数 | 14,414 |
| 监控时长 | 15,042.05 秒（约 4 小时 10 分 42 秒） |
| GPU 平均利用率 | 25.04% |
| GPU 利用率 P50 | 0% |
| GPU 利用率 P95 | 87% |
| GPU 峰值利用率 | 100% |
| GPU 活跃样本占比（利用率 ≥10%） | 30.22% |
| 平均显存占用 | 4,630.84 MiB |
| 峰值显存占用 | 32,268 MiB |
| 平均功耗 | 49.49 W |
| 峰值功耗 | 144.66 W |
| 估算总能耗 | 206.68 Wh |
| 峰值温度 | 53°C |
| 平均 SM 时钟 | 1,221.95 MHz |

### 9.2 分阶段指标

| 阶段 | 时长 | 时长占比 | GPU 平均 | P50 | P95 | 峰值 | 活跃占比 | 平均/峰值显存 MiB | 平均/峰值功耗 W | 能耗 Wh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 图构建 | 27.10 秒 | 0.18% | 0.00% | 0% | 0% | 0% | 0.00% | 0 / 0 | 40.68 / 40.81 | 0.31 |
| GNN 训练 | 4,117.18 秒（68 分 37 秒） | 27.37% | **82.17%** | 83% | 88% | 94% | **99.85%** | 11,478 / 13,190 | 76.84 / 122.94 | 87.88 |
| 特征提取 | 388.51 秒（6 分 29 秒） | 2.58% | **76.59%** | 89% | 92% | **100%** | 85.83% | 24,601 / **32,268** | 115.12 / 144.66 | 12.44 |
| Portfolio | 10,505.11 秒（2 小时 55 分） | **69.84%** | 0.73% | 0% | 4% | 99% | 0.87% | 1,210 / 2,044 | 36.32 / 47.26 | 106.00 |

### 9.3 GPU 指标解释

**GNN 训练阶段达到了预期目标。** 82.17% 的平均利用率与 99.85% 的活跃率说明 GPU 几乎持续参与计算，mini-batch、AMP 和异步数据加载有效解决了原始 `batch_size=1` 带来的低利用率问题。

**特征提取阶段同样获得较高利用率。** 平均 76.59%、P50 89%、P95 92%，说明多数采样点处于高负载。峰值显存 32,268 MiB，占 V100 32,768 MiB 的约 98.47%，当前 feature batch=8 已接近显存上限，不建议直接继续增大。

**整体平均利用率被 portfolio 阶段拉低。** Portfolio 占总时长约 69.84%，但 GPU 平均利用率只有 0.73%。这一阶段以 Python 多进程、聚类、SMAC 搜索和 CPU 控制逻辑为主。GNN 和特征阶段的 GPU 优化已经成功，但如果目标是提高“全流程平均 GPU 利用率”，还需要缩短或重新设计 portfolio 阶段，而不是继续放大 GNN batch。

## 10. GNN 训练效果

训练共运行 25 个 epoch，总训练报告时长为 4,112.65 秒，约 68 分 33 秒。

| 指标 | 结果 |
|---|---:|
| 平均训练吞吐 | 113.92 samples/s |
| 平均验证吞吐 | 300.26 samples/s |
| PyTorch 峰值 allocated 显存 | 2,929.26 MiB |
| PyTorch 峰值 reserved 显存 | 12,800 MiB |
| 最佳 validation ranking loss | 0.027518 |
| 保存 checkpoint 的 epoch | **0** |

关键 epoch 对比：

| Epoch | 学习率 | Train loss | Train Top-1 | Train selected PAR2 | Val loss | Val Top-1 | Val selected PAR2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0（最终保存） | 1.0e-3 | 0.033647 | 48.78% | 6.9646 | **0.027518** | **92.49%** | 3.6233 |
| 6（验证 PAR2 最低） | 5.0e-4 | 0.028337 | 54.02% | 0.4254 | 0.028673 | 68.02% | **1.2254** |
| 24（最后一轮） | 1.5625e-5 | **0.027510** | 55.03% | **0.0250** | 0.033206 | 34.53% | 1.2304 |

当前保存策略以 validation ranking loss 为准，因此最终 checkpoint 来自 epoch 0。后续 epoch 虽然训练 loss 和训练集 selected PAR2 明显下降，但 validation ranking loss 没有超过 epoch 0。

这暴露出两个问题：

1. **checkpoint 选择目标与最终任务不完全一致。** Epoch 6 的 validation selected PAR2 明显优于 epoch 0，但没有被保存。
2. **训练和验证指标存在波动。** Validation Top-1 在不同 epoch 间变化很大，说明模型对类别分布、稀有超时样本或 ranking loss 比较敏感。

因此，25 个 epoch 确实完成了训练，但按当前 checkpoint 策略，真正用于特征提取的是第 0 个 epoch 的模型。下一轮应同时保存“最低 ranking loss”和“最低 validation PAR2”两个 checkpoint，再用最终 SMTgazer PAR2 决定哪个更合适。

## 11. SMTgazer 最终效果

### 11.1 本轮 seed 0 结果

| 指标 | 结果 |
|---|---:|
| 测试实例数 | 79,929 |
| 平均 PAR2 | **0.031061** |
| 未解实例数（UNK） | **0** |
| 聚类数 | 3 |
| Portfolio size | 4 |

### 11.2 与求解器基线比较

比较范围统一为本轮拥有特征和调度结果的 79,929 个测试实例。

| 方法 | PAR2 | UNK | 相对本轮 |
|---|---:|---:|---:|
| 虚拟最佳求解器（VBS） | 0.019443 | 0 | 理论下界 |
| **本轮 SMTgazer** | **0.031061** | **0** | — |
| 最佳单求解器 cvc5（SBS） | 0.036222 | 0 | 本轮优约 14.25% |
| UltimateEliminator + MathSAT | 1.370926 | 45 | 明显更差 |
| z3 | 23.664911 | 787 | 明显更差 |
| smtinterpol | 24.063717 | 788 | 明显更差 |
| veriT | 2398.498682 | 79,879 | 明显更差 |

由此得到：

- 本轮相比最佳单求解器 cvc5，PAR2 下降约 **14.25%**；
- 本轮填补了 SBS 到 VBS 之间约 **30.76%** 的差距；
- 本轮 PAR2 仍比 VBS 高约 **59.76%**；
- 本轮在评价范围内没有未解实例。

这说明新特征产生的 portfolio 是有效的，能够优于任意单一求解器；但距离理想的按实例选择仍有明显空间。

### 11.3 与仓库历史结果的参考比较

仓库保留的历史 SyGuS seed 1–9 结果如下：

| Seed | 历史 PAR2 | UNK | 测试数 |
|---:|---:|---:|---:|
| 1 | 0.030220 | 0 | 79,981 |
| 2 | 0.030694 | 0 | 79,981 |
| 3 | 0.022396 | 0 | 79,981 |
| 4 | 0.020814 | 0 | 79,981 |
| 5 | 0.024874 | 0 | 79,981 |
| 6 | 0.030235 | 0 | 79,981 |
| 7 | 0.025166 | 0 | 79,981 |
| 8 | 0.030153 | 0 | 79,981 |
| 9 | 0.030700 | 0 | 79,981 |
| **历史均值** | **0.027250** | **0** | — |
| **历史中位数** | **0.030153** | **0** | — |
| **本轮 seed 0** | **0.031061** | **0** | **79,929** |

本轮 PAR2 比历史 seed 1–9 平均值高约 13.99%，也略高于历史中位数。然而这不是严格对照实验，原因包括：

- 本轮 seed 为 0，历史表中是 seed 1–9；
- 本轮使用自主训练 checkpoint，历史结果使用旧特征/旧模型；
- 本轮评价 79,929 个实例，历史结果评价 79,981 个实例；
- 旧 seed 0 输出已被 `--overwrite` 覆盖。

因此只能得出“本轮尚未表现出超过历史结果的证据”，不能直接断言自主训练模型一定更差。

## 12. 结果文件与完整性

### 12.1 主要产物

| 产物 | 路径 | 大小 |
|---|---|---:|
| 完整运行日志 | [`logs/20260808_004232_39745/run.log`](logs/20260808_004232_39745/run.log) | 约 80 KiB |
| GPU 原始采样 | [`logs/20260808_004232_39745/gpu.csv`](logs/20260808_004232_39745/gpu.csv) | 约 1.5 MiB |
| GPU 汇总 | [`logs/20260808_004232_39745/gpu-summary.json`](logs/20260808_004232_39745/gpu-summary.json) | 约 3.6 KiB |
| GNN checkpoint | [`logs/20260808_004232_39745/checkpoints/SyGuS_gnn_seed0.pt`](logs/20260808_004232_39745/checkpoints/SyGuS_gnn_seed0.pt) | 约 492 KiB |
| GNN 指标 | [`logs/20260808_004232_39745/checkpoints/SyGuS_gnn_seed0.metrics.json`](logs/20260808_004232_39745/checkpoints/SyGuS_gnn_seed0.metrics.json) | 约 21 KiB |
| 训练特征 | [`SMTgazer/machfea/infer_result/SyGuS_train_feature.json`](SMTgazer/machfea/infer_result/SyGuS_train_feature.json) | 约 71 MiB |
| 测试特征 | [`SMTgazer/machfea/infer_result/SyGuS_test_feature.json`](SMTgazer/machfea/infer_result/SyGuS_test_feature.json) | 约 281 MiB |
| Portfolio 训练结果 | [`SMTgazer/output/train_result_SyGuS_4_3_0.json`](SMTgazer/output/train_result_SyGuS_4_3_0.json) | 约 17 KiB |
| 测试调度结果 | [`SMTgazer/output/test_result_SyGuS_0_3.json`](SMTgazer/output/test_result_SyGuS_0_3.json) | 约 21 MiB |
| 图缓存 manifest | [`.cache/sibyl_graphs/SyGuS/manifest.json`](.cache/sibyl_graphs/SyGuS/manifest.json) | — |

### 12.2 输出哈希

```text
d0d2ca484cbbd1ba3bb4223e77b5dfc1ace069bae85bbf05641a62959e561a6f  SyGuS_gnn_seed0.pt
d35fa15fb71a5196528915f410415c1cae5c875961d468a34311ebb426670a1e  train_result_SyGuS_4_3_0.json
35b4c9800bd6cf216b97c900cbc1225e875a42ff170e8b7a34e538d712d742fd  test_result_SyGuS_0_3.json
```

## 13. 局限性

本轮仍有以下局限：

1. **缺少严格的发布模型 A/B 基线。** 需要在同一测试子集、同一 seed、同一 portfolio 配置下比较发布 checkpoint 与自主训练 checkpoint。
2. **旧 seed 0 被覆盖。** 当前无法直接恢复完全同 seed 的旧结果进行对比。
3. **评价覆盖不完全一致。** 68 个可求解实例缺少 GNN 图/特征，本轮评价数量少于 SMTgazer 标签中的可求解实例数。
4. **checkpoint 选择标准可能不理想。** 最低 ranking loss 与最低 validation PAR2 不在同一 epoch。
5. **只运行了一个新 GNN seed。** 单次结果不能量化训练随机性的方差。
6. **Portfolio 是全流程瓶颈。** 其耗时接近 3 小时，占总时长约 70%，但基本不使用 GPU。
7. **特征提取显存已接近上限。** 峰值占用约 98.47%，继续增大 batch size 有 OOM 风险。

## 14. 下一步建议

建议按以下优先级继续：

### P0：严格 A/B 实验

固定以下条件：

- 相同 SyGuS 测试子集 79,929；
- 相同 portfolio seed 0；
- 相同 cluster 数与 portfolio size；
- 相同 GNN batch 与特征处理；
- 输出写入独立目录，禁止覆盖。

至少比较：

| 实验组 | Checkpoint | 目的 |
|---|---|---|
| A | 发布的 `SyGuS_model_0.pt` | 严格基线 |
| B | 本轮最低 ranking loss checkpoint | 当前实现 |
| C | 最低 validation PAR2 checkpoint | 检验 checkpoint 选择目标 |

### P1：改进 checkpoint 策略

- 每个 epoch 都保存轻量指标和候选 checkpoint；
- 同时维护 best-ranking-loss、best-validation-PAR2、best-Top1；
- 使用独立验证集跑一次下游 SMTgazer 小规模 portfolio，选择最终模型；
- 增加 early stopping，避免无收益的后续训练。

### P2：多 seed 稳定性

至少运行 GNN seed 0、1、2，并为每个模型运行相同 portfolio seed，报告均值、标准差和置信区间。

### P3：缩短 portfolio 阶段

- 分析 `portfolio_smac3.py` 的重复配置计算；
- 缓存相同候选配置的结果；
- 减少 Python 进程启动开销；
- 检查 SMAC/LightGBM 调用是否能批量化；
- 将 GPU 利用率目标限定为适合 GPU 的 GNN/特征阶段，同时单独报告全流程利用率。

## 15. 最终结论

本轮实验完成了关键工程闭环：**SMTgazer2 已能够自主在 GPU 上训练 Sibyl GNN，并将新模型无缝用于特征生成与 seed 0 portfolio 验证。**

GPU 方面，GNN 训练平均利用率 82.17%、活跃率 99.85%，批量特征提取平均利用率 76.59%，说明 GPU 优化方案有效。整体平均利用率只有 25.04%，主要由耗时最长且以 CPU 为主的 portfolio 阶段导致。

算法方面，本轮在 79,929 个测试实例上实现 0 个未解，PAR2 为 0.031061，相比最佳单求解器提升 14.25%，证明新流程产生的调度具有实际效果。但本次未超过历史结果的参考平均水平，而且 checkpoint 在 epoch 0 即达到最低验证损失，表明模型选择标准和训练目标仍需优化。

因此，本轮最准确的评价是：

> **工程成功，GPU 利用率目标达成，最终调度有效；但自主训练 GNN 的算法优势尚未被严格证明，需要下一轮同条件 A/B 实验。**

