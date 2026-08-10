# SMTgazer2

SMTgazer2 是一个面向 SMT（Satisfiability Modulo Theories）求解的算法组合与调度项目。项目使用 Sibyl 的图神经网络（GNN）从 SMT2 查询中提取 201 维实例特征，再通过聚类和 SMAC/LightGBM 为不同实例生成由 4 个候选求解器组成的时间调度方案。

仓库提供统一入口 `run_smtgazer.sh`，可自动完成 Python/CUDA 环境配置、GPU 可用性检查、Sibyl 特征生成或 GNN 训练、SMTgazer portfolio 优化、结果校验和 GPU 指标采集。

## 目录结构

```text
SMTgazer2/
├── run_smtgazer.sh              # 环境安装、检查和实验的统一入口
├── SMTgazer/                    # SMTgazer portfolio 训练、推理与评测代码
│   ├── run_experiment.py        # 实验编排入口
│   ├── SMTportfolio.py          # Portfolio 训练和推理
│   ├── calc_portfolio_result.py # PAR2 和未解数计算
│   ├── summarize_gpu_metrics.py # GPU 采样汇总
│   ├── data/                    # 各数据集的求解器运行时间标签
│   ├── machfea/                 # 特征生成代码、求解器列表和特征输出
│   └── output/                  # Portfolio 训练结果与测试调度结果
├── sibyl/                       # SMT2 建图、GNN 训练与 embedding 提取
├── logs/                        # 每次统一入口运行产生的日志和指标
└── .cache/sibyl_graphs/         # 可复用的 Sibyl 图缓存
```

## 支持的数据集

Portfolio 阶段支持以下数据集：

- `BMC`
- `Equality+LinearArith`
- `QF_Bitvec`
- `QF_LinearRealArith`
- `QF_NonLinearIntArith`
- `SyGuS`
- `SymEx`

其中本地 Sibyl GNN 自训练流程支持 `BMC`、`SyGuS` 和 `SymEx`。

原始 BMC、SymEx 和 SyGuS 数据可从 [Zenodo](https://doi.org/10.5281/zenodo.6521826) 获取；SMT-COMP 数据可从 [SMT-COMP 2021](https://smt-comp.github.io/2021/benchmarks.html) 获取。运行特征阶段前，SMT2 查询应位于 `sibyl/data/<DATASET>/`，或通过 `--query-root` 指定其他目录。

## 环境要求

统一脚本当前要求：

- Linux x86-64；
- 至少 8 GiB 可用磁盘空间；
- NVIDIA GPU、可用的 NVIDIA 驱动及 `nvidia-smi`；
- CUDA backend 编译时需要 `g++`；
- OpenCL GPU backend 需要 `clinfo` 和厂商 GPU ICD；
- 可访问 Python/Conda 和 pip 软件源。

脚本会优先使用已有 Conda；若未找到，则在项目内下载固定版本的 Miniconda。默认环境目录为 `.venv`，也可通过环境变量修改：

```bash
export SMTGAZER_ENV_PREFIX=/path/to/environment
```

主要固定版本包括 Python 3.10、CUDA 12.1、PyTorch 2.2.0、PyTorch Geometric 2.5.3、LightGBM 4.5.0、pySMT 0.9.6、NumPy 1.26.4 和 scikit-learn 1.6.1。脚本同时以 editable 模式安装仓库内的 SMAC 实现。

## 环境配置方法

以下命令均在项目根目录执行：

```bash
cd /path/to/SMTgazer2
```

### 1. 仅准备环境

```bash
bash ./run_smtgazer.sh prepare --lightgbm-backend cuda
```

`prepare` 可在没有 GPU 的节点上安装 Python 依赖、CUDA 12.1 用户态编译工具并构建 LightGBM CUDA；GPU 运行验证会推迟到 GPU 节点执行。

### 2. 配置环境并检查 GPU LightGBM

```bash
bash ./run_smtgazer.sh setup --lightgbm-backend cuda
```

`setup` 创建或复用 `.venv`、安装固定依赖、构建并验证 LightGBM GPU backend，但不会执行完整 smoke test。

### 3. 执行完整环境与 GPU 检查

```bash
bash ./run_smtgazer.sh check --lightgbm-backend cuda
```

不指定 action 时默认执行 `check`：

```bash
bash ./run_smtgazer.sh
```

该检查会实际验证 NVIDIA GPU、PyTorch CUDA、LightGBM fit/predict、仓库 SMAC、混合 surrogate 以及 Sibyl GPU embedding，任一 GPU 后端不可用时直接失败，不会静默回退到 CPU。

### LightGBM backend

```bash
--lightgbm-backend auto   # 默认；WSL2 优先 CUDA，其余环境自动判断
--lightgbm-backend cuda   # NVIDIA CUDA backend，推荐本项目实验配置
--lightgbm-backend gpu    # LightGBM OpenCL GPU backend
```

## 项目运行方法

查看统一入口的全部参数：

```bash
bash ./run_smtgazer.sh --help
```

### 使用发布的 Sibyl checkpoint 运行完整实验

若 checkpoint 和查询位于默认目录，可运行：

```bash
bash ./run_smtgazer.sh run \
  --dataset SyGuS \
  --stage full \
  --seeds 0 \
  --torch-device cuda:0 \
  --lightgbm-backend cuda
```

默认 checkpoint 为：

```text
sibyl/inference/<DATASET>/model_checkpoints/<DATASET>_model_0.pt
```

也可以显式指定输入：

```bash
bash ./run_smtgazer.sh run \
  --dataset SyGuS \
  --query-root /path/to/SyGuS/queries \
  --checkpoint /path/to/SyGuS_model_0.pt \
  --seeds 0
```

### 在本机 GPU 自训练 Sibyl GNN并运行完整实验

```bash
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

该流程依次执行：

1. 环境检查和 GPU smoke test；
2. 将 SMT2 查询转换为可复用图缓存；
3. 在 GPU 上 mini-batch 训练 Sibyl GNN；
4. 使用新 checkpoint 批量提取 201 维训练/测试特征；
5. 进行 X-means 聚类和 SMAC/LightGBM portfolio 优化；
6. 生成并校验测试调度结果；
7. 汇总各阶段 GPU 指标。

注意：`--overwrite` 会直接替换同名 canonical 特征和实验结果，且不会自动备份。仅在确认需要覆盖已有结果时使用。

### 分阶段运行

只生成特征：

```bash
bash ./run_smtgazer.sh run \
  --dataset SyGuS \
  --stage features \
  --train-sibyl \
  --seeds 0 \
  --overwrite
```

复用已有特征，只运行 portfolio：

```bash
bash ./run_smtgazer.sh run \
  --dataset SyGuS \
  --stage portfolio \
  --seeds 0
```

运行多个 portfolio seed：

```bash
bash ./run_smtgazer.sh run \
  --dataset SyGuS \
  --stage portfolio \
  --seeds 0,1,2
```

常用参数说明：

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--stage` | `full`、`features` 或 `portfolio` | `full` |
| `--seeds` | 逗号分隔的 portfolio 随机种子 | `0` |
| `--clusters` | 覆盖数据集的聚类数上限 | 数据集默认值 |
| `--train-sibyl` | 先在本机训练新的 Sibyl GNN | 关闭 |
| `--gnn-epochs` | GNN 训练轮数 | 25 |
| `--gnn-batch-size` | GNN 训练 batch size | 8 |
| `--gnn-workers` | 训练和特征提取 DataLoader workers | 4 |
| `--graph-workers` | SMT2 并行建图 worker 数 | 6 |
| `--feature-batch-size` | GPU embedding batch size | 8 |
| `--gnn-seed` | GNN 训练随机种子 | 0 |
| `--gpu-sample-interval` | GPU 采样间隔（秒） | 1 |
| `--torch-device` | PyTorch/Sibyl 使用的设备 | `cuda:0` |

## 输出文件

每次执行 `run_smtgazer.sh` 都会创建 `logs/<run-id>/`：

| 文件 | 内容 |
|---|---|
| `run.log` | 完整运行日志和实际子命令 |
| `pip-freeze.txt` | 本次运行的 Python 环境快照 |
| `nvidia-smi.txt` | GPU 和驱动快照 |
| `gpu.csv` | 按指定间隔采集的原始 GPU 数据 |
| `gpu-summary.json` | 整体及各阶段 GPU 汇总指标 |
| `checkpoints/<dataset>_gnn_seed<N>.pt` | 自训练 GNN 的最佳 checkpoint |
| `checkpoints/<dataset>_gnn_seed<N>.metrics.json` | 每轮 GNN 训练与验证指标 |

特征与 portfolio 输出位于：

```text
SMTgazer/machfea/infer_result/<DATASET>_train_feature.json
SMTgazer/machfea/infer_result/<DATASET>_test_feature.json
SMTgazer/output/train_result_<DATASET>_4_<CLUSTERS>_<SEED>.json
SMTgazer/output/test_result_<DATASET>_<SEED>_<ACTUAL_CLUSTERS>.json
```

训练结果保存归一化参数、聚类中心，以及每个聚类的 4 个求解器和时间片。测试结果则保存每个实例对应的求解器调度顺序和时间片。

## 指标说明

### Portfolio 指标

- **PAR2**：每个成功求解实例计入实际调度耗时；超时或未解实例计入 cutoff 的 2 倍惩罚，再对全部评价实例取平均。项目 cutoff 为 2400 秒，因此标准 PAR2 的未解惩罚应为 4800 秒。本次 seed0 无未解实例，所以是否触发惩罚不会影响其 PAR2。
- **UNK**：调度内所有求解器均未能在分配时间片内求解的实例数量，越低越好。
- **VBS（Virtual Best Solver）**：对每个实例事后选择最快求解器得到的理论下界。
- **SBS（Single Best Solver）**：在整个测试集上表现最好的单一求解器，用作实际基线。

仓库原始批量评测脚本可在 `SMTgazer/` 下运行：

```bash
cd SMTgazer
../.venv/bin/python calc_portfolio_result.py
```

该脚本默认统计 seed 0–9 和多个数据集，要求对应输出文件均已生成。需要注意，当前实现对未解实例累加 2400 秒，而不是标准 PAR2 的 4800 秒；当 `UNK > 0` 时应根据实验协议确认口径。本次 seed0 的 `UNK=0`，因此报告值不受此差异影响。

### GPU 指标

`gpu-summary.json` 对整体以及 `sibyl_graph_build`、`sibyl_gnn_train`、`sibyl_feature_extract`、`portfolio` 等阶段分别给出：

| 指标 | 含义 |
|---|---|
| `samples` | 有效 GPU 采样点数 |
| `duration_seconds` | 第一条和最后一条采样之间的时长 |
| `gpu_util_avg_pct` | GPU utilization 的算术平均值 |
| `gpu_util_p50_pct` / `p95` | GPU 利用率的第 50/95 百分位 |
| `gpu_util_max_pct` | GPU 峰值利用率 |
| `gpu_active_samples_pct` | 利用率不低于 10% 的采样占比 |
| `memory_used_avg_mib` / `peak` | 平均/峰值显存占用 |
| `power_draw_avg_w` / `peak` | 平均/峰值功耗 |
| `estimated_energy_wh` | 根据相邻采样功耗积分得到的估算能耗 |
| `temperature_peak_c` | 峰值 GPU 温度 |
| `sm_clock_avg_mhz` | 平均 SM 时钟频率 |

### GNN 训练指标

GNN metrics 文件记录每个 epoch 的训练/验证 ranking loss、Top-1、selected PAR2、求解率、吞吐量和显存峰值。当前最佳 checkpoint 按最低 validation ranking loss 保存，不是按最低 validation PAR2 保存。

## SyGuS Seed 0 实验结果

本次实验于 2026-08-08（UTC）完成，Run ID 为 `20260808_004232_39745`。实验使用 Tesla V100-PCIE-32GB、GNN seed 0 和 portfolio seed 0，执行的核心命令与上文“自训练 Sibyl GNN”示例一致。

### 最终结果摘要

| 指标 | 结果 |
|---|---:|
| 测试实例数 | 79,929 |
| **PAR2** | **0.031061** |
| **UNK** | **0** |
| 聚类数 | 3 |
| Portfolio size | 4 |
| 全流程时长 | 15,042.05 秒（约 4 小时 10 分 42 秒） |
| 全流程 GPU 平均利用率 | **25.04%** |
| GNN 训练 GPU 平均利用率 | **82.17%** |
| 特征提取 GPU 平均利用率 | **76.59%** |

### 分阶段 GPU 利用率

| 阶段 | 时长 | GPU 平均 | P50 | P95 | 峰值 | 活跃样本占比 |
|---|---:|---:|---:|---:|---:|---:|
| 图构建 | 27.10 秒 | 0.00% | 0% | 0% | 0% | 0.00% |
| GNN 训练 | 4,117.18 秒 | **82.17%** | 83% | 88% | 94% | **99.85%** |
| 特征提取 | 388.51 秒 | **76.59%** | 89% | 92% | **100%** | 85.83% |
| Portfolio | 10,505.11 秒 | 0.73% | 0% | 4% | 99% | 0.87% |

整体平均 GPU 利用率较低的主要原因是 portfolio 阶段占总时长约 69.84%，但该阶段主要执行 Python 多进程、聚类、SMAC 搜索和 CPU 控制逻辑。对于真正适合 GPU 的 GNN 训练和 embedding 提取阶段，平均利用率分别达到 82.17% 和 76.59%。

其他整体资源指标：

| 指标 | 结果 |
|---|---:|
| 采样数 | 14,414 |
| GPU 利用率 P50 / P95 / 峰值 | 0% / 87% / 100% |
| GPU 活跃样本占比 | 30.22% |
| 平均/峰值显存 | 4,630.84 / 32,268 MiB |
| 平均/峰值功耗 | 49.49 / 144.66 W |
| 估算总能耗 | 206.68 Wh |
| 峰值温度 | 53°C |

特征提取峰值显存达到 32,268 MiB，约占 V100 32,768 MiB 的 98.47%，因此不建议直接提高 `--feature-batch-size 8`。

### PAR2 对比

比较范围统一为本轮具有特征和调度结果的 79,929 个测试实例：

| 方法 | PAR2 | UNK |
|---|---:|---:|
| VBS（理论下界） | 0.019443 | 0 |
| **本轮 SMTgazer seed0** | **0.031061** | **0** |
| SBS：cvc5 | 0.036222 | 0 |

本轮相对最佳单求解器 cvc5 的 PAR2 降低约 **14.25%**，填补 SBS 与 VBS 差距的约 **30.76%**；本轮 PAR2 仍比 VBS 高约 59.76%。

仓库历史 SyGuS seed 1–9 的平均 PAR2 为 0.027250，中位数为 0.030153。本轮 seed0 比历史平均值高约 13.99%，但二者使用的 seed、GNN checkpoint 和测试覆盖不同（本轮 79,929，历史 79,981），不能视为严格 A/B 对照。合理结论是：本次流程和 GPU 优化成功，调度优于 SBS 且无未解实例，但自主训练 GNN 相对发布模型的算法优势仍需同条件实验验证。

### Seed 0 产物

```text
logs/20260808_004232_39745/run.log
logs/20260808_004232_39745/gpu.csv
logs/20260808_004232_39745/gpu-summary.json
logs/20260808_004232_39745/checkpoints/SyGuS_gnn_seed0.pt
logs/20260808_004232_39745/checkpoints/SyGuS_gnn_seed0.metrics.json
SMTgazer/output/train_result_SyGuS_4_3_0.json
SMTgazer/output/test_result_SyGuS_0_3.json
```

更完整的实验过程、训练指标、基线比较和局限性见根目录的 `SYGUS_SEED0_GPU_GNN_EXPERIMENT_REPORT.md`。

## 复现实验时的注意事项

- `--overwrite` 会覆盖已有特征和同 seed 输出，不创建备份。
- 使用 `--stage portfolio` 前必须已有对应数据集的训练和测试特征。
- 自训练 GNN 时，上游 GNN 标签可能有意跳过无法解析的查询，因此实际特征数可能小于 SMTgazer 标签中的可求解实例数。
- seed0 实验中 79,929 个测试实例全部求解，因此 PAR2 不受未解惩罚口径差异影响。
- GNN checkpoint 当前按最低 validation ranking loss 选择；本次保存的是 epoch 0，而最低 validation selected PAR2 出现在 epoch 6，后续对比实验应同时考察两种选择标准。
- 若目标是提升全流程 GPU 平均利用率，瓶颈是耗时接近 3 小时且主要使用 CPU 的 portfolio 阶段，而不是当前已达到较高利用率的 GNN 阶段。
