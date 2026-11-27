# 使用说明

本文档只说明当前代码的运行方式。实验设计和网络结构分别见 [experiment.md](experiment.md) 和 [workflow.md](workflow.md)。

## 1. 运行环境

> 新克隆用户注意：`.venv/` 不会随 Git 仓库下载，请先按
> [environment.md](environment.md) 的“Python 与虚拟环境”和“Requirements 与后端”两节创建虚拟环境并安装依赖。
> `data/MNIST/` 和 `artifacts/` 已纳入版本库；如果数据缺失，仍可按 README 的“开始使用”重新准备。
> 后文命令默认环境已经就绪。

- 支持 Linux 和 Windows。
- Python 解释器：Linux 为 `.venv/bin/python`，Windows 为 `.venv\Scripts\python.exe`。
- 当前工作区依赖已安装在项目虚拟环境中：PyTorch、TorchVision、Matplotlib、psutil。
- GUI 使用标准库 Tkinter；Linux 需要系统包 `python3-tk`，Windows 由 python.org 安装包提供。
- 所有命令都从仓库根目录执行。Windows 下把下文命令中的 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`，路径分隔符按 Windows 写法调整。

MNIST 数据默认放在 `data/MNIST/raw/`。如果数据缺失，可以在训练命令中加 `--download` 自动下载。

## 2. 常用命令

### 2.1 训练一组模型

```sh
.venv/bin/python main.py train \
  --stage basic \
  --model cnn \
  --seed 42 \
  --device cpu
```

正式实验使用 10 个 epoch、batch size 128、Adam 学习率 0.001，这些也是默认值。快速冒烟测试可以限制每轮 batch 数：

```sh
.venv/bin/python main.py train \
  --stage augmented \
  --model mlp \
  --seed 42 \
  --epochs 1 \
  --max-batches 2 \
  --device cpu
```

### 2.2 评估一个运行目录

```sh
.venv/bin/python main.py evaluate \
  --run-dir artifacts/<run目录> \
  --device cpu
```

会生成测试曲线、卷积核、特征图和错分样本。MLP 没有卷积层，因此只生成曲线和错分样本。

### 2.3 单张图片预测

```sh
.venv/bin/python main.py predict \
  --checkpoint artifacts/<run目录>/weights/final.pt \
  --image examples/digits/digit_7.png \
  --true-label 7 \
  --output /tmp/prediction.png \
  --device cpu
```

`--invert` 可选 `auto`、`yes`、`no`，默认根据背景亮度自动判断是否反色。

### 2.4 运行完整对照实验

```sh
.venv/bin/python main.py batch --device auto --output artifacts
```

默认运行 2 个阶段 × 2 个模型 × 3 个种子，共 12 组；每组在独立子进程中训练，训练完成后自动评估并生成汇总。

常用限制参数：

```sh
# 只跑基础阶段的 CNN，用于调试
.venv/bin/python main.py batch \
  --stages basic \
  --models cnn \
  --seeds 42 \
  --epochs 1 \
  --max-batches 1 \
  --device cpu
```

CUDA 显存不足时，`batch` 会先以 batch size 64 从头重跑；`auto` 模式仍失败时整批转 CPU。

### 2.5 重新生成汇总表

```sh
.venv/bin/python main.py report \
  --run-root artifacts/<batch目录>/<attempt目录>
```

生成 `summary.csv`、`summary.md` 和 `summary.json`，包含各次运行明细、阶段/模型均值 ± 标准差，以及 `CNN − MLP`、`增强 − 基础` 差值。

### 2.6 启动图形界面

```sh
.venv/bin/python main.py gui --device cpu --artifacts artifacts
```

默认 `--device auto`；CUDA 不可用时自动回退 CPU。默认读取 `artifacts/` 下最新的 `summary.json` 和权重。

## 3. GUI 使用方法

界面有两个页签。

### 单图预测

1. 点击“选择图片”载入本地图片，或直接使用示例。
2. “示例集”可选：
   - **清晰示例**：数字 0–9 各一张。
   - **模糊示例**：10 张低置信度、易混淆样本，其中部分会被模型判错。
3. 点击“使用示例”在当前示例集内循环切换，并自动重新预测。
4. “模型权重”下拉框可切换任意一次运行的 `final.pt`。
5. “图片处理”支持自动判断、强制反色和不反色。

结果显示预测数字、置信度、Top-3 和 10 类概率柱状图。示例带有真实标签；模型判错时预测数字会标红，状态栏显示“误判”。

### 实验对比

- “实验批次”下拉框切换不同批量实验。
- 查看阶段/模型汇总、关键差值和逐次运行明细。
- 在右侧选择运行，并点击“曲线”“卷积核”“特征图”“错分样本”查看对应图片。
- “打开汇总”使用系统默认程序打开 `summary.md`。

## 4. 主要参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--stage` | `basic` 或 `augmented` | `basic` |
| `--model` | `cnn` 或 `mlp` | `cnn` |
| `--seed` | 随机种子 | `42` |
| `--epochs` | 训练轮数 | `10` |
| `--batch-size` | 批量大小 | `128` |
| `--lr` | Adam 学习率 | `0.001` |
| `--device` | `auto`、`cuda` 或 `cpu` | `auto` |
| `--output` | 训练产物根目录 | `artifacts` |
| `--max-batches` | 每轮最多 batch 数，仅用于冒烟测试 | 不限制 |
| `--download` | 数据缺失时自动下载 MNIST | 关闭 |

`gui` 额外使用 `--artifacts` 指定实验产物目录。

## 5. 输出文件

每个运行目录通常包含：

```text
<run>/
├── config.json
├── env.json
├── history.json
├── pip-freeze.txt
├── test_metrics.json    # 执行 evaluate 后生成
├── curves.png           # 执行 evaluate 后生成
├── kernels.png          # 仅 CNN，执行 evaluate 后生成
├── feature_maps.png     # 仅 CNN，执行 evaluate 后生成
├── misclassified.png    # 执行 evaluate 后生成
├── evaluation.json      # 执行 evaluate 后生成
└── weights/
    ├── epoch_01.pt
    ├── ...
    ├── epoch_10.pt
    └── final.pt
```

批量实验根目录还包含 `summary.csv`、`summary.md` 和 `summary.json`。当前 MNIST 数据和 `artifacts/` 已纳入版本库；后续新增运行目录需要手动确认后再提交。

## 6. 示例图片

- `examples/digits/`：清晰示例，`digit_0.png` 到 `digit_9.png`。
- `examples/ambiguous/`：模糊示例，文件名格式为 `digit_<真实标签>_amb_<序号>.png`。
- `examples/ambiguous/samples.json`：记录每张模糊示例的测试集索引、真实标签、模型预测和 Top-2 概率。

## 7. 常见问题

**没有图形界面或 Tkinter？**

安装 `python3-tk`，或在无显示环境使用 `train`、`evaluate`、`predict`、`batch`、`report` 命令行。

**CUDA 不可用？**

使用 `--device auto` 或 `--device cpu`。`auto` 会先做 CUDA 自检，失败时记录原因并使用 CPU。

**想查看当前正式实验？**

```sh
.venv/bin/python main.py gui --device cpu --artifacts artifacts
```

界面会自动读取最新的 `artifacts/*/summary.json`。
