# CNN 手写数字识别

本仓库用于课程实验和课堂演示。项目基于 Python 3.14、PyTorch 和 MNIST，比较双卷积 CNN 与参数量接近的全连接网络。实验分基础训练和轻量数据增强两个阶段，记录测试准确率、训练时间、内存和显存。课堂演示包含卷积核、特征图、错分样本和单张图片预测。

当前已完成网络、数据、训练、评估、单图预测、批量实验调度、多种子汇总报告和 Tkinter 图形界面。完整操作步骤见 [USAGE.md](USAGE.md)。

## 开始使用

按[环境配置与工程结构](environment.md)创建项目虚拟环境，并安装 [requirements.txt](requirements.txt) 中的依赖。实验优先使用通过自检的 CUDA 环境，CPU 路径可用于复现。

安装依赖后，用项目虚拟环境中的 Python 从仓库根目录执行下面的代码，准备 MNIST 的官方训练集与测试集。

```python
from torchvision.datasets import MNIST

MNIST(root="data", train=True, download=True)
MNIST(root="data", train=False, download=True)
```

TorchVision 将数据放在 `data/MNIST/raw/`。也可从 CVDF 镜像下载四个原始 IDX 压缩包并解压到同一目录。[TorchVision MNIST 文档](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html) · [CVDF 下载镜像](https://github.com/cvdfoundation/mnist)

## 运行实验

项目支持 Linux 和 Windows，以下命令以 Linux 为例，均从仓库根目录执行。Windows 下把 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`，环境准备见 [environment.md](environment.md#windows-环境准备)。

```sh
# 单组训练、评估和单图预测
.venv/bin/python main.py train --stage basic --model cnn --seed 42
.venv/bin/python main.py evaluate --run-dir artifacts/<run>
.venv/bin/python main.py predict \
  --checkpoint artifacts/<run>/weights/final.pt \
  --image digit.png

# 正式矩阵：2 阶段 × 2 模型 × 3 种子
# 每组在独立子进程中训练，训练结束后自动评估并写出 summary.*
.venv/bin/python main.py batch --device auto --output artifacts

# 对已有结果重新生成汇总表
.venv/bin/python main.py report --run-root artifacts/<batch>/<attempt>

# 图形界面：单图预测 + 实验对比
.venv/bin/python main.py gui
```

`--device auto` 会先做 CUDA 自检，失败时整批回退 CPU；也可显式使用 `--device cpu` 或 `--device cuda`。`batch` 遇到 CUDA OOM 时先以批量大小 64 从头重跑，仍失败则在 `auto` 模式下转 CPU。

## 图形界面

`python main.py gui` 使用标准库 Tkinter 启动轻量界面，不需要额外安装 GUI 框架；Linux 系统需要 `python3-tk`。默认读取 `artifacts/` 下最新的 `summary.json` 和运行权重，也可以用 `--artifacts` 指定其他产物目录。

- **单图预测**：选择图片、切换模型权重、选择自动/强制/不反色；显示预测数字、置信度、Top-3 和 10 类概率条。
- **实验对比**：展示 4 组阶段/模型汇总、CNN−MLP 与增强−基础差值、12 次运行明细，并可预览曲线、卷积核、特征图和错分样本。
- 示例集支持“清晰示例”和“模糊示例”：
  - 清晰示例为 `examples/digits/` 中的数字 0–9，每类一张。
  - 模糊示例为 `examples/ambiguous/` 中低置信度、易混淆的 MNIST 测试图，包含模型判对和判错的情况；误判时预测数字会标红。
- 启动时自动载入清晰示例；点击“使用示例”在当前示例集内循环切换，每次切换后自动重新预测。
- `--device` 支持 `auto/cuda/cpu`，推理与命令行 `predict` 使用同一套预处理和权重加载逻辑。

## 正式实验结果

正式实验在 CPU 上完成：10 个 epoch、batch size 128、Adam 学习率 0.001，使用种子 42、123、2026 做配对重复。该机器 CUDA 不可用，因此统一使用 `--device cpu`。下表为 3 个种子的均值 ± 样本标准差，完整明细见 `artifacts/20261127-140126_batch/summary.md`（已随仓库上传）。

| 阶段 | 模型 | 测试准确率 | 测试损失 | 训练耗时 (s) | CPU 时间 (s) | 峰值 RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| basic | CNN | 98.99% ± 0.15% | 0.0345 ± 0.0070 | 188.82 ± 40.73 | 872.88 ± 139.20 | 485.5 ± 4.8 |
| basic | MLP | 97.76% ± 0.21% | 0.0785 ± 0.0062 | 107.32 ± 1.34 | 338.31 ± 2.74 | 442.6 ± 0.2 |
| augmented | CNN | 99.10% ± 0.12% | 0.0263 ± 0.0068 | 269.24 ± 21.47 | 1211.08 ± 152.67 | 486.0 ± 4.0 |
| augmented | MLP | 98.08% ± 0.23% | 0.0596 ± 0.0032 | 163.29 ± 2.51 | 405.15 ± 3.54 | 442.4 ± 0.4 |

主要差值：

- `CNN − MLP` 测试准确率：basic 为 **+1.23 ± 0.06 个百分点**，augmented 为 **+1.02 ± 0.12 个百分点**。
- `增强 − 基础`：CNN 为 **+0.11 ± 0.03 个百分点**，MLP 为 **+0.33 ± 0.08 个百分点**；训练耗时分别增加约 80.4 秒和 56.0 秒。
- 单次运行最高准确率为 augmented-CNN-seed42 的 **99.22%**。

每个运行目录都保存了 10 轮权重、`curves.png` 和 `misclassified.png`；CNN 额外包含 `kernels.png` 与 `feature_maps.png`。命令行单图预测演示位于 `artifacts/20261127-140126_batch/prediction_demo/`，使用 augmented-CNN-seed42 权重将测试集数字 7 预测为 7，置信度约 99.995%。GUI 的清晰示例位于 `examples/digits/`（0–9 各一张），模糊示例位于 `examples/ambiguous/`（10 张低置信度/易混淆样本），均来自 MNIST 测试集。

## 数据集引用

本实验使用 MNIST 官方划分。训练集有 60,000 张，测试集有 10,000 张。每个样本为一张 `28 × 28` 的灰度图，标签为 `0–9`。

- LeCun, Yann, Corinna Cortes, and Christopher J. C. Burges. *MNIST handwritten digit database*. [数据集说明与引用](https://huggingface.co/datasets/ylecun/mnist#citation-information) · [原始发布页](https://yann.lecun.com/exdb/mnist/)
- LeCun, Yann, Léon Bottou, Yoshua Bengio, and Patrick Haffner. *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 86(11), 2278–2324, 1998. [论文 DOI](https://doi.org/10.1109/5.726791)

## 项目文档

| 文档 | 内容 |
| --- | --- |
| [USAGE.md](USAGE.md) | 环境准备、命令行和 GUI 使用方法 |
| [workflow.md](workflow.md) | CNN 和全连接网络的层配置、张量形状与参数量 |
| [environment.md](environment.md) | 虚拟环境、依赖、设备回退与目录结构 |
| [experiment.md](experiment.md) | 两阶段实验、数据增强、指标口径与产出 |
| [history.md](history.md) | 团队成员和 Agent 的进度与交接 |
| [课程指导文件](refs/翻转课堂1-卷积神经网络（CNN）原理与手写数字识别实战.md) | 课程要求与小组分工 |

## 仓库协作规范

- 根据工作范围开分支，例如 `feat/model` 或 `docs/experiment`。提交前检查 `git status` 和 `git diff --cached`，让每次提交集中在一项可复核的改动上。
- 按工程文档划分数据、模型、训练和评估模块，脚本使用项目相对路径。改动网络或实验条件时同步更新对应文档，并记录验证命令与结果。
- 团队成员和 Agent 在开工前阅读 `history.md`，记录任务与涉及文件，避免多人同时修改同一处。完成、受阻或交接时追加进度，随相关改动一并提交。
- 版本库跟踪源码、文档、依赖清单、MNIST 数据和当前实验产物。本地 Agent 指令文件、虚拟环境与缓存由 [.gitignore](.gitignore) 管理。

## 许可证

项目原创代码和文档采用 MIT 许可证，复用时保留版权和许可声明。MNIST 数据与 `refs/` 中的第三方课程资料沿用各自的权利声明。[LICENSE](LICENSE)
