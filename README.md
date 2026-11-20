# CNN 手写数字识别

本仓库用于课程实验和课堂演示。项目基于 Python 3.14、PyTorch 和 MNIST，比较双卷积 CNN 与参数量接近的全连接网络。实验分基础训练和轻量数据增强两个阶段，记录测试准确率、训练时间、内存和显存。课堂演示计划展示卷积核、特征图、错分样本和单张图片预测。

当前已完成网络设计、实验方案与依赖清单。训练、评估和演示入口待实现。

## 开始使用

按[环境配置与工程结构](environment.md)创建项目虚拟环境，并安装 [requirements.txt](requirements.txt) 中的依赖。实验优先使用通过自检的 CUDA 环境，CPU 路径可用于复现。

安装依赖后，用项目虚拟环境中的 Python 从仓库根目录执行下面的代码，准备 MNIST 的官方训练集与测试集。

```python
from torchvision.datasets import MNIST

MNIST(root="data", train=True, download=True)
MNIST(root="data", train=False, download=True)
```

TorchVision 将数据放在 `data/MNIST/raw/`。也可从 CVDF 镜像下载四个原始 IDX 压缩包并解压到同一目录。[TorchVision MNIST 文档](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html) · [CVDF 下载镜像](https://github.com/cvdfoundation/mnist)

## 数据集引用

本实验使用 MNIST 官方划分。训练集有 60,000 张，测试集有 10,000 张。每个样本为一张 `28 × 28` 的灰度图，标签为 `0–9`。

- LeCun, Yann, Corinna Cortes, and Christopher J. C. Burges. *MNIST handwritten digit database*. [数据集说明与引用](https://huggingface.co/datasets/ylecun/mnist#citation-information) · [原始发布页](https://yann.lecun.com/exdb/mnist/)
- LeCun, Yann, Léon Bottou, Yoshua Bengio, and Patrick Haffner. *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 86(11), 2278–2324, 1998. [论文 DOI](https://doi.org/10.1109/5.726791)

## 项目文档

| 文档 | 内容 |
| --- | --- |
| [workflow.md](workflow.md) | CNN 和全连接网络的层配置、张量形状与参数量 |
| [environment.md](environment.md) | 虚拟环境、依赖、设备回退与目录结构 |
| [experiment.md](experiment.md) | 两阶段实验、数据增强、指标口径与产出 |
| [history.md](history.md) | 团队成员和 Agent 的进度与交接 |
| [课程指导文件](refs/翻转课堂1-卷积神经网络（CNN）原理与手写数字识别实战.md) | 课程要求与小组分工 |

## 仓库协作规范

- 根据工作范围开分支，例如 `feat/model` 或 `docs/experiment`。提交前检查 `git status` 和 `git diff --cached`，让每次提交集中在一项可复核的改动上。
- 按工程文档划分数据、模型、训练和评估模块，脚本使用项目相对路径。改动网络或实验条件时同步更新对应文档，并记录验证命令与结果。
- 团队成员和 Agent 在开工前阅读 `history.md`，记录任务与涉及文件，避免多人同时修改同一处。完成、受阻或交接时追加进度，随相关改动一并提交。
- 版本库跟踪源码、文档和依赖清单。本地 Agent 指令文件、虚拟环境、缓存、数据与训练产物由 [.gitignore](.gitignore) 管理。

## 许可证

项目原创代码和文档采用 MIT 许可证，复用时保留版权和许可声明。MNIST 数据与 `refs/` 中的第三方课程资料沿用各自的权利声明。[LICENSE](LICENSE)
