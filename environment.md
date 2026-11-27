# 环境配置与工程结构

## Python 与虚拟环境

本项目支持 Linux 和 Windows。使用 CPython `3.14.x` 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。以下命令以 Linux 为例；Windows 环境准备见本节末尾。有同版本的 `.venv/` 时可以直接使用该环境安装依赖。

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv --torch-backend=auto -r requirements.txt
uv pip check --python .venv
```

Linux 下 IDE 使用项目中的 Python 解释器，路径为 `.venv/bin/python`。图形界面使用标准库 Tkinter；Linux 系统需要安装 `python3-tk`（多数桌面发行版已预装）。

### Windows 环境准备

1. 从 [python.org](https://www.python.org/downloads/) 安装 Python 3.14，安装时保留 Tcl/Tk 组件，这样 Tkinter GUI 可直接使用。
2. 在 PowerShell 中进入项目根目录，创建虚拟环境：

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
uv pip install --python .venv\Scripts\python.exe --torch-backend=auto -r requirements.txt
uv pip check --python .venv\Scripts\python.exe
```

如果 PowerShell 禁止执行激活脚本，可以只对当前终端临时放行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

之后所有 Python 命令使用：

```text
.venv\Scripts\python.exe main.py ...
```

无 NVIDIA GPU 时把 `--torch-backend=auto` 改为 `--torch-backend=cpu`，运行时也可加 `--device cpu`。

## Requirements 与后端

[`requirements.txt`](requirements.txt) 是此安装流程的直接依赖清单。

| 依赖 | 用途 |
| --- | --- |
| `torch` | 网络、训练、评估和设备管理 |
| `torchvision` | MNIST 读取和数据变换 |
| `matplotlib` | 学习曲线、卷积核、特征图与错分样本 |
| `psutil` | 采样训练进程的内存 |

新环境安装时解析当前平台兼容的稳定版。uv 的 `--torch-backend=auto` 根据当前环境选择 PyTorch 构建。只做 CPU 实验时，可将安装命令的后端改为 `--torch-backend=cpu`。需要手动指定 GPU 构建时，按平台、GPU 与驱动选择后端，并让 `torch` 和 `torchvision` 走同一构建源。[uv 的 PyTorch 安装说明](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection) · [PyTorch 安装指南](https://pytorch.org/get-started/locally/)

正式跑实验前，用 `uv pip freeze --python .venv` 保存已安装的版本清单到本次运行目录，并记录 Python 版本和实际 PyTorch 后端。同机复跑时使用该清单和对应后端。项目已有的 `pyproject.toml` 和 `uv.lock` 属于初始化模板，安装运行依赖时使用上述 `uv pip` 命令。

## 设备选择与回退

以下规则供训练入口实现。设备选择在整批对照实验开始时完成。

| 选项 | 行为 |
| --- | --- |
| `--device auto` | 优先检查 `torch.cuda.is_available()`，并在独立进程中对小批量做卷积前向、反向与同步。CUDA 自检通过后使用 GPU，否则记录原因，四组统一使用 CPU。 |
| `--device cuda` | 要求 CUDA 自检通过，失败时退出并报告原因。 |
| `--device cpu` | 直接使用 CPU。 |

CUDA 路径将模型、样本和标签放入同一设备，统一启用 `pin_memory=True` 和 `non_blocking=True`。CPU 路径使用 CPU 的数据加载和传输设置。遇到 CUDA 显存不足时，统一以批量大小 `64` 从头重跑。仍无法在 CUDA 上完成时，`auto` 模式将整批实验转到 CPU 重跑。数据或张量维度错误直接报告。

## 工程文件

根目录已有 `main.py` 和 `src/`。训练代码实现时，按下列结构组织模块，输出目录由运行入口创建。

```text
.
├── main.py                 命令行入口
├── src/                    源码目录
│   ├── __init__.py
│   ├── data.py             数据集、预处理和 DataLoader
│   ├── models.py           CNN 和全连接网络
│   ├── train.py            设备选择、训练、资源计量和权重保存
│   ├── evaluate.py         测试、单图预测和结果可视化
│   ├── report.py           多种子汇总、差值表和 CSV/Markdown 输出
│   ├── gui.py              Tkinter 单图预测与实验对比界面
│   └── utils.py            运行目录、JSON 和环境信息
├── data/MNIST/raw/          MNIST 原始数据
├── examples/digits/         GUI 用的 0-9 清晰示例
├── examples/ambiguous/      GUI 用的低置信度模糊示例
├── artifacts/              按运行保存权重、指标、版本和图表
├── refs/                   课程原始资料
├── .venv/                  本地 Python 环境
├── .gitignore              版本库忽略规则
├── pyproject.toml          项目元数据模板
├── uv.lock                 uv 项目初始化锁文件
├── requirements.txt        直接运行依赖
├── USAGE.md                常用命令和 GUI 使用说明
├── environment.md          环境和工程说明
├── workflow.md             网络构建
└── experiment.md           实验设置与指标
```

以项目根目录作为运行位置，数据通过相对路径 `data/MNIST/raw` 读取。版本库收录源码、文档、依赖清单、MNIST 数据和当前实验产物；虚拟环境、IDE 配置和缓存仍按 [`.gitignore`](.gitignore) 管理。课程演示所需权重和图表从 `artifacts/` 中选取并打包。
