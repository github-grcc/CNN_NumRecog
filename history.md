# 协作记录

本文件跟踪团队成员和 Agent 的开发进度，随项目版本管理。开始任务前查看已有记录，追加任务、负责人和涉及文件。完成、遇到阻塞或交接时，另起一条记录并写明验证结果与后续工作。并行协作时，由负责合并的人汇总涉及同一文件的进度。

每条记录包括日期、成员或 Agent 名称、状态和任务。正文写明涉及文件、已验证的内容，以及需要接手的事项。

## 2026-11-20 · Codex · 完成 · 整理仓库文档

- 新增 `README.md`、`LICENSE` 和本文件，整理 MNIST 引用与协作约定。
- 更新 `.gitignore`，排除本地 Agent 指令、缓存、虚拟环境、数据和实验产物。网络、环境、实验方案与课程指导文件纳入版本管理。
- 核对文档中的相对链接，用 `git check-ignore` 验证原始数据、预览图、缓存和 Agent 指令的忽略规则。
- 训练实现待接手。按 `environment.md` 的模块布局和 `workflow.md` 的输入输出约定实现，再按 `experiment.md` 验证。

## 2026-11-26 · OpenCode · 完成 · 实现基础模型与数据模块

- 新增 `src/__init__.py`、`src/models.py`、`src/data.py`。
- `models.py` 按 `workflow.md` 实现 CNN 与 MLP，统一输入 `[B, 1, 28, 28]`、输出 `[B, 10]` logits，并提供参数量统计和最小自检。
- `data.py` 实现两阶段预处理、MNIST 读取、固定种子的 DataLoader，以及 `set_epoch_seed` 增强同步机制；`num_workers=0` 的同步前提写在模块注释中。
- 验证命令：
  - `.venv/bin/python -m src.models`：CNN 105,866 参数、MLP 105,745 参数，输出均为 `(2, 10)`。
  - `.venv/bin/python -m src.data`：基础/增强/测试 batch 形状正确，相同种子下两个增强 loader 的图像和标签完全一致。
  - 单批次 Adam 训练：两个模型均完成前向、反向和参数更新。
- 待接手：`src/train.py`、`src/evaluate.py` 和 `main.py` 入口。

## 2026-11-26 · OpenCode · 完成 · 实现训练入口

- 新增 `src/utils.py`：运行目录、JSON 读写、环境信息、包版本清单和耗时格式化。
- 新增 `src/train.py`：
  - `auto/cuda/cpu` 设备选择，CUDA 在独立进程中做卷积前向/反向自检，失败时 `auto` 回退 CPU。
  - 固定种子、关闭 TF32 和 cuDNN benchmark。
  - 每轮训练计时、吞吐量、进程 CPU 时间、RSS 峰值和 CUDA 显存峰值。
  - 每轮保存 `weights/epoch_XX.pt`，末轮另存 `final.pt`；配置、环境、`pip-freeze.txt` 和历史写入运行目录。
  - 测试评估在训练计时区间之外执行。
- 重写 `main.py`，提供 `train` 子命令：
  - `python main.py train --stage basic --model cnn --seed 42 --device auto`
  - 支持 `--epochs`、`--batch-size`、`--lr`、`--output`、`--max-batches`、`--download`。
- 验证：
  - `.venv/bin/python -m src.train` 随机张量冒烟测试通过。
  - `basic-CNN`、`augmented-MLP`、`auto` 回退 CPU 三组小规模训练均完成，产物包含配置、历史、环境和权重。
  - 用 `torch.load(..., map_location="cpu", weights_only=True)` 成功读取 `final.pt`。
- 待接手：`src/evaluate.py`（测试曲线、卷积核/特征图、错分样本、单图预测）和批量实验入口。

## 2026-11-26 · OpenCode · 完成 · 实现评估、可视化与单图预测

- 新增 `src/evaluate.py`：
  - 从每个 `epoch_*.pt` 恢复模型并生成 `test_metrics.json`。
  - `curves.png`：训练/测试损失与准确率四联图。
  - `kernels.png`：第一层 16 个 3x3 卷积核。
  - `feature_maps.png`：同一张测试图经过两个卷积块的特征图。
  - `misclassified.png`：末轮错分样本、真实/预测标签和置信度。
  - 单图预处理、预测和概率柱状图。
- `main.py` 新增子命令：
  - `python main.py evaluate --run-dir <run>`
  - `python main.py predict --checkpoint <final.pt> --image <png>`
- 更新 `README.md`、`environment.md` 和 `src/__init__.py` 的完成状态。
- 验证：
  - 在 `basic-CNN` 冒烟运行上生成全部 4 张图和 `evaluation.json`、`test_metrics.json`。
  - 在 `augmented-MLP` 上正确跳过卷积核/特征图，仍生成曲线和错分样本。
  - 用 MNIST 测试集第 1 张图完成单图预测，并生成 `prediction.png`。
- 待接手：批量实验入口（2 阶段 × 2 模型 × 3 种子）和多种子汇总报告。

## 2026-11-27 · OpenCode · 完成 · 批量实验调度与多种子汇总

- 新增 `src/report.py`：
  - 递归读取运行目录中的 `history.json`，汇总测试准确率、损失、训练耗时、CPU 时间、RSS 峰值和 GPU 峰值。
  - 按阶段/模型输出均值 ± 样本标准差，计算 `CNN − MLP` 与 `增强 − 基础` 的准确率差值（百分点）和耗时差。
  - 写出 `summary.csv`、`summary.md` 和 `summary.json`。
- `main.py` 新增 `batch` 子命令：
  - 按 2 阶段 × 2 模型 × 3 种子启动 12 个独立训练子进程，设备在整批开始时统一选择。
  - CUDA OOM 时整批以 batch size 64 从头重跑；`auto` 模式仍失败则整批转 CPU。
  - 训练完成后自动对每个运行目录生成测试曲线、卷积核/特征图和错分样本。
  - 支持 `--skip-evaluate`、`--stages/--models/--seeds` 和 `--max-batches` 便于冒烟测试。
- `main.py` 新增 `report` 子命令，便于对已有运行目录重新生成汇总。
- 更新 `README.md`、`environment.md` 和 `src/__init__.py`，补充 Linux-only 说明、批量命令和模块清单。
- 验证：
  - `.venv/bin/python -m compileall -q main.py src` 通过。
  - 4 组 × 1 epoch × 1 batch 的 batch 冒烟测试完成训练、评估和汇总。
  - 2 组 × 1 epoch × 1 batch 的 batch 冒烟测试完成自动评估，CNN 生成 4 张图，MLP 生成 2 张图。
- 正式 12 组实验已完成，结果见下一条记录。

## 2026-11-27 · OpenCode · 完成 · 正式 12 组对照实验与单图演示

- 运行命令：
  - `.venv/bin/python main.py batch --device cpu --output artifacts`
  - 批量目录：`artifacts/20261127-140126_batch/`
  - 最终尝试目录：`artifacts/20261127-140126_batch/cpu_b128_1/`
- 实验条件：CPU、10 epochs、batch size 128、Adam lr 0.001、种子 42/123/2026，共 12 组。
- 产物完整性：
  - 12 个 `history.json`、120 个 `epoch_*.pt`、12 个 `final.pt`。
  - 36 张评估图：6 个 CNN 各 4 张（curves/kernels/feature_maps/misclassified），6 个 MLP 各 2 张（curves/misclassified）。
  - `summary.csv`、`summary.md`、`summary.json` 已生成。
- 测试准确率（3 种子均值 ± 样本标准差）：
  - basic CNN：98.99% ± 0.15%；basic MLP：97.76% ± 0.21%。
  - augmented CNN：99.10% ± 0.12%；augmented MLP：98.08% ± 0.23%。
  - `CNN − MLP`：basic +1.23 ± 0.06 pp；augmented +1.02 ± 0.12 pp。
  - `增强 − 基础`：CNN +0.11 ± 0.03 pp；MLP +0.33 ± 0.08 pp。
- 单图预测演示：
  - 使用 `augmented-CNN-seed42` 的 `final.pt`。
  - 输入 MNIST 测试集第 1 张图（真实标签 7），预测为 7，置信度 0.999954。
  - 产物：`artifacts/20261127-140126_batch/prediction_demo/{input_digit.png,prediction.png,sample.json}`。
- 已回填 `README.md` 的正式结果表，并在 `history.md` 记录产物路径与验证结果。

## 2026-11-27 · OpenCode · 完成 · Tkinter 单图预测与实验对比界面

- 新增 `src/gui.py`：
  - 单图预测页签：选择图片、切换权重、自动/强制/不反色，显示预测数字、置信度、Top-3 和 10 类概率柱状图。
  - 实验对比页签：自动读取最新 `summary.json`，显示阶段/模型汇总、关键差值、12 次运行明细，并预览 curves/kernels/feature_maps/misclassified。
  - 启动时自动使用 `prediction_demo/input_digit.png` 和最佳 CNN 权重预测；没有 `summary.json` 时从 `history.json` 临时汇总。
  - 仅使用标准库 Tkinter 和已有 PIL/torch 依赖，图表预览直接读取训练生成的 PNG。
- `main.py` 新增 `gui` 子命令，支持 `--artifacts` 和 `--device`。
- 更新 `README.md`、`environment.md` 和 `src/__init__.py`，补充 GUI 使用方式、Tkinter 依赖和模块清单。
- 验证：
  - GUI 实际创窗口、自动示例预测：数字 7，置信度 99.995%。
  - 切换到 MLP 权重预测成功；MLP 无卷积核时图表自动回退曲线。
  - 实验对比页显示 12 个运行、4 组汇总、差值和图表预览。
  - 无 artifacts 目录时界面正常显示“未找到实验结果”，不抛异常。
  - `main.py gui --device cpu` 可正常启动。

## 2026-11-27 · OpenCode · 完成 · GUI 多示例循环

- 新增 `examples/digits/digit_0.png` 到 `digit_9.png` 共 10 张示例图：
  - 来自 MNIST 官方测试集，每类一张。
  - 使用正式实验 `augmented-CNN-seed42` 权重挑选正确分类且置信度最高的样本。
  - 附 `examples/digits/README.md` 说明来源。
- `src/gui.py` 的示例逻辑改为列表：
  - 优先读取当前批次的 `prediction_demo/samples/`，否则读取 `examples/digits/`，再回退旧的 `input_digit.png`。
  - “使用示例”按钮循环切换 0–9，每次切换后自动预测。
  - 按钮显示当前序号，如“使用示例 4/10”；状态栏显示序号和真实标签。
- 验证：
  - 启动自动载入 digit_0，预测 0，状态“示例 1/10”。
  - 连续点击依次预测 1、2、3，按钮显示“使用示例 4/10”。
  - 第 10 张之后会回到第 1 张。

## 2026-11-27 · OpenCode · 完成 · GUI 表格行距调整

- 将实验对比页 `Treeview` 的 `rowheight` 从 24 px 调整为 30 px，增加表格行间距。
- 窗口高度从 960 调整为 1000，保证汇总表和运行明细完整显示。
- 已实际启动 GUI 检查，4 行汇总表和运行明细均正常显示。

## 2026-11-27 · OpenCode · 完成 · GitHub 公开仓库与完整产物上传

- 仓库 `github-grcc/CNN_NumRecog` 已创建并改为 Public。
- 更新 `.gitignore`：不再忽略 `data/MNIST/` 和 `artifacts/`，将 MNIST 原始数据和正式实验产物纳入版本库。
- 提交课程 `refs/` 的既有修改、`data/` 和 `artifacts/`。
- 说明：数据目录约 64 MB，artifacts 约 58 MB，最大单文件 47 MB，低于 GitHub 100 MB 限制。

## 2026-11-27 · OpenCode · 完成 · GUI 模糊示例集

- 新增 `examples/ambiguous/`：
  - 10 张来自 MNIST 测试集的低置信度/易混淆样本。
  - 使用正式实验 `augmented-CNN-seed42` 权重挑选，最高类别概率约 0.37–0.54。
  - 同时包含模型判对和判错的样本（6 张判对、4 张判错）；文件名格式 `digit_<真实标签>_amb_<序号>.png`。
  - `samples.json` 记录测试集索引、真实标签、模型预测、Top-1/Top-2 概率。
  - `README.md` 说明挑选标准和用途。
- GUI 单图预测页新增“示例集”下拉框：
  - `清晰示例`：0–9 各一张。
  - `模糊示例`：10 张低置信度/易混淆样本。
  - 切换示例集后立即载入第一张并自动预测。
  - “使用示例”按钮在当前示例集内循环；状态栏显示示例集、序号、真实标签和模型预测。
- 验证：
  - 切换到模糊示例后，第一张 digit_0 预测置信度 50.48%，第二名 digit_6 为 46.41%。
  - 第二张 digit_1 预测置信度 37.48%，第二名 digit_7 为 36.33%，概率条能直观看到竞争关系。
  - 模糊示例集中包含模型判错样本，可继续循环演示。
  - 第 6 张 digit_6 被模型预测为 5，状态栏显示“真实标签 6 · 误判”，预测数字标红。
  - GUI 窗口布局和清晰/模糊示例切换均正常。
- `USAGE.md` 开头补充新克隆提示：`.venv/` 和 `data/MNIST/` 不随仓库下载，环境步骤指向 `environment.md`，数据准备指向 README。
- 新增 `USAGE.md`，集中说明环境准备、训练/评估/预测/批量/汇总命令、GUI 用法、参数、产物和常见问题；README 文档表已加入入口。

## 2026-11-27 · OpenCode · 完成 · Windows 兼容支持

- `src/gui.py` 的“打开汇总”改为跨平台实现：
  - Windows 使用 `os.startfile`；
  - macOS 使用 `open`；
  - Linux 继续使用 `xdg-open`。
- `environment.md` 新增 Windows 环境准备：
  - python.org 安装 Python 3.14 + Tcl/Tk；
  - `py -3.14 -m venv .venv`；
  - `.venv\Scripts\python.exe` 安装依赖和运行命令；
  - CPU/CUDA 后端选择说明。
- `README.md` 和 `USAGE.md` 补充 Windows 解释器路径，不再声明仅支持 Linux。
- 验证：
  - `python -m compileall -q main.py src` 通过。
  - 使用模拟 `sys.platform == "win32"` 和 `os.startfile` 验证 GUI 打开汇总的 Windows 分支。
  - 训练、评估、批量实验和 GUI 的其余代码只依赖 `pathlib`、`sys.executable` 和现有跨平台依赖，无需额外改动。
