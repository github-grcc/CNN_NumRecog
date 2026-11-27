"""测试评估、可视化与单图预测。

本模块对应实验文档中的实验产出：

    1. 用各轮权重在测试集上生成准确率/损失曲线；
    2. 展示第一层卷积核和两个卷积块的特征图；
    3. 展示末轮模型的错分样本（真实标签 + 预测标签）；
    4. 加载 CNN 权重对单张图片做预测。

所有绘图函数使用 Matplotlib 的 ``Agg`` 后端，适合在无图形界面的 Linux
服务器上运行；图片保存到运行目录，不弹窗。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

# 必须在导入 pyplot 之前设置后端，否则无显示环境下可能初始化失败。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps, ImageStat
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.data import MNIST_MEAN, MNIST_STD, build_loader
from src.models import build_model
from src.train import evaluate_loss_acc
from src.utils import load_json, save_json


# ---------------------------------------------------------------------------
# 模型与检查点加载
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    model_name: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """从权重文件恢复模型。

    Args:
        checkpoint_path: ``weights/epoch_XX.pt`` 或 ``weights/final.pt``。
        device: 恢复后模型所在设备。
        model_name: 可选的模型名；默认从检查点保存的 ``config["model"]`` 读取。

    Returns:
        ``(model, checkpoint)``。模型已调用 ``eval()``，可直接推理。

    Raises:
        FileNotFoundError: 权重文件不存在。
        KeyError: 文件不是本项目保存的检查点格式。
        ValueError: 无法确定模型结构。
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到权重文件：{path}")

    # 先加载到 CPU，再按需移动到目标设备，避免 GPU 不可用时加载失败。
    # weights_only=True 只允许安全的基础类型和张量，不执行任意 pickle 代码。
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{path} 中缺少 model_state_dict")

    config = checkpoint.get("config", {})
    resolved_name = model_name or config.get("model")
    if resolved_name not in ("cnn", "mlp"):
        raise ValueError(
            f"无法从检查点确定模型结构，config['model']={config.get('model')!r}"
        )

    model = build_model(resolved_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def evaluate_checkpoints(
    run_dir: str | Path,
    test_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    """用每一轮的权重在测试集上评估，生成测试曲线所需数据。

    Args:
        run_dir: 单次运行目录，内部应包含 ``weights/epoch_*.pt``。
        test_loader: 固定顺序的测试集 DataLoader。
        device: 评估设备。
        max_batches: 冒烟测试时限制测试 batch 数；正式评估为 ``None``。

    Returns:
        每个 epoch 一条记录，字段包括 ``epoch``、``test_loss``、
        ``test_accuracy``、``samples`` 等，同时写入 ``test_metrics.json``。
    """
    run_path = Path(run_dir)
    weights_dir = run_path / "weights"
    checkpoint_paths = sorted(weights_dir.glob("epoch_*.pt"))

    if not checkpoint_paths:
        raise FileNotFoundError(f"{weights_dir} 中没有 epoch_*.pt")

    criterion = nn.CrossEntropyLoss()
    records: list[dict[str, Any]] = []

    for checkpoint_path in checkpoint_paths:
        model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
        metrics = evaluate_loss_acc(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            max_batches=max_batches,
        )

        record = {
            "epoch": int(checkpoint.get("epoch", 0)),
            "checkpoint": checkpoint_path.name,
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
            "samples": metrics["samples"],
            "batches": metrics["batches"],
        }
        records.append(record)
        print(
            f"[test epoch {record['epoch']:02d}] "
            f"loss={metrics['loss']:.4f} "
            f"acc={metrics['accuracy']:.4f}",
            flush=True,
        )

        # 及时释放模型，避免逐个加载检查点时内存持续增长。
        del model

    save_json(run_path / "test_metrics.json", records)
    return records


# ---------------------------------------------------------------------------
# 训练/测试曲线
# ---------------------------------------------------------------------------
def plot_learning_curves(
    history_data: dict[str, Any],
    test_metrics: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """绘制训练和测试的损失/准确率曲线。

    训练数据来自 ``history.json``，测试数据来自各轮权重评估结果。
    图内文字使用英文，避免 Linux 环境缺少中文字体导致方框乱码。
    """
    history = history_data.get("history", [])
    config = history_data.get("config", {})

    train_epochs = [record["epoch"] for record in history]
    train_losses = [record["train_loss"] for record in history]
    train_accuracies = [record["train_accuracy"] for record in history]

    test_epochs = [record["epoch"] for record in test_metrics]
    test_losses = [record["test_loss"] for record in test_metrics]
    test_accuracies = [record["test_accuracy"] for record in test_metrics]

    figure, axes = plt.subplots(2, 2, figsize=(11, 7))

    # 左上：训练损失。
    axes[0, 0].plot(train_epochs, train_losses, marker="o", color="tab:blue")
    axes[0, 0].set_title("Training loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3)

    # 右上：训练准确率。
    axes[0, 1].plot(train_epochs, train_accuracies, marker="o", color="tab:green")
    axes[0, 1].set_title("Training accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(alpha=0.3)

    # 左下：测试损失。若没有测试记录，显示提示文字。
    if test_metrics:
        axes[1, 0].plot(test_epochs, test_losses, marker="o", color="tab:red")
        axes[1, 0].set_title("Test loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, "No test metrics", ha="center", va="center")
        axes[1, 0].axis("off")

    # 右下：测试准确率。
    if test_metrics:
        axes[1, 1].plot(test_epochs, test_accuracies, marker="o", color="tab:orange")
        axes[1, 1].set_title("Test accuracy")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].set_ylim(0.0, 1.0)
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "No test metrics", ha="center", va="center")
        axes[1, 1].axis("off")

    figure.suptitle(
        f"{config.get('stage', '?')} / {config.get('model', '?')} / "
        f"seed {config.get('seed', '?')}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# 卷积核与特征图
# ---------------------------------------------------------------------------
def plot_conv_kernels(
    model: nn.Module,
    output_path: str | Path,
    max_kernels: int = 16,
) -> None:
    """可视化第一层卷积的 3x3 卷积核。

    CNN 第一层权重形状为 ``[16, 1, 3, 3]``，每个输出通道对应一个
    可以直接观察的 3x3 卷积核。使用以 0 为中心的统一灰度范围，
    便于比较不同核之间的正负响应。

    Args:
        model: CNN 模型。
        output_path: 输出 PNG 路径。
        max_kernels: 最多展示多少个卷积核，默认 16。
    """
    if max_kernels <= 0:
        raise ValueError(f"max_kernels 必须为正数，当前为 {max_kernels}")

    if not hasattr(model, "features"):
        raise TypeError("该模型没有 features 属性，无法展示卷积核")

    first_layer = model.features[0]
    if not isinstance(first_layer, nn.Conv2d):
        raise TypeError("features[0] 不是 Conv2d，无法展示卷积核")

    weights = first_layer.weight.detach().cpu()
    kernel_count = min(max_kernels, weights.shape[0])

    columns = 4
    rows = math.ceil(kernel_count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 1.7, rows * 1.7),
        squeeze=False,
    )

    # 用全局最大绝对值作为对称灰度范围，正负权重都能看清。
    value_limit = float(weights.abs().max().item()) or 1.0

    for index, axis in enumerate(axes.flatten()):
        axis.axis("off")
        if index >= kernel_count:
            continue

        # weights[index, 0] 是第 index 个输出通道的 3x3 核。
        kernel = weights[index, 0].numpy()
        axis.imshow(
            kernel,
            cmap="gray",
            vmin=-value_limit,
            vmax=value_limit,
        )
        axis.set_title(f"out {index}", fontsize=9)

    figure.suptitle("First conv layer kernels (3x3)", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_feature_maps(
    model: nn.Module,
    image: torch.Tensor,
    output_path: str | Path,
    max_channels: int = 16,
) -> None:
    """展示同一张输入图经过两个卷积层后的特征图。

    通过在 ``Conv2d`` 上注册前向 hook 捕获输出，再对每个通道做 ReLU
    后可视化。第一层特征图空间尺寸为 28x28，第二层为 14x14。

    Args:
        model: CNN 模型。
        image: 单张图片，形状为 ``[1, 1, 28, 28]`` 或 ``[1, 28, 28]``。
        output_path: 输出 PNG 路径。
        max_channels: 每层最多展示多少个通道，默认 16。
    """
    if max_channels <= 0:
        raise ValueError(f"max_channels 必须为正数，当前为 {max_channels}")

    if not hasattr(model, "features"):
        raise TypeError("该模型没有 features 属性，无法展示特征图")

    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.shape[1] != 1:
        raise ValueError(f"image 形状应为 [1, 1, 28, 28]，实际为 {tuple(image.shape)}")

    conv_layers = [
        layer for layer in model.features if isinstance(layer, nn.Conv2d)
    ]
    if not conv_layers:
        raise ValueError("模型中没有卷积层，无法展示特征图")

    # hook 捕获每个卷积层的输出；前向结束后统一读取。
    activations: list[torch.Tensor] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        activations.append(output.detach())

    for layer in conv_layers:
        handles.append(layer.register_forward_hook(capture))

    try:
        model.eval()
        model_device = next(model.parameters()).device
        with torch.no_grad():
            model(image.to(model_device))
    finally:
        # 无论前向是否成功，都要移除 hook，避免影响后续推理。
        for handle in handles:
            handle.remove()

    if len(activations) != len(conv_layers):
        raise RuntimeError("未能捕获全部卷积层输出")

    figure, axes = plt.subplots(
        len(conv_layers),
        max_channels,
        figsize=(max_channels * 1.1, len(conv_layers) * 1.9),
        squeeze=False,
    )

    for row, (activation, axis_row) in enumerate(zip(activations, axes)):
        # activation: [1, C, H, W] -> [C, H, W]。
        # 显示 ReLU 后的响应，负值置零后更接近实际激活。
        feature_maps = torch.relu(activation[0]).cpu()
        channel_count = min(max_channels, feature_maps.shape[0])

        for column, axis in enumerate(axis_row):
            axis.axis("off")
            if column >= channel_count:
                continue
            axis.imshow(feature_maps[column].numpy(), cmap="viridis")
            axis.set_title(f"conv{row + 1} ch{column}", fontsize=8)

    figure.suptitle("Feature maps after each conv layer", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# 错分样本
# ---------------------------------------------------------------------------
def plot_misclassified(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_path: str | Path,
    max_items: int = 16,
    max_batches: int | None = None,
) -> None:
    """展示测试集中被模型分错的样本。

    每张小图标题包含真实标签、预测标签和预测置信度。图片在显示前会
    从标准化空间还原到 [0, 1]，方便观察原始笔画。

    Args:
        model: 训练好的模型。
        test_loader: 固定顺序的测试集 DataLoader。
        device: 推理设备。
        output_path: 输出 PNG 路径。
        max_items: 最多展示多少个错分样本。
        max_batches: 冒烟测试时限制扫描的 batch 数。
    """
    if max_items <= 0:
        raise ValueError(f"max_items 必须为正数，当前为 {max_items}")
    if max_batches is not None and max_batches <= 0:
        raise ValueError(f"max_batches 必须为正数或 None，当前为 {max_batches}")

    model.eval()
    collected: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            wrong_indices = (predictions != labels).nonzero(as_tuple=False).flatten()
            for index in wrong_indices.tolist():
                if len(collected) >= max_items:
                    break

                collected.append(
                    {
                        # 保存标准化后的张量，画图时再反标准化。
                        "image": images[index].detach().cpu(),
                        "true": int(labels[index].item()),
                        "pred": int(predictions[index].item()),
                        "confidence": float(
                            probabilities[index, int(predictions[index].item())].item()
                        ),
                    }
                )

            if len(collected) >= max_items:
                break

    columns = 4
    rows = math.ceil(max_items / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.0, rows * 2.2),
        squeeze=False,
    )

    for axis in axes.flatten():
        axis.axis("off")

    if not collected:
        figure.text(
            0.5,
            0.5,
            "No misclassified samples",
            ha="center",
            va="center",
            fontsize=14,
        )
    else:
        for axis, item in zip(axes.flatten(), collected):
            # 反标准化：x_original = x_normalized * std + mean。
            display = item["image"] * MNIST_STD + MNIST_MEAN
            display = display.clamp(0.0, 1.0).squeeze(0).numpy()

            axis.imshow(display, cmap="gray")
            axis.set_title(
                f"true={item['true']} pred={item['pred']}\n"
                f"p={item['confidence']:.2f}",
                fontsize=9,
            )

    figure.suptitle(
        f"Misclassified test samples (showing {len(collected)})",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# 单图预测
# ---------------------------------------------------------------------------
def preprocess_single_image(
    image_path: str | Path,
    invert: str = "auto",
) -> torch.Tensor:
    """把普通图片转换成模型输入张量。

    处理步骤：

    1. 转为灰度图；
    2. 根据背景亮度决定是否反色（MNIST 是黑底白字）；
    3. 缩放到 28x28；
    4. 缩放到 [0, 1] 并做与训练集一致的标准化。

    Args:
        image_path: 输入图片路径，支持 PIL 能读取的格式。
        invert: ``"auto"``、``"yes"`` 或 ``"no"``。
            ``"auto"`` 在图片平均亮度大于 127 时自动反色，
            适合白底黑字的手写图片。

    Returns:
        形状为 ``[1, 1, 28, 28]`` 的 float32 张量。
    """
    if invert not in ("auto", "yes", "no"):
        raise ValueError(f"invert 必须是 'auto'、'yes' 或 'no'，当前为 {invert!r}")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到图片：{path}")

    with Image.open(path) as source:
        image = source.convert("L")

    # 自动判断背景：白底图片平均亮度高，需要反色成黑底白字。
    if invert == "yes" or (
        invert == "auto" and ImageStat.Stat(image).mean[0] > 127
    ):
        image = ImageOps.invert(image)

    # 简单缩放到 28x28；演示图片应尽量让数字位于画面中央。
    image = image.resize((28, 28), Image.Resampling.BILINEAR)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD]),
        ]
    )

    # transform 输出 [1, 28, 28]，补上 batch 维变成 [1, 1, 28, 28]。
    return transform(image).unsqueeze(0)


def predict_single_image(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> tuple[int, torch.Tensor]:
    """对单张图片做预测。

    Args:
        model: 训练好的模型。
        image: 形状为 ``[1, 1, 28, 28]`` 的输入张量。
        device: 推理设备。

    Returns:
        ``(预测类别, 10 类概率)``。概率在 CPU 上，便于打印和画图。
    """
    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError(f"image 形状应为 [1, 1, 28, 28]，实际为 {tuple(image.shape)}")

    model.eval()
    with torch.no_grad():
        logits = model(image.to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    predicted = int(probabilities.argmax().item())
    return predicted, probabilities


def plot_prediction(
    image: torch.Tensor,
    probabilities: torch.Tensor,
    predicted: int,
    output_path: str | Path,
    true_label: int | None = None,
) -> None:
    """保存单图预测结果：左侧输入图，右侧 10 类概率柱状图。"""
    # 反标准化后显示，避免把标准化后的负数直接画出来。
    display = image[0, 0] * MNIST_STD + MNIST_MEAN
    display = display.clamp(0.0, 1.0).numpy()

    figure, (image_axis, bar_axis) = plt.subplots(1, 2, figsize=(10, 4))

    image_axis.imshow(display, cmap="gray")
    image_axis.set_title("Input image (denormalized)")
    image_axis.axis("off")

    classes = list(range(10))
    bars = bar_axis.bar(classes, probabilities.numpy(), color="tab:blue")
    bars[predicted].set_color("tab:red")

    bar_axis.set_xticks(classes)
    bar_axis.set_xlabel("Digit")
    bar_axis.set_ylabel("Probability")
    bar_axis.set_ylim(0.0, 1.0)
    bar_axis.grid(axis="y", alpha=0.3)

    title = f"Prediction: {predicted} (p={probabilities[predicted].item():.3f})"
    if true_label is not None:
        title += f"\nTrue label: {true_label}"
    bar_axis.set_title(title)

    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# 一次性完成某个运行目录的全部评估
# ---------------------------------------------------------------------------
def evaluate_run(
    run_dir: str | Path,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """对一个训练运行目录生成全部测试指标和图表。

    产物写入该运行目录：

        - test_metrics.json    各轮测试指标
        - curves.png           训练/测试曲线
        - kernels.png          第一层卷积核（仅 CNN）
        - feature_maps.png     卷积特征图（仅 CNN）
        - misclassified.png    错分样本
        - evaluation.json      本次评估摘要

    Args:
        run_dir: 训练入口创建的运行目录。
        device: 评估设备。
        max_batches: 冒烟测试时限制测试 batch 数。

    Returns:
        评估摘要字典。
    """
    run_path = Path(run_dir)
    history_path = run_path / "history.json"

    if not history_path.exists():
        raise FileNotFoundError(f"找不到训练历史：{history_path}")

    history_data = load_json(history_path)
    config = history_data.get("config", {})

    stage = config.get("stage", "basic")
    batch_size = int(config.get("batch_size") or 128)

    # 测试集始终只做确定性预处理；stage 只影响训练集。
    test_loader = build_loader(
        stage=stage,
        train=False,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print(f"evaluating: {run_path}", flush=True)

    # 1. 各轮权重的测试指标 + 曲线。
    test_metrics = evaluate_checkpoints(
        run_dir=run_path,
        test_loader=test_loader,
        device=device,
        max_batches=max_batches,
    )
    plot_learning_curves(
        history_data=history_data,
        test_metrics=test_metrics,
        output_path=run_path / "curves.png",
    )

    # 2. 加载末轮模型，生成卷积核、特征图和错分样本。
    final_checkpoint = run_path / "weights" / "final.pt"
    if not final_checkpoint.exists():
        candidates = sorted((run_path / "weights").glob("epoch_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"{run_path / 'weights'} 中没有可用权重")
        final_checkpoint = candidates[-1]

    model, checkpoint = load_model_from_checkpoint(final_checkpoint, device)
    model_name = checkpoint.get("config", {}).get(
        "model", config.get("model", "unknown")
    )

    figures = ["curves.png"]

    if model_name == "cnn":
        plot_conv_kernels(model, run_path / "kernels.png")
        figures.append("kernels.png")

        # 取测试集第一张图，展示它经过两个卷积层后的响应。
        images, _ = next(iter(test_loader))
        plot_feature_maps(model, images[0:1], run_path / "feature_maps.png")
        figures.append("feature_maps.png")
    else:
        print("MLP has no conv layers; skip kernels and feature maps.", flush=True)

    plot_misclassified(
        model=model,
        test_loader=test_loader,
        device=device,
        output_path=run_path / "misclassified.png",
        max_batches=max_batches,
    )
    figures.append("misclassified.png")

    summary = {
        "run_dir": str(run_path),
        "config": config,
        "test_metrics": test_metrics,
        "figures": figures,
    }
    save_json(run_path / "evaluation.json", summary)

    return summary
