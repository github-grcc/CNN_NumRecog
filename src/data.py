"""MNIST 数据读取、预处理和 DataLoader。

数据流程（与 experiment.md 一致）::

    MNIST 原始图片 (PIL, 28x28 灰度)
      -> [仅“增强阶段 + 训练集”] RandomAffine
      -> ToImage()                        # CHW uint8 张量
      -> ToDtype(float32, scale=True)     # 像素缩放到 [0, 1]
      -> Normalize(mean=0.13066, std=0.30811)
      -> DataLoader(batch_size=128, num_workers=0)

两个阶段：

    basic      训练/测试都只做缩放和标准化。
    augmented  训练集额外做随机仿射；测试集仍然只做缩放和标准化。

增强同步（实验文档要求）：
    同一个种子下，CNN 和 MLP 必须看到相同的数据顺序和相同的仿射参数。
    本模块通过以下两条约定实现：

    1. 两个 DataLoader 使用相同的 ``shuffle_seed``，因此每轮的样本顺序相同。
    2. 每轮开始迭代前调用 ``set_epoch_seed(seed, epoch)``，重置全局随机状态。
       在 ``num_workers=0`` 时，transform 在主进程执行，两个模型会从相同的
       随机状态开始，对同一张图生成相同的 RandomAffine 参数。

    因此正式实验必须保持 ``num_workers=0``；改成多进程后，每个 worker 有独立
    随机状态，需要额外的 worker 种子同步逻辑。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import InterpolationMode, v2

# ---------------------------------------------------------------------------
# 路径与数据统计常量
# ---------------------------------------------------------------------------
# 以 src/data.py 的位置反推项目根目录，避免依赖启动脚本时的当前工作目录。
# 例如：/path/to/CNN_NumRecog/src/data.py -> /path/to/CNN_NumRecog
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# torchvision 会把 MNIST 放到 <root>/MNIST/raw/。
# 项目文档约定使用仓库根目录下的 data/。
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

# 从 MNIST 官方训练集计算出的均值和标准差。
# 注意：测试集也使用这组训练集统计量，避免数据泄漏。
MNIST_MEAN = 0.13066
MNIST_STD = 0.30811

# 支持的两个实验阶段。
SUPPORTED_STAGES = ("basic", "augmented")
Stage = Literal["basic", "augmented"]


def _normalize_stage(stage: str) -> Stage:
    """把阶段名称规范化为 ``"basic"`` 或 ``"augmented"``。

    Args:
        stage: 用户传入的阶段名称，大小写不敏感。

    Returns:
        规范化后的阶段名称。

    Raises:
        ValueError: 阶段名称不受支持。
    """
    normalized = stage.strip().lower()
    if normalized == "basic":
        return "basic"
    if normalized == "augmented":
        return "augmented"
    raise ValueError(
        f"未知实验阶段 {stage!r}，可选值为 {SUPPORTED_STAGES}"
    )


def build_transforms(stage: str, train: bool) -> v2.Compose:
    """构建指定阶段的数据预处理流水线。

    Args:
        stage: ``"basic"`` 或 ``"augmented"``。
        train: 是否为训练集。只有“增强阶段 + 训练集”才会加入随机仿射；
            测试集始终使用确定性的缩放和标准化。

    Returns:
        一个 ``torchvision.transforms.v2.Compose`` 对象。
    """
    normalized_stage = _normalize_stage(stage)

    # 按顺序收集要执行的变换。
    operations: list[Callable] = []

    if train and normalized_stage == "augmented":
        # 随机仿射只在增强阶段的训练集上使用：
        #   degrees=8           旋转角度在 [-8°, +8°] 内均匀采样
        #   translate=(0.07, 0.07)  横向/纵向最多平移图像尺寸的 7%
        #                        28 * 0.07 ≈ 2 像素
        #   interpolation=BILINEAR  双线性插值，边缘更平滑
        #   fill=0              旋转/平移后露出的空白区域填黑，与 MNIST 背景一致
        operations.append(
            v2.RandomAffine(
                degrees=8,
                translate=(0.07, 0.07),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
        )

    # ToImage 把 PIL 图片转换成 CHW 形状的张量，保持 uint8 像素值。
    operations.append(v2.ToImage())

    # ToDtype(scale=True) 在转换到 float32 的同时把像素从 [0, 255] 缩放到 [0, 1]。
    operations.append(v2.ToDtype(torch.float32, scale=True))

    # 标准化：output = (input - mean) / std。
    # mean/std 使用训练集统计量，单通道所以用长度为 1 的列表。
    operations.append(
        v2.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD])
    )

    return v2.Compose(operations)


def build_dataset(
    stage: str,
    train: bool,
    root: str | Path = DEFAULT_DATA_ROOT,
    download: bool = False,
) -> MNIST:
    """创建 MNIST 数据集。

    Args:
        stage: ``"basic"`` 或 ``"augmented"``。
        train: ``True`` 读取官方训练集（60,000 张），``False`` 读取测试集
            （10,000 张）。
        root: 数据根目录，默认是项目根目录下的 ``data/``。
            torchvision 实际读取 ``<root>/MNIST/raw/``。
        download: 数据不存在时是否自动下载。默认 ``False``，避免实验过程中
            意外访问网络；首次准备数据时可显式设为 ``True``。

    Returns:
        带有预处理流水线的 ``MNIST`` 数据集。
    """
    transform = build_transforms(stage=stage, train=train)
    return MNIST(
        root=str(root),
        train=train,
        transform=transform,
        download=download,
    )


def build_loader(
    stage: str,
    train: bool,
    *,
    batch_size: int = 128,
    shuffle_seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    root: str | Path = DEFAULT_DATA_ROOT,
    download: bool = False,
) -> DataLoader:
    """创建 MNIST DataLoader。

    Args:
        stage: ``"basic"`` 或 ``"augmented"``。
        train: 是否为训练集。
        batch_size: 每个 batch 的样本数，实验默认 128。
        shuffle_seed: 训练集打乱顺序的随机种子。传入相同种子时，不同模型会得到
            相同的样本顺序；正式实验应始终传入。测试集不需要打乱。
        num_workers: DataLoader 工作进程数。实验固定为 0，原因见模块文档：
            随机仿射的同步依赖主进程的全局随机状态。
        pin_memory: 是否使用锁页内存。CUDA 训练时由调用方设为 ``True``；
            CPU 训练保持 ``False`` 即可。
        root: 数据根目录。
        download: 数据缺失时是否下载。

    Returns:
        配置好的 ``DataLoader``。

    Raises:
        ValueError: ``batch_size`` 或 ``num_workers`` 不合法。
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正数，当前为 {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers 不能为负数，当前为 {num_workers}")

    dataset = build_dataset(
        stage=stage,
        train=train,
        root=root,
        download=download,
    )

    # DataLoader 的 shuffle 使用独立的 torch.Generator。
    # 同一个 shuffle_seed 会产生相同的样本排列，不受全局随机状态影响。
    generator: torch.Generator | None = None
    if train and shuffle_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        # 测试集必须保持固定顺序，方便按索引定位错分样本。
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # 不丢弃最后一个不完整 batch；MNIST 的 60,000/128 会剩 96 张。
        drop_last=False,
        generator=generator,
    )


def set_epoch_seed(seed: int, epoch: int) -> int:
    """为某一轮训练设置数据增强的随机种子。

    必须在每轮开始迭代 DataLoader **之前**调用，并且 CNN 和 MLP 使用相同的
    ``seed`` 和 ``epoch``。这样两个模型的 RandomAffine 会从相同的随机状态
    开始；在样本顺序相同且 ``num_workers=0`` 时，同一张图会得到相同的仿射参数。

    Args:
        seed: 本次实验的主种子，例如 42、123 或 2026。
        epoch: 当前轮次，从 0 开始。

    Returns:
        实际使用的 epoch 种子，便于写入日志。

    Raises:
        ValueError: ``epoch`` 为负数。
    """
    if epoch < 0:
        raise ValueError(f"epoch 不能为负数，当前为 {epoch}")

    # 用一个大质数把主种子和轮次混合，避免相邻轮次的种子过于接近。
    # 取模保证结果落在 torch.manual_seed 接受的范围内。
    epoch_seed = (int(seed) * 1_000_003 + int(epoch)) % (2**63 - 1)

    # 重置 PyTorch 的全局随机状态（CPU 和所有 CUDA 设备）。
    # RandomAffine 内部使用 torch 的随机数生成器，因此会受这里影响。
    torch.manual_seed(epoch_seed)

    return epoch_seed


def _smoke_test() -> None:
    """最小自检：检查数据形状、取值范围和增强同步。

    直接运行 ``python -m src.data`` 时会执行本函数。
    需要 ``data/MNIST/raw/`` 中已经存在官方数据文件。
    """
    # ------------------------------------------------------------------
    # 1. 基础阶段：检查一个训练 batch 的形状和类型。
    # ------------------------------------------------------------------
    basic_loader = build_loader(
        stage="basic",
        train=True,
        batch_size=4,
        shuffle_seed=42,
    )
    images, labels = next(iter(basic_loader))

    print(f"basic batch: images={tuple(images.shape)} labels={tuple(labels.shape)}")
    print(f"basic dtype: {images.dtype}, value range=[{images.min():.3f}, {images.max():.3f}]")

    assert images.shape == (4, 1, 28, 28), "图像 batch 形状应为 [B, 1, 28, 28]"
    assert labels.shape == (4,), "标签 batch 形状应为 [B]"
    assert images.dtype == torch.float32, "图像应转换为 float32"

    # ------------------------------------------------------------------
    # 2. 增强阶段：两个 DataLoader 使用相同的种子，检查仿射参数是否同步。
    # ------------------------------------------------------------------
    augmented_loader_a = build_loader(
        stage="augmented",
        train=True,
        batch_size=4,
        shuffle_seed=42,
    )
    augmented_loader_b = build_loader(
        stage="augmented",
        train=True,
        batch_size=4,
        shuffle_seed=42,
    )

    # 两个模型在同一轮开始前都应该调用 set_epoch_seed(seed, epoch)。
    set_epoch_seed(seed=42, epoch=0)
    images_a, labels_a = next(iter(augmented_loader_a))

    # 模拟第二个模型：重置到相同的随机状态，再读取同一轮的第一个 batch。
    set_epoch_seed(seed=42, epoch=0)
    images_b, labels_b = next(iter(augmented_loader_b))

    same_images = torch.equal(images_a, images_b)
    same_labels = torch.equal(labels_a, labels_b)

    print(f"augmented sync: images={same_images}, labels={same_labels}")
    assert same_images, "相同种子下，两个模型应得到相同的增强图像"
    assert same_labels, "相同种子下，两个模型应得到相同的标签顺序"

    # ------------------------------------------------------------------
    # 3. 测试集：确认不做随机增强，只做确定性预处理。
    # ------------------------------------------------------------------
    test_loader = build_loader(
        stage="augmented",
        train=False,
        batch_size=4,
    )
    test_images, test_labels = next(iter(test_loader))
    print(f"test batch: images={tuple(test_images.shape)} labels={tuple(test_labels.shape)}")

    assert test_images.shape == (4, 1, 28, 28)
    assert test_labels.shape == (4,)

    print("data smoke test: ok")


if __name__ == "__main__":
    _smoke_test()
