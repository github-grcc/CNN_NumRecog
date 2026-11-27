"""MNIST 手写数字识别网络。

本模块只定义网络结构，不包含数据读取、训练或评估逻辑。
两个模型遵循同一套输入输出约定：

    输入:  [B, 1, 28, 28] 的 float32 张量
    输出:  [B, 10] 的 logits，第 i 列对应数字 i

训练时把 logits 和 [B] 的整数标签直接交给 ``CrossEntropyLoss``；
预测时对第 1 维做 ``argmax`` 得到数字类别。
"""

from __future__ import annotations

import torch
from torch import nn

# ---------------------------------------------------------------------------
# 网络尺寸常量
# ---------------------------------------------------------------------------
# MNIST 图片是单通道 28x28 灰度图。
INPUT_CHANNELS = 1
INPUT_HEIGHT = 28
INPUT_WIDTH = 28

# 数字 0-9 共 10 类。
NUM_CLASSES = 10

# CNN 分类头隐藏层宽度。
CNN_HIDDEN_DIM = 64

# MLP 隐藏层宽度。选 133 是为了让 MLP 的参数量接近 CNN：
#   MLP 参数量 = 784 * 133 + 133 + 133 * 10 + 10 = 105,745
#   CNN 参数量 = 105,866，两者只差 121 个参数。
MLP_HIDDEN_DIM = 133

# 经过两次 2x2 最大池化后，特征图从 28x28 变为 7x7。
CNN_FLATTEN_DIM = 32 * 7 * 7  # 32 个通道 * 7 * 7 = 1568


class CNN(nn.Module):
    """双卷积 CNN。

    结构（与 workflow.md 中的表格一致）::

        INPUT [B, 1, 28, 28]
          -> Conv2d(1, 16, kernel_size=3, stride=1, padding=1) -> ReLU
          -> MaxPool2d(kernel_size=2, stride=2)
          -> Conv2d(16, 32, kernel_size=3, stride=1, padding=1) -> ReLU
          -> MaxPool2d(kernel_size=2, stride=2)
          -> Flatten(start_dim=1)
          -> Linear(1568, 64) -> ReLU
          -> Linear(64, 10)
          -> logits [B, 10]

    可训练参数合计 105,866 个。
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # 特征提取部分：两个“卷积 -> ReLU -> 池化”块。
        # ------------------------------------------------------------------
        self.features = nn.Sequential(
            # 第 1 个卷积块。
            # 输入 [B, 1, 28, 28]；padding=1 且 stride=1，所以空间尺寸保持 28x28。
            # 输出 [B, 16, 28, 28]。
            # 参数量 = 16 * 1 * 3 * 3 + 16 = 160。
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # ReLU 不改变形状，只把负响应置零，给网络引入非线性。
            nn.ReLU(inplace=True),
            # 2x2 最大池化：每个 2x2 区域取最大值。
            # 输出 [B, 16, 14, 14]，没有可训练参数。
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第 2 个卷积块。
            # 输入 [B, 16, 14, 14]；输出 [B, 32, 14, 14]。
            # 每个输出通道会聚合全部 16 个输入通道的 3x3 响应。
            # 参数量 = 32 * 16 * 3 * 3 + 32 = 4,640。
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            # 第二次池化：输出 [B, 32, 7, 7]，没有可训练参数。
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ------------------------------------------------------------------
        # 分类部分：把 32x7x7 的特征图展平后接两层全连接。
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            # 只展平通道和空间维度，保留批量维 B。
            # [B, 32, 7, 7] -> [B, 1568]。
            nn.Flatten(start_dim=1),
            # 全连接隐藏层：1568 -> 64。
            # 参数量 = 1568 * 64 + 64 = 100,416。
            nn.Linear(in_features=CNN_FLATTEN_DIM, out_features=CNN_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            # 输出层：64 -> 10，得到每个数字类别的 logit。
            # 参数量 = 64 * 10 + 10 = 650。
            nn.Linear(in_features=CNN_HIDDEN_DIM, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 形状为 ``[B, 1, 28, 28]`` 的 float32 图像张量。

        Returns:
            形状为 ``[B, 10]`` 的 logits。这里不做 softmax，
            因为 ``CrossEntropyLoss`` 内部会把 logits 转成概率。
        """
        # [B, 1, 28, 28] -> [B, 32, 7, 7]
        features = self.features(x)
        # [B, 32, 7, 7] -> [B, 10]
        return self.classifier(features)


class MLP(nn.Module):
    """与 CNN 参数量接近的全连接对照网络。

    结构（与 workflow.md 一致）::

        INPUT [B, 1, 28, 28]
          -> Flatten(start_dim=1)          # [B, 784]
          -> Linear(784, 133) -> ReLU      # [B, 133]
          -> Linear(133, 10)               # [B, 10]
          -> logits

    可训练参数合计 105,745 个。
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            # 把每张 28x28 图片拉平成 784 维向量。
            # 与 CNN 不同，这里不再保留任何空间结构。
            nn.Flatten(start_dim=1),
            # 隐藏层：784 -> 133。
            # 参数量 = 784 * 133 + 133 = 104,405。
            nn.Linear(
                in_features=INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH,
                out_features=MLP_HIDDEN_DIM,
            ),
            nn.ReLU(inplace=True),
            # 输出层：133 -> 10。
            # 参数量 = 133 * 10 + 10 = 1,340。
            nn.Linear(in_features=MLP_HIDDEN_DIM, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 形状为 ``[B, 1, 28, 28]`` 的 float32 图像张量。

        Returns:
            形状为 ``[B, 10]`` 的 logits。
        """
        return self.layers(x)


def build_model(name: str) -> nn.Module:
    """按名称创建模型。

    Args:
        name: ``"cnn"`` 或 ``"mlp"``，大小写不敏感。

    Returns:
        新建的模型实例，权重使用 PyTorch 默认初始化。

    Raises:
        ValueError: 名称不是支持的模型。
    """
    normalized = name.strip().lower()
    if normalized == "cnn":
        return CNN()
    if normalized == "mlp":
        return MLP()
    raise ValueError(f"未知模型 {name!r}，可选值为 'cnn' 或 'mlp'")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """统计模型参数量。

    Args:
        model: 任意 PyTorch 模型。
        trainable_only: 为 ``True`` 时只统计 ``requires_grad=True`` 的参数，
            与实验文档中的“可训练参数”口径一致。

    Returns:
        参数个数。
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def _smoke_test() -> None:
    """最小自检：检查输出形状和参数量。

    直接运行 ``python -m src.models`` 时会执行本函数。
    """
    # 用 2 张随机图片模拟一个 batch，验证两个模型都接受 [2, 1, 28, 28]。
    dummy_input = torch.randn(2, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

    expected_parameters = {
        "cnn": 105_866,
        "mlp": 105_745,
    }

    for model_name in ("cnn", "mlp"):
        model = build_model(model_name)
        model.eval()

        # 推理时不需要梯度，减少显存/内存占用。
        with torch.no_grad():
            output = model(dummy_input)

        parameter_count = count_parameters(model)
        print(
            f"{model_name.upper():<3} "
            f"params={parameter_count:,} "
            f"output={tuple(output.shape)}"
        )

        # 形状和参数量是本实验的硬性约定，不符合时直接报错。
        assert output.shape == (2, NUM_CLASSES), (
            f"{model_name} 输出形状应为 (2, {NUM_CLASSES})，实际为 {tuple(output.shape)}"
        )
        assert parameter_count == expected_parameters[model_name], (
            f"{model_name} 参数量应为 {expected_parameters[model_name]:,}，"
            f"实际为 {parameter_count:,}"
        )

    print("models smoke test: ok")


if __name__ == "__main__":
    _smoke_test()
