# MNIST 网络构建

## 输入与输出

两种模型都接收 `[B, 1, 28, 28]` 的 `float32` 张量，其中 `B` 是批量大小。模型输出形状为 `[B, 10]` 的类别分数（logits）。训练时将分数和 `[B]` 的整数标签交给 `CrossEntropyLoss`，预测时用 `argmax(dim=1)` 取数字类别。[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

## 卷积网络（CNN）

```text
INPUT → [CONV → ReLU → MAXPOOL] × 2 → FLATTEN → FC → ReLU → FC
```

| 顺序 | PyTorch 层 | 输出形状 | 可训练参数 |
| ---: | --- | --- | ---: |
| 1 | `Conv2d(1, 16, kernel_size=3, stride=1, padding=1)` → `ReLU()` | `[B, 16, 28, 28]` | 160 |
| 2 | `MaxPool2d(kernel_size=2, stride=2)` | `[B, 16, 14, 14]` | 0 |
| 3 | `Conv2d(16, 32, kernel_size=3, stride=1, padding=1)` → `ReLU()` | `[B, 32, 14, 14]` | 4,640 |
| 4 | `MaxPool2d(kernel_size=2, stride=2)` | `[B, 32, 7, 7]` | 0 |
| 5 | `Flatten(start_dim=1)` | `[B, 1568]` | 0 |
| 6 | `Linear(1568, 64)` → `ReLU()` | `[B, 64]` | 100,416 |
| 7 | `Linear(64, 10)` | `[B, 10]` | 650 |
| | **合计** | | **105,866** |

按照表中顺序将各层加入 `nn.Sequential`。卷积和线性层采用 PyTorch 默认初始化与偏置。第一次池化后，每个位置的理论感受野为 `4×4`，第二次池化后扩大到 `10×10`，便于后续分类层利用笔画及其局部组合。[PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

第一层卷积权重为 `[16, 1, 3, 3]`，可直接展示每个输出通道的 `3×3` 权重。第二层权重为 `[32, 16, 3, 3]`，每个输出通道聚合 16 个输入通道的响应。展示时可用同一张输入图查看两个卷积块的特征图，或在第二层权重图上标明所选输入通道。

## 全连接对照网络（MLP）

按 `Flatten(start_dim=1) → Linear(784, 133) → ReLU() → Linear(133, 10)` 构建。隐藏层宽度选为 `133`，全网有 **105,745** 个可训练参数，与 CNN 相差 121 个，约为 `0.11%`。

构建完毕后用 `[2, 1, 28, 28]` 的张量分别检查两个模型的前向输出，预期均为 `[2, 10]`。将单独构建模型与数据、训练和评估入口对接时，沿用这一输入输出约定。

相关文档见[环境和工程结构](environment.md)与[实验方案](experiment.md)。
