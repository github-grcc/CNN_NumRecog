"""训练入口的核心逻辑：设备选择、训练循环、资源计量和权重保存。

本模块只负责“训练一个模型”的流程，不负责命令行参数解析（见 main.py），
也不负责测试曲线和可视化（见 evaluate.py）。

训练流程（与 experiment.md 一致）::

    set_global_seed(seed)
      -> 每轮 set_epoch_seed(seed, epoch)  # 同步数据增强随机状态
      -> 遍历 DataLoader（计时开始）
      -> 前向、损失、反向、Adam 更新
      -> 记录损失、准确率、耗时（计时结束）
      -> 保存该轮权重
      -> 训练结束后在测试集上评估末轮模型

计时口径：
    - 每轮耗时包含数据读取、预处理、数据传输、前向、反向和参数更新。
    - CUDA 在每个计时边界调用 ``torch.cuda.synchronize()``。
    - 保存权重、测试评估和后续画图不计入每轮训练耗时。

资源计量：
    - CPU 时间：进程 ``cpu_times()`` 的 user + system 增量。
    - 内存：后台线程定期采样进程 RSS，记录训练期间峰值。
    - GPU：训练前重置峰值统计，训练后读取 allocated/reserved 峰值。
"""

from __future__ import annotations

import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import set_epoch_seed
from src.models import build_model, count_parameters
from src.utils import (
    collect_environment,
    save_installed_packages,
    save_json,
)


# ---------------------------------------------------------------------------
# 随机种子与确定性设置
# ---------------------------------------------------------------------------
def configure_determinism() -> None:
    """关闭会引入额外随机性的加速选项。

    实验文档要求 CUDA 路径关闭 AMP 和 TF32。这里同时关闭 cuDNN 的
    自动算法选择，尽量让同一台机器上的重复实验保持一致。
    """
    # cuDNN 使用确定性算法，并关闭 benchmark 自动调优。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 关闭 TF32；本实验统一使用 FP32。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # 让 float32 矩阵乘法使用最高精度设置。
    torch.set_float32_matmul_precision("highest")


def set_global_seed(seed: int) -> None:
    """设置 Python 和 PyTorch 的全局随机种子。

    注意：模型初始化也会消耗随机数，因此正式实验应在创建模型之前调用本函数。
    ``fit`` 内部会再次调用，主要用于保证数据增强和训练过程可复现。

    Args:
        seed: 实验主种子，例如 42、123 或 2026。
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # CUDA 可用时同步所有 GPU 的随机状态；CPU 机器上该调用无副作用。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    configure_determinism()


# ---------------------------------------------------------------------------
# 设备选择与 CUDA 自检
# ---------------------------------------------------------------------------
# 在独立进程中执行的小脚本：真正创建 CUDA 张量并完成一次卷积前向/反向。
# 使用独立进程可以避免父进程的 CUDA 状态影响判断，也便于捕获驱动错误。
_CUDA_SELF_TEST_CODE = r"""
import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is False")

device = torch.device("cuda")
torch.manual_seed(0)

# 用与 MNIST 输入相同的形状做一次最小卷积训练步骤。
x = torch.randn(2, 1, 28, 28, device=device)
conv = torch.nn.Conv2d(1, 4, kernel_size=3, padding=1).to(device)
y = conv(x)
loss = y.square().mean()
loss.backward()
torch.cuda.synchronize(device)

print("CUDA_SELF_TEST_OK")
"""


def cuda_self_test(timeout_seconds: int = 120) -> tuple[bool, str]:
    """在独立进程中检查 CUDA 是否真的可以完成卷积前向和反向。

    Args:
        timeout_seconds: 子进程超时时间，防止驱动卡死。

    Returns:
        ``(是否通过, 说明文字)``。失败时说明文字包含子进程的错误输出。
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CUDA_SELF_TEST_CODE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"CUDA 自检超时（>{timeout_seconds}s）"
    except Exception as error:  # 包括无法启动子进程等情况
        return False, f"CUDA 自检启动失败：{error}"

    if result.returncode == 0 and "CUDA_SELF_TEST_OK" in result.stdout:
        return True, "CUDA 自检通过"

    # 优先返回 stderr，其次返回 stdout，方便定位驱动或显存问题。
    detail = (result.stderr or result.stdout).strip()
    if not detail:
        detail = f"子进程退出码 {result.returncode}"
    return False, detail


def resolve_device(mode: str) -> tuple[torch.device, str]:
    """根据命令行选项选择训练设备。

    Args:
        mode: ``"auto"``、``"cuda"`` 或 ``"cpu"``。

    Returns:
        ``(device, 说明)``。说明会写入实验配置，便于解释为什么使用 CPU/GPU。

    Raises:
        ValueError: ``mode`` 不是支持的选项。
        RuntimeError: 显式要求 CUDA 但自检失败。
    """
    normalized = mode.strip().lower()

    if normalized == "cpu":
        return torch.device("cpu"), "用户显式指定 CPU"

    if normalized == "cuda":
        passed, detail = cuda_self_test()
        if not passed:
            raise RuntimeError(f"CUDA 自检失败，无法使用 --device cuda：{detail}")
        return torch.device("cuda"), detail

    if normalized == "auto":
        # 先做轻量检查；没有 CUDA 时不必启动自检子进程。
        if not torch.cuda.is_available():
            return torch.device("cpu"), "CUDA 不可用，auto 自动回退 CPU"

        passed, detail = cuda_self_test()
        if passed:
            return torch.device("cuda"), detail

        # auto 模式允许回退，但要把原因记录下来。
        return torch.device("cpu"), f"CUDA 自检失败，auto 回退 CPU：{detail}"

    raise ValueError(
        f"未知设备选项 {mode!r}，可选值为 'auto'、'cuda' 或 'cpu'"
    )


# ---------------------------------------------------------------------------
# 资源计量
# ---------------------------------------------------------------------------
class ResourceMonitor:
    """采样训练期间的 CPU 时间、RSS 峰值和 CUDA 显存峰值。

    使用后台线程按固定间隔采样进程 RSS。CPU 时间使用进程级
    ``psutil.Process.cpu_times()``，GPU 使用 PyTorch 的峰值统计接口。

    Args:
        device: 实际训练设备。
        interval_seconds: RSS 采样间隔，默认 0.1 秒。
    """

    def __init__(self, device: torch.device, interval_seconds: float = 0.1) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._peak_rss_bytes = 0
        self._cpu_start_seconds = 0.0
        self._cpu_end_seconds = 0.0
        self._gpu_peak_allocated_bytes: int | None = None
        self._gpu_peak_reserved_bytes: int | None = None

    def _process_cpu_seconds(self) -> float:
        """返回当前进程累计的 user + system CPU 时间（秒）。"""
        times = self.process.cpu_times()
        return times.user + times.system

    def _sample_loop(self) -> None:
        """后台线程：定期读取 RSS 并更新峰值。"""
        while not self._stop_event.wait(self.interval_seconds):
            try:
                rss = self.process.memory_info().rss
            except psutil.Error:
                # 进程异常结束时停止采样，不影响主流程。
                return

            with self._lock:
                if rss > self._peak_rss_bytes:
                    self._peak_rss_bytes = rss

    def start(self) -> None:
        """开始计量。

        CUDA 路径会先同步并重置显存峰值统计，确保只统计训练区间。
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        self._peak_rss_bytes = self.process.memory_info().rss
        self._cpu_start_seconds = self._process_cpu_seconds()
        self._gpu_peak_allocated_bytes = None
        self._gpu_peak_reserved_bytes = None

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """结束计量并读取最终峰值。"""
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._cpu_end_seconds = self._process_cpu_seconds()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self._gpu_peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
            self._gpu_peak_reserved_bytes = torch.cuda.max_memory_reserved(self.device)

    def summary(self) -> dict[str, Any]:
        """返回可直接写入 JSON 的计量结果。

        CPU 组的 GPU 字段为 ``None``，与实验文档中的 ``N/A`` 对应。
        """
        mebibyte = 1024**2

        with self._lock:
            peak_rss = self._peak_rss_bytes

        result: dict[str, Any] = {
            "cpu_time_seconds": max(
                0.0, self._cpu_end_seconds - self._cpu_start_seconds
            ),
            "peak_rss_mib": peak_rss / mebibyte,
            "gpu_peak_allocated_mib": None,
            "gpu_peak_reserved_mib": None,
        }

        if self._gpu_peak_allocated_bytes is not None:
            result["gpu_peak_allocated_mib"] = (
                self._gpu_peak_allocated_bytes / mebibyte
            )
        if self._gpu_peak_reserved_bytes is not None:
            result["gpu_peak_reserved_mib"] = (
                self._gpu_peak_reserved_bytes / mebibyte
            )

        return result


# ---------------------------------------------------------------------------
# 单轮训练与评估
# ---------------------------------------------------------------------------
def _move_batch_to_device(
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把一个 batch 移到训练设备。

    CUDA 路径使用 ``non_blocking=True``，配合 DataLoader 的 ``pin_memory``
    可以让主机到设备的拷贝与计算重叠。CPU 路径保持普通拷贝。
    """
    non_blocking = device.type == "cuda"
    return (
        images.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """训练一个 epoch，返回平均损失和准确率。

    Args:
        model: 待训练模型。
        loader: 训练集 DataLoader。
        optimizer: 优化器。
        criterion: 损失函数，通常是 ``CrossEntropyLoss``。
        device: 训练设备。
        max_batches: 只运行前若干个 batch，用于快速冒烟测试；``None`` 表示
            运行完整数据集。

    Returns:
        包含 ``loss``、``accuracy``、``samples`` 和 ``batches`` 的字典。
    """
    if max_batches is not None and max_batches <= 0:
        raise ValueError(f"max_batches 必须为正数或 None，当前为 {max_batches}")

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0

    for batch_index, (images, labels) in enumerate(loader):
        # 冒烟测试时提前结束，避免在 CPU 上跑完整数据集。
        if max_batches is not None and batch_index >= max_batches:
            break

        images, labels = _move_batch_to_device(images, labels, device)

        # set_to_none=True 比把梯度清零更省内存。
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # loss.item() 是整个 batch 的平均损失，乘 batch 大小后便于求全局平均。
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
        batch_count += 1

    if total_samples == 0:
        raise RuntimeError("训练 DataLoader 没有产生任何样本")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
        "batches": batch_count,
    }


@torch.no_grad()
def evaluate_loss_acc(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """在给定数据集上计算损失和准确率，不更新参数。

    用于测试集评估；也可以在训练过程中做快速验证。
    """
    if max_batches is not None and max_batches <= 0:
        raise ValueError(f"max_batches 必须为正数或 None，当前为 {max_batches}")

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images, labels = _move_batch_to_device(images, labels, device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
        batch_count += 1

    if total_samples == 0:
        raise RuntimeError("评估 DataLoader 没有产生任何样本")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
        "batches": batch_count,
    }


# ---------------------------------------------------------------------------
# 权重保存与完整训练流程
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> None:
    """保存模型权重和必要的元数据。

    只保存 ``state_dict``，不保存整个模型对象，加载时更稳定。权重先复制到
    CPU，避免保存过程中继续占用 GPU 显存。

    Args:
        path: 输出文件，例如 ``weights/epoch_01.pt``。
        model: 当前模型。
        epoch: 轮次，从 1 开始。
        config: 实验配置，加载模型时用于确认结构和预处理。
        metrics: 该轮的训练/测试指标，可选。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }

    payload = {
        "model_state_dict": state_dict,
        "epoch": epoch,
        "config": config,
        "metrics": metrics or {},
    }
    torch.save(payload, output_path)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    test_loader: DataLoader | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """执行完整训练流程并保存权重、配置、历史和资源指标。

    Args:
        model: 已创建并移动到 ``device`` 的模型。
        train_loader: 训练集 DataLoader。
        test_loader: 测试集 DataLoader；传入时在训练结束后评估末轮模型。
        epochs: 训练轮数，正式实验为 10。
        lr: Adam 学习率，正式实验为 0.001。
        seed: 实验主种子。
        device: 训练设备。
        output_dir: 本次运行的输出目录。
        config: 额外实验配置（阶段、模型名等），会写入 ``config.json``。
        max_batches: 冒烟测试时限制每轮 batch 数；``None`` 表示完整训练。

    Returns:
        包含 ``config``、``history``、``resources``、``test`` 等字段的字典，
        同时写入 ``<output_dir>/history.json``。
    """
    if epochs <= 0:
        raise ValueError(f"epochs 必须为正数，当前为 {epochs}")
    if lr <= 0:
        raise ValueError(f"lr 必须为正数，当前为 {lr}")

    output_path = Path(output_dir)
    weights_dir = output_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 合并调用方配置和 fit 自身的参数，保证配置文件包含完整实验条件。
    run_config: dict[str, Any] = dict(config or {})
    run_config.setdefault("epochs", epochs)
    run_config.setdefault("lr", lr)
    run_config.setdefault("seed", seed)
    run_config.setdefault("device", str(device))
    run_config.setdefault("batch_size", train_loader.batch_size)
    run_config.setdefault("num_workers", train_loader.num_workers)
    run_config.setdefault("max_batches", max_batches)
    run_config.setdefault("parameter_count", count_parameters(model))

    # 在训练开始前保存配置、环境和包版本，即使训练中断也能追溯条件。
    save_json(output_path / "config.json", run_config)
    save_json(output_path / "env.json", collect_environment(device))
    save_installed_packages(output_path / "pip-freeze.txt")

    # 再次设置种子，确保数据增强和训练过程从确定状态开始。
    # 注意：模型权重初始化应在此之前完成，main.py 会先调用 set_global_seed。
    set_global_seed(seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    monitor = ResourceMonitor(device)
    history: list[dict[str, Any]] = []

    monitor.start()
    try:
        for epoch in range(epochs):
            # 每轮开始前重置数据增强的随机状态。
            # CNN 和 MLP 使用相同的 seed/epoch，才能得到相同的仿射参数。
            set_epoch_seed(seed, epoch)

            # CUDA 计时边界：先同步，再记录开始时间。
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_start = time.perf_counter()

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                max_batches=max_batches,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_seconds = time.perf_counter() - epoch_start

            # 吞吐量按实际处理样本数计算；冒烟测试时样本数会小于 60,000。
            throughput = train_metrics["samples"] / epoch_seconds

            record = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_samples": train_metrics["samples"],
                "train_seconds": epoch_seconds,
                "throughput_samples_per_second": throughput,
            }
            history.append(record)

            # 保存该轮权重；保存时间不计入上面的 epoch_seconds。
            save_checkpoint(
                weights_dir / f"epoch_{epoch + 1:02d}.pt",
                model=model,
                epoch=epoch + 1,
                config=run_config,
                metrics=record,
            )

            print(
                f"[epoch {epoch + 1:02d}/{epochs:02d}] "
                f"loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.4f} "
                f"time={epoch_seconds:.2f}s "
                f"throughput={throughput:.1f} samples/s",
                flush=True,
            )
    finally:
        # 即使训练中途异常，也要停止采样线程并保留已测得的资源数据。
        monitor.stop()

    resources = monitor.summary()
    total_training_seconds = sum(
        record["train_seconds"] for record in history
    )

    # 测试评估放在训练计时和资源计量区间之外，符合实验文档的口径。
    test_metrics: dict[str, Any] | None = None
    if test_loader is not None:
        test_metrics = evaluate_loss_acc(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            max_batches=max_batches,
        )
        print(
            f"[test] loss={test_metrics['loss']:.4f} "
            f"acc={test_metrics['accuracy']:.4f}",
            flush=True,
        )

    # 保存末轮权重；如果训练了多轮，final.pt 与最后一个 epoch_XX.pt 内容相近。
    save_checkpoint(
        weights_dir / "final.pt",
        model=model,
        epoch=epochs,
        config=run_config,
        metrics={"test": test_metrics} if test_metrics else {},
    )

    result = {
        "config": run_config,
        "history": history,
        "resources": resources,
        "test": test_metrics,
        "total_training_seconds": total_training_seconds,
        "environment": collect_environment(device),
    }
    save_json(output_path / "history.json", result)

    return result


def _smoke_test() -> None:
    """用随机张量检查训练、保存和评估流程，不依赖 MNIST。

    直接运行 ``python -m src.train`` 时会执行本函数。
    """
    import tempfile

    from torch.utils.data import TensorDataset

    # 16 张随机图片和随机标签，只用于验证代码路径。
    images = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=False)

    model = build_model("cnn")
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as temporary_directory:
        result = fit(
            model=model,
            train_loader=loader,
            test_loader=loader,
            epochs=1,
            lr=1e-3,
            seed=42,
            device=device,
            output_dir=temporary_directory,
            config={"stage": "basic", "model": "cnn", "smoke_test": True},
            max_batches=1,
        )

        assert result["history"], "训练历史不应为空"
        assert result["test"] is not None, "冒烟测试应产生测试指标"
        assert (Path(temporary_directory) / "weights" / "final.pt").exists()

    print(
        "train smoke test: ok "
        f"train_acc={result['history'][0]['train_accuracy']:.4f} "
        f"test_acc={result['test']['accuracy']:.4f}"
    )


if __name__ == "__main__":
    _smoke_test()
