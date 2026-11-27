"""项目通用工具：运行目录、JSON 读写和环境信息。

这些函数不包含模型或训练逻辑，只负责文件和环境层面的杂务，
供 ``train.py``、``evaluate.py`` 和 ``main.py`` 共用。
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import psutil
import torch
import torchvision

# 以 src/utils.py 的位置反推项目根目录，避免依赖当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 默认实验产物目录；该目录已在 .gitignore 中排除。
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"


def current_timestamp() -> str:
    """返回适合用作目录名的本地时间戳。

    使用 ``YYYYMMDD-HHMMSS`` 格式，避免文件名中出现冒号，
    便于脚本处理和排序。
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(
    root: str | Path = DEFAULT_ARTIFACT_ROOT,
    *,
    stage: str,
    model: str,
    seed: int,
) -> Path:
    """为一次训练创建独立的输出目录。

    目录名包含时间、阶段、模型和种子，例如::

        artifacts/20261126-171500_basic_cnn_seed42/

    Args:
        root: 产物根目录。相对路径会相对于项目根目录解析。
        stage: ``"basic"`` 或 ``"augmented"``。
        model: ``"cnn"`` 或 ``"mlp"``。
        seed: 随机种子。

    Returns:
        已创建的空目录路径。

    如果同一秒内重复启动相同配置，会自动追加 ``_2``、``_3`` 等后缀，
    避免覆盖已有结果。
    """
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT / root_path

    base_name = f"{current_timestamp()}_{stage}_{model}_seed{seed}"
    run_dir = root_path / base_name

    # 极短时间内重复启动时，时间戳可能相同；追加序号保证目录唯一。
    suffix = 2
    while run_dir.exists():
        run_dir = root_path / f"{base_name}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_batch_dir(root: str | Path = DEFAULT_ARTIFACT_ROOT) -> Path:
    """为一次批量实验创建根目录。

    批量实验内部会按设备/批量大小创建尝试目录，例如::

        artifacts/20261126-180000_batch/
        └── cuda_b128_1/
            ├── 20261126-180001_basic_cnn_seed42/
            └── ...

    Args:
        root: 产物根目录。相对路径会相对于项目根目录解析。

    Returns:
        已创建的批量实验目录。
    """
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT / root_path

    base_name = f"{current_timestamp()}_batch"
    batch_dir = root_path / base_name

    suffix = 2
    while batch_dir.exists():
        batch_dir = root_path / f"{base_name}_{suffix}"
        suffix += 1

    batch_dir.mkdir(parents=True, exist_ok=False)
    return batch_dir


def save_json(path: str | Path, data: Any) -> None:
    """把数据写入 UTF-8 JSON 文件。

    Args:
        path: 目标文件路径，父目录不存在时会自动创建。
        data: 可被 ``json.dumps`` 序列化的对象；Path、torch.device 等
            无法直接序列化的值会通过 ``default=str`` 转成字符串。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(
        data,
        ensure_ascii=False,  # 保留中文，方便直接阅读实验记录
        indent=2,
        default=str,
    )
    output_path.write_text(text + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """读取 UTF-8 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_environment(device: torch.device | None = None) -> dict[str, Any]:
    """收集当前运行环境信息，写入实验产物便于复现。

    Args:
        device: 实际使用的设备；传入后会记录为字符串。

    Returns:
        包含 Python、PyTorch、CUDA、CPU 和 GPU 信息的字典。
    """
    environment: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_model": platform.processor() or "unknown",
    }

    if device is not None:
        environment["device"] = str(device)

    # 只有 CUDA 可用时才查询 GPU 名称，避免在 CPU 机器上触发 CUDA 初始化。
    if torch.cuda.is_available():
        environment["gpu_count"] = torch.cuda.device_count()
        environment["gpu_name"] = torch.cuda.get_device_name(0)

    return environment


def save_installed_packages(path: str | Path) -> None:
    """把当前虚拟环境中的包及版本写入文本文件。

    效果接近 ``uv pip freeze``，但直接使用标准库 ``importlib.metadata``，
    不依赖外部命令，方便在训练入口中自动保存版本清单。

    Args:
        path: 输出文件路径，例如 ``<run_dir>/pip-freeze.txt``。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    packages: list[str] = []
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.append(f"{name}=={distribution.version}")

    # 按名称排序，保证每次生成的清单顺序稳定，便于比较。
    output_path.write_text(
        "\n".join(sorted(packages, key=str.lower)) + "\n",
        encoding="utf-8",
    )


def format_duration(seconds: float) -> str:
    """把秒数格式化为易读字符串，用于日志输出。"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{remainder:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes)}m{remainder:.0f}s"
