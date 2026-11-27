"""多种子实验汇总。

批量训练会为每个“阶段 × 模型 × 种子”生成一个独立运行目录，每个目录内
都有 ``history.json``。本模块负责：

    1. 递归收集这些运行结果；
    2. 按阶段/模型汇总测试准确率、损失、耗时和资源指标；
    3. 计算 ``CNN - MLP`` 和 ``增强 - 基础`` 的差值；
    4. 输出 ``summary.csv``、``summary.md`` 和 ``summary.json``。

准确率差值统一用百分点（percentage point, pp）表示：
``(a - b) * 100``。标准差使用样本标准差；只有一个种子时为 0。
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils import current_timestamp, load_json, save_json

# 固定展示顺序，避免不同运行目录的字母序影响表格。
_STAGE_ORDER = {"basic": 0, "augmented": 1}
_MODEL_ORDER = {"cnn": 0, "mlp": 1}


def collect_runs(root: str | Path) -> list[dict[str, Any]]:
    """递归读取运行目录下的所有 ``history.json``。

    Args:
        root: 批量实验目录或单个运行目录。

    Returns:
        每个运行一条记录，按阶段、模型、种子排序。

    Raises:
        FileNotFoundError: 根目录不存在。
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"找不到运行目录：{root_path}")

    runs: list[dict[str, Any]] = []

    for history_path in sorted(root_path.rglob("history.json")):
        data = load_json(history_path)
        config = data.get("config", {})
        test_metrics = data.get("test") or {}
        resources = data.get("resources", {})
        history = data.get("history", [])
        last_epoch = history[-1] if history else {}

        runs.append(
            {
                "run_dir": str(history_path.parent),
                "stage": config.get("stage", "unknown"),
                "model": config.get("model", "unknown"),
                "seed": config.get("seed"),
                "epochs": config.get("epochs"),
                "batch_size": config.get("batch_size"),
                "device": config.get("device"),
                "parameter_count": config.get("parameter_count"),
                "test_accuracy": test_metrics.get("accuracy"),
                "test_loss": test_metrics.get("loss"),
                "train_accuracy": last_epoch.get("train_accuracy"),
                "train_loss": last_epoch.get("train_loss"),
                "total_training_seconds": data.get("total_training_seconds"),
                "cpu_time_seconds": resources.get("cpu_time_seconds"),
                "peak_rss_mib": resources.get("peak_rss_mib"),
                "gpu_peak_allocated_mib": resources.get("gpu_peak_allocated_mib"),
                "gpu_peak_reserved_mib": resources.get("gpu_peak_reserved_mib"),
            }
        )

    runs.sort(
        key=lambda run: (
            _STAGE_ORDER.get(run["stage"], 99),
            _MODEL_ORDER.get(run["model"], 99),
            run["seed"] if run["seed"] is not None else 10**9,
        )
    )
    return runs


def _mean_std(values: list[Any]) -> tuple[float | None, float | None]:
    """计算均值和样本标准差，自动忽略 ``None``。

    Returns:
        ``(mean, std)``；没有有效值时返回 ``(None, None)``。
        只有一个有效值时标准差记为 0。
    """
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None, None

    mean = statistics.fmean(cleaned)
    std = statistics.stdev(cleaned) if len(cleaned) > 1 else 0.0
    return mean, std


def _format_value(
    value: float | None,
    digits: int = 4,
    scale: float = 1.0,
    suffix: str = "",
) -> str:
    """格式化单个数值；``None`` 显示为 N/A。"""
    if value is None:
        return "N/A"
    return f"{value * scale:.{digits}f}{suffix}"


def _format_mean_std(
    mean: float | None,
    std: float | None,
    digits: int = 4,
    scale: float = 1.0,
    suffix: str = "",
) -> str:
    """格式化 ``mean ± std``；没有数据时显示 N/A。"""
    if mean is None:
        return "N/A"
    if std is None:
        std = 0.0
    return (
        f"{mean * scale:.{digits}f}{suffix} ± "
        f"{std * scale:.{digits}f}{suffix}"
    )


def _group_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 ``(stage, model)`` 汇总多个种子的均值和标准差。"""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["stage"], run["model"])].append(run)

    summary: list[dict[str, Any]] = []
    for (stage, model), items in sorted(
        groups.items(),
        key=lambda pair: (
            _STAGE_ORDER.get(pair[0][0], 99),
            _MODEL_ORDER.get(pair[0][1], 99),
        ),
    ):
        row: dict[str, Any] = {
            "stage": stage,
            "model": model,
            "runs": len(items),
        }

        for field in (
            "test_accuracy",
            "test_loss",
            "train_accuracy",
            "total_training_seconds",
            "cpu_time_seconds",
            "peak_rss_mib",
            "gpu_peak_allocated_mib",
        ):
            mean, std = _mean_std([item.get(field) for item in items])
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = std

        summary.append(row)

    return summary


def _cnn_mlp_deltas(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """计算同一阶段、同一种子下 ``CNN - MLP`` 的准确率差。"""
    by_key = {
        (run["stage"], run["model"], run["seed"]): run
        for run in runs
    }
    stages = sorted(
        {run["stage"] for run in runs},
        key=lambda stage: _STAGE_ORDER.get(stage, 99),
    )
    seeds = sorted(
        {run["seed"] for run in runs if run["seed"] is not None}
    )

    rows: list[dict[str, Any]] = []
    for stage in stages:
        for seed in seeds:
            cnn = by_key.get((stage, "cnn", seed))
            mlp = by_key.get((stage, "mlp", seed))
            if not cnn or not mlp:
                continue
            if cnn["test_accuracy"] is None or mlp["test_accuracy"] is None:
                continue

            rows.append(
                {
                    "stage": stage,
                    "seed": seed,
                    "cnn_accuracy": cnn["test_accuracy"],
                    "mlp_accuracy": mlp["test_accuracy"],
                    "delta_pp": (cnn["test_accuracy"] - mlp["test_accuracy"]) * 100,
                }
            )

    summary: list[dict[str, Any]] = []
    for stage in stages:
        values = [row["delta_pp"] for row in rows if row["stage"] == stage]
        mean, std = _mean_std(values)
        summary.append(
            {
                "stage": stage,
                "runs": len(values),
                "mean_pp": mean,
                "std_pp": std,
            }
        )

    return rows, summary


def _augmentation_deltas(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """计算同一模型、同一种子下 ``增强 - 基础`` 的准确率和耗时差。"""
    by_key = {
        (run["stage"], run["model"], run["seed"]): run
        for run in runs
    }
    models = sorted(
        {run["model"] for run in runs},
        key=lambda model: _MODEL_ORDER.get(model, 99),
    )
    seeds = sorted(
        {run["seed"] for run in runs if run["seed"] is not None}
    )

    rows: list[dict[str, Any]] = []
    for model in models:
        for seed in seeds:
            basic = by_key.get(("basic", model, seed))
            augmented = by_key.get(("augmented", model, seed))
            if not basic or not augmented:
                continue

            accuracy_delta = None
            if (
                basic["test_accuracy"] is not None
                and augmented["test_accuracy"] is not None
            ):
                accuracy_delta = (
                    augmented["test_accuracy"] - basic["test_accuracy"]
                ) * 100

            time_delta = None
            if (
                basic["total_training_seconds"] is not None
                and augmented["total_training_seconds"] is not None
            ):
                time_delta = (
                    augmented["total_training_seconds"]
                    - basic["total_training_seconds"]
                )

            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "basic_accuracy": basic["test_accuracy"],
                    "augmented_accuracy": augmented["test_accuracy"],
                    "delta_accuracy_pp": accuracy_delta,
                    "delta_time_seconds": time_delta,
                }
            )

    summary: list[dict[str, Any]] = []
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        accuracy_mean, accuracy_std = _mean_std(
            [row["delta_accuracy_pp"] for row in model_rows]
        )
        time_mean, time_std = _mean_std(
            [row["delta_time_seconds"] for row in model_rows]
        )
        summary.append(
            {
                "model": model,
                "runs": len(model_rows),
                "mean_accuracy_pp": accuracy_mean,
                "std_accuracy_pp": accuracy_std,
                "mean_time_seconds": time_mean,
                "std_time_seconds": time_std,
            }
        )

    return rows, summary


def build_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """构建完整的汇总结构，供 JSON 和 Markdown 使用。"""
    cnn_mlp_rows, cnn_mlp_summary = _cnn_mlp_deltas(runs)
    augmentation_rows, augmentation_summary = _augmentation_deltas(runs)

    return {
        "generated_at": current_timestamp(),
        "run_count": len(runs),
        "runs": runs,
        "groups": _group_summary(runs),
        "cnn_minus_mlp": {
            "rows": cnn_mlp_rows,
            "summary": cnn_mlp_summary,
        },
        "augmented_minus_basic": {
            "rows": augmentation_rows,
            "summary": augmentation_summary,
        },
    }


def write_summary_csv(runs: list[dict[str, Any]], path: str | Path) -> None:
    """写出每次运行的明细 CSV。"""
    fieldnames = [
        "stage",
        "model",
        "seed",
        "device",
        "test_accuracy",
        "test_loss",
        "train_accuracy",
        "total_training_seconds",
        "cpu_time_seconds",
        "peak_rss_mib",
        "gpu_peak_allocated_mib",
        "gpu_peak_reserved_mib",
        "parameter_count",
        "run_dir",
    ]

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: run.get(field) for field in fieldnames})


def write_summary_markdown(summary: dict[str, Any], path: str | Path) -> None:
    """写出人类可读的 Markdown 汇总。"""
    runs = summary["runs"]
    groups = summary["groups"]
    cnn_mlp = summary["cnn_minus_mlp"]
    augmentation = summary["augmented_minus_basic"]

    lines: list[str] = [
        "# MNIST 对照实验汇总",
        "",
        f"- 运行数：{summary['run_count']}",
        f"- 生成时间：{summary['generated_at']}",
        "- 准确率差值单位：百分点（pp），计算方式为 `(a - b) * 100`。",
        "- 标准差为样本标准差；只有一个种子时为 0。",
        "",
        "## 单次运行",
        "",
        "| Stage | Model | Seed | Device | Test Acc | Test Loss | Train Acc | "
        "Train Time (s) | CPU Time (s) | Peak RSS (MiB) | GPU Peak (MiB) | Params |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for run in runs:
        lines.append(
            "| {stage} | {model} | {seed} | {device} | {test_acc} | {test_loss} | "
            "{train_acc} | {train_time} | {cpu_time} | {rss} | {gpu} | {params} |".format(
                stage=run["stage"],
                model=run["model"],
                seed=run["seed"] if run["seed"] is not None else "N/A",
                device=run["device"] or "N/A",
                test_acc=_format_value(run["test_accuracy"], 2, 100, "%"),
                test_loss=_format_value(run["test_loss"], 4),
                train_acc=_format_value(run["train_accuracy"], 2, 100, "%"),
                train_time=_format_value(run["total_training_seconds"], 2),
                cpu_time=_format_value(run["cpu_time_seconds"], 2),
                rss=_format_value(run["peak_rss_mib"], 1),
                gpu=_format_value(run["gpu_peak_allocated_mib"], 1),
                params=(
                    f"{run['parameter_count']:,}"
                    if run["parameter_count"] is not None
                    else "N/A"
                ),
            )
        )

    lines.extend(
        [
            "",
            "## 按阶段/模型汇总",
            "",
            "| Stage | Model | Runs | Test Acc (mean ± std) | Test Loss | "
            "Train Time (s) | CPU Time (s) | Peak RSS (MiB) | GPU Peak (MiB) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for group in groups:
        lines.append(
            "| {stage} | {model} | {runs} | {acc} | {loss} | {train_time} | "
            "{cpu_time} | {rss} | {gpu} |".format(
                stage=group["stage"],
                model=group["model"],
                runs=group["runs"],
                acc=_format_mean_std(
                    group["test_accuracy_mean"],
                    group["test_accuracy_std"],
                    2,
                    100,
                    "%",
                ),
                loss=_format_mean_std(
                    group["test_loss_mean"],
                    group["test_loss_std"],
                    4,
                ),
                train_time=_format_mean_std(
                    group["total_training_seconds_mean"],
                    group["total_training_seconds_std"],
                    2,
                ),
                cpu_time=_format_mean_std(
                    group["cpu_time_seconds_mean"],
                    group["cpu_time_seconds_std"],
                    2,
                ),
                rss=_format_mean_std(
                    group["peak_rss_mib_mean"],
                    group["peak_rss_mib_std"],
                    1,
                ),
                gpu=_format_mean_std(
                    group["gpu_peak_allocated_mib_mean"],
                    group["gpu_peak_allocated_mib_std"],
                    1,
                ),
            )
        )

    lines.extend(
        [
            "",
            "## CNN − MLP（测试准确率）",
            "",
            "| Stage | Seed | CNN Acc | MLP Acc | Delta (pp) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in cnn_mlp["rows"]:
        lines.append(
            f"| {row['stage']} | {row['seed']} | "
            f"{_format_value(row['cnn_accuracy'], 2, 100, '%')} | "
            f"{_format_value(row['mlp_accuracy'], 2, 100, '%')} | "
            f"{_format_value(row['delta_pp'], 2)} |"
        )

    lines.extend(
        [
            "",
            "| Stage | Runs | Mean Delta (pp) | Std (pp) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in cnn_mlp["summary"]:
        lines.append(
            f"| {row['stage']} | {row['runs']} | "
            f"{_format_value(row['mean_pp'], 2)} | "
            f"{_format_value(row['std_pp'], 2)} |"
        )

    lines.extend(
        [
            "",
            "## 增强 − 基础",
            "",
            "| Model | Seed | Basic Acc | Aug Acc | Delta Acc (pp) | Delta Time (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in augmentation["rows"]:
        lines.append(
            f"| {row['model']} | {row['seed']} | "
            f"{_format_value(row['basic_accuracy'], 2, 100, '%')} | "
            f"{_format_value(row['augmented_accuracy'], 2, 100, '%')} | "
            f"{_format_value(row['delta_accuracy_pp'], 2)} | "
            f"{_format_value(row['delta_time_seconds'], 2)} |"
        )

    lines.extend(
        [
            "",
            "| Model | Runs | Mean Delta Acc (pp) | Std (pp) | "
            "Mean Delta Time (s) | Std (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in augmentation["summary"]:
        lines.append(
            f"| {row['model']} | {row['runs']} | "
            f"{_format_value(row['mean_accuracy_pp'], 2)} | "
            f"{_format_value(row['std_accuracy_pp'], 2)} | "
            f"{_format_value(row['mean_time_seconds'], 2)} | "
            f"{_format_value(row['std_time_seconds'], 2)} |"
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_runs(
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """收集运行结果并写出 CSV、Markdown 和 JSON 汇总。

    Args:
        root: 批量实验目录或单个运行目录。
        output_dir: 汇总文件输出目录；默认与 ``root`` 相同。

    Returns:
        汇总结构，包含每次运行明细、分组统计和两类差值。

    Raises:
        FileNotFoundError: 没有找到任何 ``history.json``。
    """
    runs = collect_runs(root)
    if not runs:
        raise FileNotFoundError(f"{root} 下没有找到 history.json")

    summary = build_summary(runs)
    output_path = Path(output_dir) if output_dir is not None else Path(root)
    output_path.mkdir(parents=True, exist_ok=True)

    write_summary_csv(runs, output_path / "summary.csv")
    write_summary_markdown(summary, output_path / "summary.md")
    save_json(output_path / "summary.json", summary)

    return summary
