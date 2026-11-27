"""MNIST 手写数字识别实验的命令行入口。

提供六个子命令::

    python main.py train    --stage basic --model cnn --seed 42
    python main.py evaluate --run-dir artifacts/<run>
    python main.py predict  --checkpoint artifacts/<run>/weights/final.pt --image digit.png
    python main.py batch    # 2 阶段 × 2 模型 × 3 种子，训练后自动评估
    python main.py report   --run-root artifacts/<batch>/<attempt>
    python main.py gui      # 单图预测与实验对比图形界面

``batch`` 按实验文档要求为每组配置启动独立子进程，并统一选择设备；
训练完成后逐组生成测试曲线、卷积核、特征图和错分样本，最后写出汇总表。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.data import DEFAULT_DATA_ROOT, build_loader
from src.evaluate import (
    evaluate_run,
    load_model_from_checkpoint,
    plot_prediction,
    predict_single_image,
    preprocess_single_image,
)
from src.models import build_model, count_parameters
from src.report import summarize_runs
from src.train import fit, resolve_device, set_global_seed
from src.utils import (
    PROJECT_ROOT,
    current_timestamp,
    format_duration,
    make_batch_dir,
    make_run_dir,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="MNIST CNN/MLP 对照实验",
    )

    # 使用子命令组织训练、评估、预测、批量实验和汇总入口。
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="训练一组 CNN 或 MLP",
    )
    train_parser.add_argument(
        "--stage",
        choices=("basic", "augmented"),
        default="basic",
        help="实验阶段：basic 为原始数据，augmented 为随机仿射增强",
    )
    train_parser.add_argument(
        "--model",
        choices=("cnn", "mlp"),
        default="cnn",
        help="网络结构",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，正式实验使用 42、123、2026",
    )
    train_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="训练设备；auto 在 CUDA 自检失败时回退 CPU",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数，正式实验为 10",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="批量大小，正式实验为 128",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam 学习率，正式实验为 0.001",
    )
    train_parser.add_argument(
        "--output",
        default="artifacts",
        help="实验产物根目录，默认是项目根目录下的 artifacts/",
    )
    train_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="训练和测试每轮最多使用 N 个 batch，用于快速冒烟测试；默认跑完整数据",
    )
    train_parser.add_argument(
        "--download",
        action="store_true",
        help="数据缺失时自动下载 MNIST；默认不访问网络",
    )

    # ------------------------------------------------------------------
    # evaluate：对一个已有的运行目录生成测试指标和全部图表。
    # ------------------------------------------------------------------
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="生成测试曲线、卷积核、特征图和错分样本",
    )
    evaluate_parser.add_argument(
        "--run-dir",
        required=True,
        help="训练入口创建的运行目录，例如 artifacts/20261126-...",
    )
    evaluate_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="评估设备；auto 在 CUDA 自检失败时回退 CPU",
    )
    evaluate_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="测试时最多使用 N 个 batch，用于快速冒烟测试",
    )

    # ------------------------------------------------------------------
    # predict：加载单个权重文件，对一张图片做预测。
    # ------------------------------------------------------------------
    predict_parser = subparsers.add_parser(
        "predict",
        help="加载权重并对单张图片做预测",
    )
    predict_parser.add_argument(
        "--checkpoint",
        required=True,
        help="权重文件，例如 artifacts/<run>/weights/final.pt",
    )
    predict_parser.add_argument(
        "--image",
        required=True,
        help="待预测图片路径",
    )
    predict_parser.add_argument(
        "--invert",
        choices=("auto", "yes", "no"),
        default="auto",
        help="是否反色；auto 根据背景亮度自动判断",
    )
    predict_parser.add_argument(
        "--true-label",
        type=int,
        default=None,
        help="可选的真实标签，仅用于在结果图中标注",
    )
    predict_parser.add_argument(
        "--output",
        default=None,
        help="结果图路径；默认保存到运行目录下的 prediction.png",
    )
    predict_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="推理设备；auto 在 CUDA 自检失败时回退 CPU",
    )

    # ------------------------------------------------------------------
    # batch：一次跑完 2 阶段 × 2 模型 × 多种子。
    # ------------------------------------------------------------------
    batch_parser = subparsers.add_parser(
        "batch",
        help="批量训练并评估完整对照实验矩阵",
    )
    batch_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="整批实验使用的设备；auto 在 CUDA 自检失败时回退 CPU",
    )
    batch_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="每组训练轮数，正式实验为 10",
    )
    batch_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="初始批量大小；CUDA 显存不足时自动尝试 64",
    )
    batch_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam 学习率，正式实验为 0.001",
    )
    batch_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 2026],
        help="重复种子列表，正式实验为 42 123 2026",
    )
    batch_parser.add_argument(
        "--stages",
        nargs="+",
        choices=("basic", "augmented"),
        default=["basic", "augmented"],
        help="要运行的阶段列表",
    )
    batch_parser.add_argument(
        "--models",
        nargs="+",
        choices=("cnn", "mlp"),
        default=["cnn", "mlp"],
        help="要运行的模型列表",
    )
    batch_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="训练和测试每轮最多使用 N 个 batch，用于快速冒烟测试",
    )
    batch_parser.add_argument(
        "--output",
        default="artifacts",
        help="批量实验产物根目录",
    )
    batch_parser.add_argument(
        "--download",
        action="store_true",
        help="数据缺失时自动下载 MNIST",
    )
    batch_parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="只训练和汇总，不生成测试曲线/卷积核/特征图/错分样本",
    )

    # ------------------------------------------------------------------
    # report：对已有运行目录生成 CSV/Markdown/JSON 汇总。
    # ------------------------------------------------------------------
    report_parser = subparsers.add_parser(
        "report",
        help="汇总多个种子的实验结果",
    )
    report_parser.add_argument(
        "--run-root",
        required=True,
        help="包含一个或多个 history.json 的运行目录",
    )
    report_parser.add_argument(
        "--output",
        default=None,
        help="汇总文件输出目录；默认与 run-root 相同",
    )

    # ------------------------------------------------------------------
    # gui：图形界面，用于单图预测和实验结果浏览。
    # ------------------------------------------------------------------
    gui_parser = subparsers.add_parser(
        "gui",
        help="启动单图预测与实验对比图形界面",
    )
    gui_parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="实验产物根目录，默认是项目根目录下的 artifacts/",
    )
    gui_parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="GUI 推理设备；auto 在 CUDA 自检失败时回退 CPU",
    )

    return parser


def run_train(args: argparse.Namespace) -> None:
    """执行单组训练。

    Args:
        args: 由 ``argparse`` 解析出的训练参数。
    """
    # ------------------------------------------------------------------
    # 1. 选择设备。auto 模式会在 CUDA 自检失败时回退 CPU。
    # ------------------------------------------------------------------
    device, device_reason = resolve_device(args.device)
    print(f"device: {device} ({device_reason})", flush=True)

    # ------------------------------------------------------------------
    # 2. 在创建模型之前设置种子，保证权重初始化也可复现。
    # ------------------------------------------------------------------
    set_global_seed(args.seed)

    # ------------------------------------------------------------------
    # 3. 构建数据加载器。
    #    - 训练集传入 seed，保证不同模型的数据顺序一致。
    #    - 测试集不打乱，便于后续按索引定位错分样本。
    #    - num_workers 固定为 0，保证增强阶段的随机仿射可同步。
    # ------------------------------------------------------------------
    pin_memory = device.type == "cuda"

    train_loader = build_loader(
        stage=args.stage,
        train=True,
        batch_size=args.batch_size,
        shuffle_seed=args.seed,
        num_workers=0,
        pin_memory=pin_memory,
        download=args.download,
    )
    test_loader = build_loader(
        stage=args.stage,
        train=False,
        batch_size=args.batch_size,
        shuffle_seed=None,
        num_workers=0,
        pin_memory=pin_memory,
        download=args.download,
    )

    # ------------------------------------------------------------------
    # 4. 创建模型并移动到设备。
    # ------------------------------------------------------------------
    model = build_model(args.model).to(device)
    parameter_count = count_parameters(model)

    # ------------------------------------------------------------------
    # 5. 为本次运行创建独立目录，并整理配置。
    # ------------------------------------------------------------------
    run_dir = make_run_dir(
        args.output,
        stage=args.stage,
        model=args.model,
        seed=args.seed,
    )

    config = {
        "stage": args.stage,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "device_reason": device_reason,
        "num_workers": 0,
        "pin_memory": pin_memory,
        "max_batches": args.max_batches,
        "parameter_count": parameter_count,
        "data_root": str(DEFAULT_DATA_ROOT),
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
    }

    print(
        f"run: stage={args.stage} model={args.model} seed={args.seed} "
        f"params={parameter_count:,}",
        flush=True,
    )
    print(f"output: {run_dir}", flush=True)

    # ------------------------------------------------------------------
    # 6. 训练、保存权重和测试末轮模型。
    # ------------------------------------------------------------------
    result = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=device,
        output_dir=run_dir,
        config=config,
        max_batches=args.max_batches,
    )

    # ------------------------------------------------------------------
    # 7. 打印最终摘要，完整记录已写入 history.json。
    # ------------------------------------------------------------------
    test_metrics = result["test"]
    resources = result["resources"]

    print("-" * 60)
    print(f"total train time: {format_duration(result['total_training_seconds'])}")
    print(
        f"cpu time: {resources['cpu_time_seconds']:.2f}s, "
        f"peak RSS: {resources['peak_rss_mib']:.1f} MiB"
    )
    if resources["gpu_peak_allocated_mib"] is not None:
        print(
            f"gpu peak allocated: {resources['gpu_peak_allocated_mib']:.1f} MiB, "
            f"reserved: {resources['gpu_peak_reserved_mib']:.1f} MiB"
        )
    if test_metrics is not None:
        print(
            f"final test: loss={test_metrics['loss']:.4f} "
            f"acc={test_metrics['accuracy']:.4f}"
        )
    print(f"artifacts: {run_dir}")


def run_evaluate(args: argparse.Namespace) -> None:
    """对一个训练运行目录生成测试指标和可视化图表。"""
    device, device_reason = resolve_device(args.device)
    print(f"device: {device} ({device_reason})", flush=True)

    summary = evaluate_run(
        run_dir=args.run_dir,
        device=device,
        max_batches=args.max_batches,
    )

    print("-" * 60)
    print(f"figures written to: {Path(args.run_dir).resolve()}")
    for figure_name in summary["figures"]:
        print(f"  - {figure_name}")
    print("  - evaluation.json")
    print("  - test_metrics.json")


def run_predict(args: argparse.Namespace) -> None:
    """加载权重并对单张图片做预测。"""
    device, device_reason = resolve_device(args.device)
    print(f"device: {device} ({device_reason})", flush=True)

    # 从检查点恢复模型结构；config 中保存了 model 名称。
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    model_name = checkpoint.get("config", {}).get("model", "unknown")

    # 图片预处理：灰度、按需反色、缩放到 28x28、标准化。
    image = preprocess_single_image(args.image, invert=args.invert)

    predicted, probabilities = predict_single_image(model, image, device)

    print(f"model: {model_name}")
    print(f"prediction: {predicted}")
    print("probabilities:")
    for digit, probability in enumerate(probabilities.tolist()):
        print(f"  {digit}: {probability:.4f}")

    # 默认把结果图放到运行目录下；也可以用 --output 指定。
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.checkpoint).resolve().parent.parent / "prediction.png"

    plot_prediction(
        image=image,
        probabilities=probabilities,
        predicted=predicted,
        output_path=output_path,
        true_label=args.true_label,
    )
    print(f"figure: {output_path}")


class CudaOutOfMemoryError(RuntimeError):
    """批量训练的子进程报告了 CUDA 显存不足。"""


def _is_cuda_oom(output: str) -> bool:
    """判断子进程输出中是否包含 CUDA OOM 信息。"""
    text = output.lower()
    markers = (
        "cuda out of memory",
        "outofmemoryerror",
        "cuda error: out of memory",
    )
    return any(marker in text for marker in markers)


def _run_command_streaming(command: list[str]) -> tuple[int, str]:
    """运行子进程并实时转发输出，同时保留完整文本用于错误判断。

    Args:
        command: 命令参数列表，不经过 shell。

    Returns:
        ``(退出码, 合并后的 stdout/stderr)``。
    """
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    collected: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        collected.append(line)

    return process.wait(), "".join(collected)


def _run_matrix(
    *,
    device_name: str,
    batch_size: int,
    epochs: int,
    lr: float,
    stages: list[str],
    models: list[str],
    seeds: list[int],
    run_root: Path,
    max_batches: int | None,
    download: bool,
) -> list[dict[str, object]]:
    """依次在独立子进程中运行完整实验矩阵。

    每组训练都通过 ``python main.py train ...`` 启动，符合实验文档中
    “每组依次在独立进程运行”的要求。

    Raises:
        CudaOutOfMemoryError: CUDA 子进程输出中检测到显存不足。
        RuntimeError: 其他训练失败。
    """
    main_path = Path(__file__).resolve()
    total = len(stages) * len(models) * len(seeds)
    completed: list[dict[str, object]] = []
    index = 0

    for stage in stages:
        for model in models:
            for seed in seeds:
                index += 1
                print(
                    f"\n=== [{index}/{total}] "
                    f"stage={stage} model={model} seed={seed} "
                    f"device={device_name} batch={batch_size} ===",
                    flush=True,
                )

                command = [
                    sys.executable,
                    str(main_path),
                    "train",
                    "--stage",
                    stage,
                    "--model",
                    model,
                    "--seed",
                    str(seed),
                    "--device",
                    device_name,
                    "--epochs",
                    str(epochs),
                    "--batch-size",
                    str(batch_size),
                    "--lr",
                    str(lr),
                    "--output",
                    str(run_root),
                ]
                if max_batches is not None:
                    command.extend(["--max-batches", str(max_batches)])
                if download:
                    command.append("--download")

                returncode, output = _run_command_streaming(command)

                if returncode != 0:
                    if device_name == "cuda" and _is_cuda_oom(output):
                        raise CudaOutOfMemoryError(
                            f"CUDA OOM: stage={stage} model={model} seed={seed} "
                            f"batch={batch_size}\n{output[-4000:]}"
                        )
                    raise RuntimeError(
                        f"训练失败：stage={stage} model={model} seed={seed} "
                        f"返回码={returncode}\n{output[-4000:]}"
                    )

                completed.append(
                    {
                        "stage": stage,
                        "model": model,
                        "seed": seed,
                        "returncode": returncode,
                    }
                )

    return completed


def _evaluate_runs(
    run_root: Path,
    *,
    device_name: str,
    max_batches: int | None,
) -> list[Path]:
    """为批量实验中的每个运行目录生成评估图表。

    Args:
        run_root: 本次尝试的根目录，内部每个子目录都应包含 history.json。
        device_name: ``"cuda"`` 或 ``"cpu"``。
        max_batches: 冒烟测试时限制测试 batch 数。

    Returns:
        已评估的运行目录列表。

    Raises:
        FileNotFoundError: 没有找到任何训练历史。
        RuntimeError: 某个运行目录评估失败。
    """
    main_path = Path(__file__).resolve()
    run_dirs = [
        history_path.parent
        for history_path in sorted(run_root.rglob("history.json"))
    ]
    if not run_dirs:
        raise FileNotFoundError(f"{run_root} 下没有找到 history.json")

    total = len(run_dirs)
    for index, run_dir in enumerate(run_dirs, start=1):
        print(
            f"\n=== evaluate [{index}/{total}] {run_dir.name} ===",
            flush=True,
        )
        command = [
            sys.executable,
            str(main_path),
            "evaluate",
            "--run-dir",
            str(run_dir),
            "--device",
            device_name,
        ]
        if max_batches is not None:
            command.extend(["--max-batches", str(max_batches)])

        returncode, output = _run_command_streaming(command)
        if returncode != 0:
            raise RuntimeError(
                f"评估失败：{run_dir} 返回码={returncode}\n{output[-4000:]}"
            )

    return run_dirs


def run_batch(args: argparse.Namespace) -> None:
    """运行完整批量实验，自动评估每个运行并生成汇总。"""
    # 去重，避免用户重复传入同一个配置。
    stages = list(dict.fromkeys(args.stages))
    models = list(dict.fromkeys(args.models))
    seeds = list(dict.fromkeys(args.seeds))

    if not stages or not models or not seeds:
        raise ValueError("stages、models、seeds 不能为空")

    # 设备在整批实验开始时只选择一次；子进程显式使用 cuda 或 cpu，
    # 避免每组重复自检，也保证整批实验使用同一设备。
    device, device_reason = resolve_device(args.device)
    final_device_name = device.type
    print(f"device: {device} ({device_reason})", flush=True)

    batch_dir = make_batch_dir(args.output)
    print(f"batch directory: {batch_dir}", flush=True)

    batch_info: dict[str, object] = {
        "created_at": current_timestamp(),
        "device": str(device),
        "device_reason": device_reason,
        "stages": stages,
        "models": models,
        "seeds": seeds,
        "epochs": args.epochs,
        "initial_batch_size": args.batch_size,
        "lr": args.lr,
        "max_batches": args.max_batches,
        "attempts": [],
    }
    save_json(batch_dir / "batch_config.json", batch_info)

    attempts: list[dict[str, object]] = []
    attempt_index = 0

    def try_attempt(
        attempt_device_name: str,
        batch_size: int,
        label: str,
    ) -> Path | None:
        """运行一次完整矩阵；CUDA OOM 时返回 None，其他错误继续抛出。"""
        nonlocal attempt_index

        attempt_index += 1
        run_root = batch_dir / f"{label}_{attempt_index}"
        run_root.mkdir(parents=True, exist_ok=False)
        print(f"\nattempt: {run_root.name}", flush=True)

        try:
            completed = _run_matrix(
                device_name=attempt_device_name,
                batch_size=batch_size,
                epochs=args.epochs,
                lr=args.lr,
                stages=stages,
                models=models,
                seeds=seeds,
                run_root=run_root,
                max_batches=args.max_batches,
                download=args.download,
            )
        except CudaOutOfMemoryError as error:
            attempts.append(
                {
                    "device": attempt_device_name,
                    "batch_size": batch_size,
                    "run_root": str(run_root),
                    "status": "cuda_oom",
                    "output_tail": str(error)[-4000:],
                }
            )
            return None
        except Exception as error:
            attempts.append(
                {
                    "device": attempt_device_name,
                    "batch_size": batch_size,
                    "run_root": str(run_root),
                    "status": "failed",
                    "error": str(error)[-4000:],
                }
            )
            raise
        else:
            attempts.append(
                {
                    "device": attempt_device_name,
                    "batch_size": batch_size,
                    "run_root": str(run_root),
                    "status": "completed",
                    "runs": len(completed),
                }
            )
            return run_root
        finally:
            batch_info["attempts"] = attempts
            save_json(batch_dir / "batch_config.json", batch_info)

    final_run_root: Path | None = None

    if device.type == "cuda":
        # 先按用户给定 batch size；若大于 64，再按文档尝试 64。
        batch_sizes = [args.batch_size]
        if args.batch_size > 64:
            batch_sizes.append(64)

        for batch_size in batch_sizes:
            final_run_root = try_attempt(
                "cuda",
                batch_size,
                f"cuda_b{batch_size}",
            )
            if final_run_root is not None:
                break

        # CUDA 两次都 OOM：auto 模式整批转 CPU；显式 cuda 则报错。
        if final_run_root is None:
            if args.device == "auto":
                print(
                    "CUDA attempts failed; retrying the entire batch on CPU.",
                    flush=True,
                )
                cpu_device, cpu_reason = resolve_device("cpu")
                final_device_name = "cpu"
                batch_info["device"] = str(cpu_device)
                batch_info["device_reason"] = f"{device_reason}; {cpu_reason}"
                final_run_root = try_attempt(
                    "cpu",
                    args.batch_size,
                    f"cpu_b{args.batch_size}",
                )
            else:
                raise RuntimeError(
                    "CUDA 显存不足，且 --device cuda 不允许自动回退；"
                    "请改用 --device auto 或 --device cpu。"
                )
    else:
        final_run_root = try_attempt(
            "cpu",
            args.batch_size,
            f"cpu_b{args.batch_size}",
        )

    if final_run_root is None:
        raise RuntimeError("批量实验没有完成任何一次尝试")

    batch_info["final_run_root"] = str(final_run_root)
    batch_info["final_device"] = final_device_name
    save_json(batch_dir / "batch_config.json", batch_info)

    # 为每个运行生成测试曲线、卷积核、特征图和错分样本。
    if not args.skip_evaluate:
        batch_info["status"] = "evaluating"
        save_json(batch_dir / "batch_config.json", batch_info)
        try:
            evaluated_runs = _evaluate_runs(
                final_run_root,
                device_name=final_device_name,
                max_batches=args.max_batches,
            )
        except Exception:
            batch_info["status"] = "evaluation_failed"
            save_json(batch_dir / "batch_config.json", batch_info)
            raise
        batch_info["evaluated_runs"] = len(evaluated_runs)
        save_json(batch_dir / "batch_config.json", batch_info)

    # 汇总最终成功的那次尝试；失败尝试保留在 batch_config.json 中。
    try:
        summary = summarize_runs(final_run_root, batch_dir)
    except Exception:
        batch_info["status"] = "summary_failed"
        save_json(batch_dir / "batch_config.json", batch_info)
        raise

    batch_info["status"] = "completed"
    batch_info["run_count"] = summary["run_count"]
    batch_info["summary"] = {
        "summary_csv": str(batch_dir / "summary.csv"),
        "summary_md": str(batch_dir / "summary.md"),
        "summary_json": str(batch_dir / "summary.json"),
    }
    save_json(batch_dir / "batch_config.json", batch_info)

    print("\n" + "=" * 60)
    print(f"batch complete: {summary['run_count']} runs")
    print(f"summary: {batch_dir / 'summary.md'}")
    print(f"details: {batch_dir / 'summary.csv'}")


def run_gui(args: argparse.Namespace) -> None:
    """启动单图预测与实验对比的 Tkinter 图形界面。"""
    device, device_reason = resolve_device(args.device)
    print(f"device: {device} ({device_reason})", flush=True)

    try:
        from src.gui import launch_gui
    except ImportError as error:
        raise RuntimeError(
            "无法启动图形界面，请确认当前 Python 已安装 Tkinter"
            "（Linux 上通常由 python3-tk 提供）。"
        ) from error

    launch_gui(artifacts_root=args.artifacts, device=device)


def run_report(args: argparse.Namespace) -> None:
    """对已有运行目录生成汇总。"""
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root

    output_dir: Path | None = None
    if args.output is not None:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    summary = summarize_runs(run_root, output_dir)

    final_output = output_dir or run_root
    print(f"runs: {summary['run_count']}")
    print(f"summary: {final_output / 'summary.md'}")
    print(f"details: {final_output / 'summary.csv'}")


def main() -> None:
    """解析命令行并分发到对应子命令。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "batch":
        run_batch(args)
    elif args.command == "report":
        run_report(args)
    elif args.command == "gui":
        run_gui(args)
    else:
        # argparse 已经限制了 choices，这里只做兜底。
        parser.error(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
