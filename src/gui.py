"""MNIST 单图预测与实验对比的 Tkinter 图形界面。

界面分为两个页签：

    单图预测
        选择图片和权重，调用与命令行 ``predict`` 相同的预处理和推理函数，
        显示预测数字、置信度、Top-3 和 10 类概率条。

    实验对比
        读取 ``artifacts/*/summary.json``，展示阶段/模型汇总、多种子差值
        和逐次运行明细，并预览 curves/kernels/feature_maps/misclassified。

设计取舍：

    - 只使用标准库 Tkinter 和项目已有依赖，不额外引入 GUI 框架。
    - 图表预览直接显示训练时生成的 PNG，不嵌入 Matplotlib，避免
      ``src.evaluate`` 强制使用 Agg 后端后与 TkAgg 冲突。
    - 权重和图片处理复用 ``src.evaluate`` 的现有函数，保证 GUI 与命令行
      的单图预测完全一致。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageTk

from src.evaluate import (
    load_model_from_checkpoint,
    predict_single_image,
    preprocess_single_image,
)
from src.report import build_summary
from src.utils import DEFAULT_ARTIFACT_ROOT, PROJECT_ROOT, load_json

# ---------------------------------------------------------------------------
# 颜色和字体
# ---------------------------------------------------------------------------
BACKGROUND = "#f3f5f7"
SURFACE = "#ffffff"
TEXT = "#202124"
MUTED = "#5f6368"
BORDER = "#dadce0"
ACCENT = "#1a73e8"
DANGER = "#d93025"

# 图片处理下拉框：中文显示 -> preprocess_single_image 的参数。
INVERT_OPTIONS = {
    "自动判断": "auto",
    "强制反色": "yes",
    "不反色": "no",
}


def _choose_font(root: tk.Tk) -> str:
    """选择支持中文的字体，找不到时退回 Tk 默认字体。"""
    families = set(tkfont.families(root))
    candidates = (
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    )
    for name in candidates:
        if name in families:
            return name
    return tkfont.nametofont("TkDefaultFont").actual("family")


def _find_summary_files(root: str | Path) -> list[Path]:
    """查找批量实验的 ``summary.json``，最近的批次排在最前。"""
    root_path = Path(root)
    if not root_path.exists():
        return []

    candidates: list[Path] = []
    direct = root_path / "summary.json"
    if direct.exists():
        candidates.append(direct)
    candidates.extend(root_path.glob("*/summary.json"))

    # resolve + set 去重，避免用户把 --artifacts 指到某个批次目录时重复。
    unique = {path.resolve() for path in candidates if path.exists()}
    return sorted(unique, key=lambda path: path.stat().st_mtime, reverse=True)


def _discover_runs(root: str | Path) -> list[dict[str, Any]]:
    """从 ``history.json`` 读取运行记录，作为没有 summary.json 时的兜底。"""
    root_path = Path(root)
    if not root_path.exists():
        return []

    records: list[dict[str, Any]] = []
    for history_path in sorted(root_path.rglob("history.json")):
        try:
            data = load_json(history_path)
        except Exception:
            continue

        config = data.get("config", {})
        test_metrics = data.get("test") or {}
        resources = data.get("resources") or {}
        history = data.get("history") or []
        last_epoch = history[-1] if history else {}

        records.append(
            {
                "run_dir": str(history_path.parent.resolve()),
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

    stage_order = {"basic": 0, "augmented": 1}
    model_order = {"cnn": 0, "mlp": 1}
    records.sort(
        key=lambda run: (
            stage_order.get(run.get("stage"), 99),
            model_order.get(run.get("model"), 99),
            run.get("seed") if run.get("seed") is not None else 10**9,
        )
    )
    return records


def _discover_examples(
    batch_root: Path | None,
    artifacts_root: Path | None = None,
) -> list[Path]:
    """查找可用于“使用示例”循环的图片。

    优先级：

    1. 当前批次 ``prediction_demo/samples/`` 中的图片；
    2. 仓库根目录 ``examples/digits/`` 中的 0-9 示例；
    3. 当前批次或 artifacts 中最早找到的 ``input_digit.png``。

    这样可以兼容旧批次，也保证没有实验产物时仍能展示内置示例。
    """
    candidate_dirs: list[Path] = []
    if batch_root is not None:
        candidate_dirs.append(batch_root / "prediction_demo" / "samples")
    candidate_dirs.append(PROJECT_ROOT / "examples" / "digits")

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        paths = sorted(directory.glob("digit_*.png"))
        if not paths:
            paths = sorted(
                path
                for path in directory.glob("*.png")
                if path.name != "prediction.png"
            )
        if paths:
            return paths

    if batch_root is not None:
        legacy = batch_root / "prediction_demo" / "input_digit.png"
        if legacy.exists():
            return [legacy]

    if artifacts_root is not None:
        legacy_paths = sorted(
            artifacts_root.rglob("prediction_demo/input_digit.png"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if legacy_paths:
            return [legacy_paths[0]]

    return []


def _discover_example_sets(
    batch_root: Path | None,
    artifacts_root: Path | None = None,
) -> dict[str, list[Path]]:
    """查找清晰示例和模糊示例两组图片。

    返回的字典保持“清晰示例”在前、“模糊示例”在后；没有某一组时直接省略。
    模糊示例优先使用当前批次的 ``prediction_demo/ambiguous``，否则读取
    仓库内置的 ``examples/ambiguous``。
    """
    sets: dict[str, list[Path]] = {}

    clear_examples = _discover_examples(batch_root, artifacts_root)
    if clear_examples:
        sets["清晰示例"] = clear_examples

    ambiguous_dirs: list[Path] = []
    if batch_root is not None:
        ambiguous_dirs.append(batch_root / "prediction_demo" / "ambiguous")
    ambiguous_dirs.append(PROJECT_ROOT / "examples" / "ambiguous")

    for directory in ambiguous_dirs:
        if not directory.exists():
            continue
        paths = sorted(directory.glob("digit_*_amb_*.png"))
        if not paths:
            paths = sorted(
                path
                for path in directory.glob("*.png")
                if path.name != "prediction.png"
            )
        if paths:
            sets["模糊示例"] = paths
            break

    return sets


def _format_pair(
    mean: float | None,
    std: float | None,
    digits: int = 2,
    scale: float = 1.0,
    suffix: str = "",
) -> str:
    """格式化 ``mean ± std``；缺失时显示 N/A。"""
    if mean is None:
        return "N/A"
    std_value = 0.0 if std is None else std
    return (
        f"{mean * scale:.{digits}f}{suffix}±"
        f"{std_value * scale:.{digits}f}{suffix}"
    )


def _format_signed(mean: float | None, std: float | None) -> str:
    """格式化带正负号的差值，单位由调用方写在标题里。"""
    if mean is None:
        return "N/A"
    std_value = 0.0 if std is None else std
    return f"{mean:+.2f} ± {std_value:.2f}"


class DigitRecognitionGUI:
    """Tkinter 主窗口。"""

    def __init__(
        self,
        root: tk.Tk,
        artifacts_root: str | Path = DEFAULT_ARTIFACT_ROOT,
        device: torch.device | None = None,
    ) -> None:
        self.root = root
        self.artifacts_root = Path(artifacts_root)
        if not self.artifacts_root.is_absolute():
            self.artifacts_root = PROJECT_ROOT / self.artifacts_root
        self.device = device or torch.device("cpu")
        self.font_family = _choose_font(root)

        # 运行与汇总状态。
        self.summary_files: list[Path] = []
        self.summary_map: dict[str, Path] = {}
        self.summary_data: dict[str, Any] | None = None
        self.summary_path: Path | None = None
        self.batch_root: Path | None = None

        # 运行下拉框：显示文本 -> 记录 / 权重路径 / 表格 iid。
        self.run_map: dict[str, dict[str, Any]] = {}
        self.checkpoint_map: dict[str, Path] = {}
        self.run_iid_map: dict[str, str] = {}

        # 单图预测状态。
        self.current_image_path: Path | None = None
        self.example_sets: dict[str, list[Path]] = {}
        self.example_paths: list[Path] = []
        self.example_index = -1
        self.current_probabilities: list[float] | None = None
        self.current_predicted: int | None = None
        self.model_cache: dict[tuple[str, str], tuple[torch.nn.Module, dict[str, Any]]] = {}
        self.input_photo: ImageTk.PhotoImage | None = None
        self.figure_photo: ImageTk.PhotoImage | None = None
        self.current_figure_kind = "curves"

        self._configure_window()
        self._configure_style()
        self._build_header()
        self._build_notebook()
        self._refresh_batches(initial=True)

    # ------------------------------------------------------------------
    # 窗口与样式
    # ------------------------------------------------------------------
    def _configure_window(self) -> None:
        """设置窗口标题、大小并在屏幕上居中。"""
        self.root.title("MNIST 手写数字识别 · CNN/MLP 对照实验")
        self.root.configure(bg=BACKGROUND)
        self.root.minsize(1080, 700)

        width, height = 1280, 960
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _configure_style(self) -> None:
        """统一 ttk 字体和颜色。"""
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("TFrame", background=BACKGROUND)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT)
        style.configure(
            "Title.TLabel",
            font=(self.font_family, 19, "bold"),
            foreground=TEXT,
        )
        style.configure(
            "Subtitle.TLabel",
            font=(self.font_family, 9),
            foreground=MUTED,
        )
        style.configure(
            "Section.TLabel",
            font=(self.font_family, 11, "bold"),
            foreground=TEXT,
        )
        style.configure(
            "Muted.TLabel",
            font=(self.font_family, 9),
            foreground=MUTED,
        )
        style.configure(
            "TLabelframe",
            background=BACKGROUND,
            bordercolor=BORDER,
            relief="solid",
        )
        style.configure(
            "TLabelframe.Label",
            background=BACKGROUND,
            foreground=TEXT,
            font=(self.font_family, 11, "bold"),
        )
        style.configure("TButton", font=(self.font_family, 10), padding=(8, 4))
        style.configure(
            "Accent.TButton",
            font=(self.font_family, 11, "bold"),
            padding=(10, 7),
        )
        style.configure(
            "Treeview",
            font=(self.font_family, 9),
            rowheight=24,
            background=SURFACE,
            fieldbackground=SURFACE,
        )
        style.configure(
            "Treeview.Heading",
            font=(self.font_family, 9, "bold"),
        )

    def _build_header(self) -> None:
        """标题栏：名称、副标题和设备信息。"""
        header = ttk.Frame(self.root, padding=(18, 10, 18, 4))
        header.pack(fill="x")

        title_box = ttk.Frame(header)
        title_box.pack(side="left")
        ttk.Label(
            title_box,
            text="MNIST 手写数字识别",
            style="Title.TLabel",
        ).pack(side="left")
        ttk.Label(
            title_box,
            text="CNN / MLP 对照实验 · 单图预测与结果浏览",
            style="Subtitle.TLabel",
        ).pack(side="left", padx=(14, 0), pady=(8, 0))

        ttk.Label(
            header,
            text=f"设备: {self.device.type.upper()}",
            style="Muted.TLabel",
        ).pack(side="right", anchor="n")

    def _build_notebook(self) -> None:
        """创建两个主功能页签。"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=16, pady=(2, 4))

        self.prediction_tab = ttk.Frame(self.notebook, padding=14)
        self.comparison_tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(self.prediction_tab, text="  单图预测  ")
        self.notebook.add(self.comparison_tab, text="  实验对比  ")

        self._build_prediction_tab()
        self._build_comparison_tab()

        footer = ttk.Label(
            self.root,
            text=f"权重与图表目录: {self.artifacts_root}",
            style="Muted.TLabel",
            padding=(18, 0, 18, 6),
        )
        footer.pack(fill="x")

    # ------------------------------------------------------------------
    # 页签一：单图预测
    # ------------------------------------------------------------------
    def _build_prediction_tab(self) -> None:
        tab = self.prediction_tab
        tab.columnconfigure(0, weight=0)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(tab, text="输入与模型", padding=14)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left.columnconfigure(0, weight=1)

        right = ttk.LabelFrame(tab, text="预测结果", padding=14)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        # 输入图片预览。
        self.input_canvas = tk.Canvas(
            left,
            width=260,
            height=260,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.input_canvas.grid(row=0, column=0, sticky="ew")
        self.input_canvas.create_text(
            130,
            130,
            text="尚未选择图片",
            fill=MUTED,
            font=(self.font_family, 11),
            tags="placeholder",
        )
        self.input_canvas.bind(
            "<Configure>",
            lambda _event: self._redraw_input_preview(),
        )

        button_row = ttk.Frame(left)
        button_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        ttk.Button(
            button_row,
            text="选择图片...",
            command=self._choose_image,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.example_button = ttk.Button(
            button_row,
            text="使用示例",
            command=self._load_example,
        )
        self.example_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.example_button.configure(state="disabled")

        # 示例集：清晰示例与模糊示例共用一个循环按钮。
        example_set_row = ttk.Frame(left)
        example_set_row.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(example_set_row, text="示例集").pack(side="left")
        self.example_set_var = tk.StringVar(value="清晰示例")
        self.example_set_combo = ttk.Combobox(
            example_set_row,
            textvariable=self.example_set_var,
            state="readonly",
            width=18,
        )
        self.example_set_combo.pack(side="right")
        self.example_set_combo.bind(
            "<<ComboboxSelected>>",
            self._on_example_set_change,
        )

        ttk.Label(left, text="模型权重", style="Section.TLabel").grid(
            row=3,
            column=0,
            sticky="w",
            pady=(8, 3),
        )
        self.checkpoint_combo = ttk.Combobox(left, state="readonly", width=30)
        self.checkpoint_combo.grid(row=4, column=0, sticky="ew")

        option_row = ttk.Frame(left)
        option_row.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(option_row, text="图片处理").pack(side="left")
        self.invert_var = tk.StringVar(value="自动判断")
        self.invert_combo = ttk.Combobox(
            option_row,
            textvariable=self.invert_var,
            values=list(INVERT_OPTIONS),
            state="readonly",
            width=12,
        )
        self.invert_combo.pack(side="right")

        ttk.Button(
            left,
            text="开始预测",
            style="Accent.TButton",
            command=self._predict,
        ).grid(row=6, column=0, sticky="ew", pady=(10, 8))

        self.status_var = tk.StringVar(value="请选择图片和模型权重。")
        ttk.Label(
            left,
            textvariable=self.status_var,
            style="Muted.TLabel",
            wraplength=360,
            justify="left",
        ).grid(row=7, column=0, sticky="w")

        # 右侧：预测数字和概率图。
        result_top = ttk.Frame(right)
        result_top.grid(row=0, column=0, sticky="ew")
        self.digit_label = tk.Label(
            result_top,
            text="?",
            font=(self.font_family, 60, "bold"),
            fg=ACCENT,
            bg=BACKGROUND,
            width=2,
        )
        self.digit_label.pack(side="left", padx=(4, 18))

        info_box = ttk.Frame(result_top)
        info_box.pack(side="left", fill="x", expand=True)
        self.confidence_var = tk.StringVar(value="等待预测")
        self.top3_var = tk.StringVar(value="选择图片后点击“开始预测”")
        ttk.Label(
            info_box,
            textvariable=self.confidence_var,
            style="Section.TLabel",
        ).pack(anchor="w", pady=(10, 4))
        ttk.Label(
            info_box,
            textvariable=self.top3_var,
            style="Muted.TLabel",
            wraplength=420,
            justify="left",
        ).pack(anchor="w")

        ttk.Label(right, text="各类别概率", style="Section.TLabel").grid(
            row=1,
            column=0,
            sticky="w",
            pady=(14, 4),
        )
        self.prob_canvas = tk.Canvas(
            right,
            height=200,
            bg=SURFACE,
            highlightthickness=0,
        )
        self.prob_canvas.grid(row=2, column=0, sticky="nsew")
        self.prob_canvas.bind(
            "<Configure>",
            lambda _event: self._draw_probability_canvas(),
        )

    # ------------------------------------------------------------------
    # 页签二：实验对比
    # ------------------------------------------------------------------
    def _build_comparison_tab(self) -> None:
        tab = self.comparison_tab
        tab.rowconfigure(2, weight=1)
        tab.columnconfigure(0, weight=1)

        top = ttk.Frame(tab)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="实验批次", style="Section.TLabel").pack(side="left")
        self.batch_combo = ttk.Combobox(top, state="readonly", width=32)
        self.batch_combo.pack(side="left", padx=(8, 8))
        self.batch_combo.bind("<<ComboboxSelected>>", self._on_batch_select)
        ttk.Button(top, text="刷新", command=self._refresh_batches).pack(side="left")
        ttk.Button(
            top,
            text="打开汇总",
            command=self._open_summary,
        ).pack(side="left", padx=(8, 0))

        self.summary_meta_var = tk.StringVar(value="")
        ttk.Label(
            tab,
            textvariable=self.summary_meta_var,
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 10))

        paned = ttk.Panedwindow(tab, orient="horizontal")
        paned.grid(row=2, column=0, sticky="nsew")

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        # ---- 左侧：汇总、差值和逐次运行 ----
        group_frame = ttk.LabelFrame(left, text="阶段 / 模型汇总", padding=10)
        group_frame.pack(fill="x")
        group_columns = ("stage", "model", "runs", "acc", "loss", "time")
        group_headings = ("阶段", "模型", "次数", "准确率", "损失", "耗时 (s)")
        group_widths = (55, 52, 40, 170, 170, 130)
        self.group_tree = self._create_tree(
            group_frame,
            columns=group_columns,
            headings=group_headings,
            widths=group_widths,
            height=4,
        )

        delta_frame = ttk.LabelFrame(left, text="关键差值", padding=(10, 6))
        delta_frame.pack(fill="x", pady=(10, 0))
        self.delta_var = tk.StringVar(value="")
        ttk.Label(
            delta_frame,
            textvariable=self.delta_var,
            style="Muted.TLabel",
            justify="left",
            wraplength=700,
        ).pack(anchor="w")
        # 差值保持三行短句，避免在左侧面板里换行过多、挤占运行明细表。

        run_frame = ttk.LabelFrame(left, text="逐次运行明细", padding=10)
        run_frame.pack(fill="both", expand=True, pady=(10, 0))
        run_columns = (
            "stage",
            "model",
            "seed",
            "acc",
            "loss",
            "time",
            "cpu",
            "rss",
            "params",
        )
        run_headings = (
            "阶段",
            "模型",
            "种子",
            "准确率",
            "损失",
            "训练(s)",
            "CPU(s)",
            "RSS",
            "参数量",
        )
        run_widths = (55, 52, 40, 78, 78, 76, 76, 60, 84)
        self.run_tree = self._create_tree(
            run_frame,
            columns=run_columns,
            headings=run_headings,
            widths=run_widths,
            height=5,
            with_scrollbar=True,
        )
        self.run_tree.bind("<<TreeviewSelect>>", self._on_run_tree_select)

        # ---- 右侧：运行图表 ----
        figure_frame = ttk.LabelFrame(right, text="运行图表", padding=10)
        figure_frame.pack(fill="both", expand=True)
        figure_frame.columnconfigure(0, weight=1)
        figure_frame.rowconfigure(4, weight=1)

        ttk.Label(figure_frame, text="选择运行", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        self.figure_run_var = tk.StringVar(value="")
        self.figure_run_combo = ttk.Combobox(
            figure_frame,
            textvariable=self.figure_run_var,
            state="readonly",
            width=34,
        )
        self.figure_run_combo.grid(row=1, column=0, sticky="ew", pady=(3, 8))
        self.figure_run_combo.bind(
            "<<ComboboxSelected>>",
            self._on_figure_run_change,
        )

        figure_buttons = ttk.Frame(figure_frame)
        figure_buttons.grid(row=2, column=0, sticky="ew")
        figure_buttons.columnconfigure(0, weight=1)
        figure_buttons.columnconfigure(1, weight=1)
        for index, (kind, text) in enumerate(
            (
                ("curves", "曲线"),
                ("kernels", "卷积核"),
                ("feature_maps", "特征图"),
                ("misclassified", "错分样本"),
            )
        ):
            row = index // 2
            column = index % 2
            ttk.Button(
                figure_buttons,
                text=text,
                command=lambda selected_kind=kind: self._show_figure(selected_kind),
            ).grid(
                row=row,
                column=column,
                sticky="ew",
                padx=(0 if column == 0 else 4, 0),
                pady=(0 if row == 0 else 4, 0),
            )

        self.figure_canvas = tk.Canvas(
            figure_frame,
            width=420,
            height=300,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.figure_canvas.grid(row=4, column=0, sticky="nsew", pady=(10, 4))
        self.figure_canvas.create_text(
            210,
            150,
            text="选择运行后查看图表",
            fill=MUTED,
            font=(self.font_family, 11),
            tags="placeholder",
        )
        self.figure_canvas.bind(
            "<Configure>",
            lambda _event: self._refresh_figure(notify=False),
        )

        self.figure_caption_var = tk.StringVar(value="")
        ttk.Label(
            figure_frame,
            textvariable=self.figure_caption_var,
            style="Muted.TLabel",
            wraplength=420,
        ).grid(row=5, column=0, sticky="w")

    def _create_tree(
        self,
        parent: ttk.Frame,
        *,
        columns: tuple[str, ...],
        headings: tuple[str, ...],
        widths: tuple[int, ...],
        height: int,
        with_scrollbar: bool = False,
    ) -> ttk.Treeview:
        """创建只读表格，可选垂直滚动条。"""
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        tree = ttk.Treeview(
            container,
            columns=columns,
            show="headings",
            height=height,
            selectmode="browse",
        )
        tree.pack(side="left", fill="both", expand=True)

        for column, heading, width in zip(columns, headings, widths):
            tree.heading(column, text=heading)
            tree.column(column, width=width, minwidth=width, anchor="center")

        tree.tag_configure("odd", background="#f7f9fb")
        tree.tag_configure("even", background=SURFACE)

        if with_scrollbar:
            scrollbar = ttk.Scrollbar(
                container,
                orient="vertical",
                command=tree.yview,
            )
            scrollbar.pack(side="right", fill="y")
            tree.configure(yscrollcommand=scrollbar.set)

        return tree

    # ------------------------------------------------------------------
    # 汇总加载与表格刷新
    # ------------------------------------------------------------------
    def _display_batch_name(self, path: Path) -> str:
        """把批次目录显示成较短但可区分的名称。"""
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    def _refresh_batches(self, initial: bool = False) -> None:
        """重新查找 summary.json 并加载最新批次。"""
        current = self.batch_combo.get()
        self.summary_files = _find_summary_files(self.artifacts_root)
        self.summary_map = {}

        values: list[str] = []
        for path in self.summary_files:
            display = self._display_batch_name(path.parent)
            if display in self.summary_map:
                display = str(path)
            self.summary_map[display] = path
            values.append(display)

        self.batch_combo["values"] = values

        if not values:
            self.batch_combo.set("")
            self._load_run_records_from_histories(initial=initial)
            return

        selected = current if current in self.summary_map else values[0]
        self.batch_combo.set(selected)
        self._load_summary(self.summary_map[selected], initial=initial)

    def _on_batch_select(self, _event: tk.Event | None = None) -> None:
        """批次下拉框切换事件。"""
        selected = self.batch_combo.get()
        path = self.summary_map.get(selected)
        if path is not None:
            self._load_summary(path, initial=False)

    def _load_summary(self, path: Path, *, initial: bool) -> None:
        """读取 summary.json，填充表格、差值和运行图表选项。"""
        try:
            data = load_json(path)
        except Exception as error:
            messagebox.showerror("读取失败", f"无法读取 {path}：\n{error}")
            return

        self.summary_data = data
        self.summary_path = path
        self.batch_root = path.parent
        self.current_figure_kind = "curves"

        devices = sorted(
            {
                str(run.get("device"))
                for run in data.get("runs", [])
                if run.get("device")
            }
        )
        device_text = " / ".join(devices) if devices else "unknown"
        self.summary_meta_var.set(
            f"批次 {path.parent.name} · 运行 {data.get('run_count', '?')} 个 · "
            f"生成 {data.get('generated_at', '?')} · 设备 {device_text} · "
            "汇总表数值为 mean ± std"
        )

        self._fill_group_tree(data.get("groups", []))
        self._fill_delta_text(data)
        self._populate_run_choices(data.get("runs", []))

        self._set_example_sets(
            _discover_example_sets(path.parent, self.artifacts_root)
        )

        if initial and self.current_image_path is None and self.example_paths:
            self.root.after(180, self._load_example_and_predict)

    def _load_run_records_from_histories(self, *, initial: bool) -> None:
        """没有 summary.json 时，从 history.json 临时生成对比数据。"""
        records = _discover_runs(self.artifacts_root)
        if not records:
            self._clear_comparison("未找到实验结果。请先运行 batch 或 train/evaluate。")
            self._populate_run_choices([])
            self._set_example_sets(
                _discover_example_sets(None, self.artifacts_root)
            )
            return

        self.summary_data = build_summary(records)
        self.summary_path = None
        self.batch_root = None
        self.current_figure_kind = "curves"
        self.summary_meta_var.set(
            f"未找到 summary.json；已从 history.json 临时汇总 {len(records)} 个运行。"
        )
        self._fill_group_tree(self.summary_data.get("groups", []))
        self._fill_delta_text(self.summary_data)
        self._populate_run_choices(records)

        self._set_example_sets(
            _discover_example_sets(None, self.artifacts_root)
        )

        if initial and self.current_image_path is None and self.example_paths:
            self.root.after(180, self._load_example_and_predict)

    def _clear_comparison(self, message: str) -> None:
        """清空对比页并显示提示。"""
        self.summary_meta_var.set(message)
        self.delta_var.set("")
        for tree in (self.group_tree, self.run_tree):
            for item in tree.get_children():
                tree.delete(item)
        self.figure_run_combo["values"] = ()
        self.figure_run_var.set("")
        self._show_canvas_message(self.figure_canvas, "暂无可预览的运行")
        self.figure_caption_var.set("")

    def _fill_group_tree(self, groups: list[dict[str, Any]]) -> None:
        """填充阶段/模型汇总表。"""
        for item in self.group_tree.get_children():
            self.group_tree.delete(item)

        for index, group in enumerate(groups):
            values = (
                self._stage_text(group.get("stage", "?")),
                self._model_text(group.get("model", "?")),
                group.get("runs", 0),
                _format_pair(
                    group.get("test_accuracy_mean"),
                    group.get("test_accuracy_std"),
                    digits=2,
                    scale=100,
                    suffix="%",
                ),
                _format_pair(
                    group.get("test_loss_mean"),
                    group.get("test_loss_std"),
                    digits=4,
                ),
                _format_pair(
                    group.get("total_training_seconds_mean"),
                    group.get("total_training_seconds_std"),
                    digits=1,
                ),
            )
            self.group_tree.insert(
                "",
                "end",
                values=values,
                tags=("odd" if index % 2 else "even",),
            )

    def _fill_delta_text(self, data: dict[str, Any]) -> None:
        """整理 CNN−MLP 和增强−基础差值，显示成三行。"""
        cnn_mlp = {
            row.get("stage"): row
            for row in data.get("cnn_minus_mlp", {}).get("summary", [])
        }
        augmentation = {
            row.get("model"): row
            for row in data.get("augmented_minus_basic", {}).get("summary", [])
        }

        basic = cnn_mlp.get("basic", {})
        augmented = cnn_mlp.get("augmented", {})
        cnn_aug = augmentation.get("cnn", {})
        mlp_aug = augmentation.get("mlp", {})

        lines = [
            "CNN−MLP 准确率差 (pp)："
            f"basic {_format_signed(basic.get('mean_pp'), basic.get('std_pp'))}；"
            f"aug {_format_signed(augmented.get('mean_pp'), augmented.get('std_pp'))}",
            "增强−基础 准确率差 (pp)："
            f"CNN {_format_signed(cnn_aug.get('mean_accuracy_pp'), cnn_aug.get('std_accuracy_pp'))}；"
            f"MLP {_format_signed(mlp_aug.get('mean_accuracy_pp'), mlp_aug.get('std_accuracy_pp'))}",
            "增强−基础 耗时差 (s)："
            f"CNN {_format_signed(cnn_aug.get('mean_time_seconds'), cnn_aug.get('std_time_seconds'))}；"
            f"MLP {_format_signed(mlp_aug.get('mean_time_seconds'), mlp_aug.get('std_time_seconds'))}",
        ]
        self.delta_var.set("\n".join(lines))

    def _populate_run_choices(self, records: list[dict[str, Any]]) -> None:
        """填充逐次运行表，并生成权重/图表下拉框选项。"""
        current_checkpoint = self.checkpoint_combo.get()
        current_figure_run = self.figure_run_var.get()

        self.run_map.clear()
        self.checkpoint_map.clear()
        self.run_iid_map.clear()

        for item in self.run_tree.get_children():
            self.run_tree.delete(item)

        labels: list[str] = []
        for index, run in enumerate(records):
            run_dir = Path(run.get("run_dir", ""))
            if not run_dir.is_absolute():
                run_dir = PROJECT_ROOT / run_dir
            run["run_dir"] = str(run_dir)

            label = self._run_label(run)
            if label in self.run_map:
                label = f"{label} [{run_dir.name}]"
            self.run_map[label] = run
            labels.append(label)

            final_checkpoint = run_dir / "weights" / "final.pt"
            if final_checkpoint.exists():
                self.checkpoint_map[label] = final_checkpoint

            values = (
                self._stage_text(run.get("stage", "?")),
                self._model_text(run.get("model", "?")),
                run.get("seed", "?"),
                self._percent(run.get("test_accuracy")),
                self._number(run.get("test_loss"), 4),
                self._number(run.get("total_training_seconds"), 1),
                self._number(run.get("cpu_time_seconds"), 1),
                self._number(run.get("peak_rss_mib"), 1),
                (
                    f"{run['parameter_count']:,}"
                    if run.get("parameter_count") is not None
                    else "N/A"
                ),
            )
            iid = str(index)
            self.run_tree.insert(
                "",
                "end",
                iid=iid,
                values=values,
                tags=("odd" if index % 2 else "even",),
            )
            self.run_iid_map[iid] = label

        self.figure_run_combo["values"] = labels
        self.checkpoint_combo["values"] = list(self.checkpoint_map)

        best_label = self._best_run_label()
        selected_figure = (
            current_figure_run if current_figure_run in self.run_map else best_label
        )
        selected_checkpoint = (
            current_checkpoint
            if current_checkpoint in self.checkpoint_map
            else self._best_checkpoint_label()
        )

        self.figure_run_var.set(selected_figure or "")
        self.checkpoint_combo.set(selected_checkpoint or "")

        if records:
            best_index = 0
            if best_label is not None:
                for iid, label in self.run_iid_map.items():
                    if label == best_label:
                        best_index = int(iid)
                        break
            iid = str(best_index)
            if self.run_tree.exists(iid):
                self.run_tree.selection_set(iid)
                self.run_tree.see(iid)

        self.current_figure_kind = "curves"
        self._refresh_figure(notify=False)

    def _best_run_label(self) -> str | None:
        """返回测试准确率最高的运行标签。"""
        if not self.run_map:
            return None
        return max(
            self.run_map,
            key=lambda label: self.run_map[label].get("test_accuracy") or -1.0,
        )

    def _best_checkpoint_label(self) -> str | None:
        """返回拥有 final.pt 且准确率最高的权重标签。"""
        if not self.checkpoint_map:
            return None
        return max(
            self.checkpoint_map,
            key=lambda label: self.run_map[label].get("test_accuracy") or -1.0,
        )

    @staticmethod
    def _stage_text(stage: str) -> str:
        """表格里用简短中文阶段名，避免长英文挤占列宽。"""
        return {"basic": "基础", "augmented": "增强"}.get(stage, stage)

    @staticmethod
    def _model_text(model: str) -> str:
        """统一表格中的模型大小写。"""
        return model.upper() if model in ("cnn", "mlp") else model

    def _run_label(self, run: dict[str, Any]) -> str:
        accuracy = run.get("test_accuracy")
        accuracy_text = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"
        return (
            f"{self._stage_text(run.get('stage', '?'))} / "
            f"{self._model_text(run.get('model', '?'))} / "
            f"seed {run.get('seed', '?')}  ·  {accuracy_text}"
        )

    @staticmethod
    def _percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.2f}%"

    @staticmethod
    def _number(value: float | None, digits: int) -> str:
        return "N/A" if value is None else f"{value:.{digits}f}"

    # ------------------------------------------------------------------
    # 单图预测
    # ------------------------------------------------------------------
    def _choose_image(self) -> None:
        """打开文件选择框，载入用户图片。"""
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择手写数字图片",
            filetypes=(
                ("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
                ("所有文件", "*.*"),
            ),
        )
        if path:
            self._set_image(Path(path))

    def _set_example_sets(self, sets: dict[str, list[Path]]) -> None:
        """记录清晰/模糊示例集，并填充示例集下拉框。"""
        self.example_sets = {
            name: [path for path in paths if path.exists()]
            for name, paths in sets.items()
        }
        self.example_sets = {
            name: paths for name, paths in self.example_sets.items() if paths
        }

        names = list(self.example_sets)
        self.example_set_combo["values"] = names
        if not names:
            self.example_set_var.set("")
            self.example_paths = []
            self.example_index = -1
            self._update_example_button()
            return

        current = self.example_set_var.get()
        selected = current if current in self.example_sets else names[0]
        self.example_set_var.set(selected)
        self._apply_example_set(selected)

    def _apply_example_set(self, name: str) -> None:
        """切换当前示例列表，但不自动载入图片。"""
        self.example_paths = list(self.example_sets.get(name, []))
        self.example_index = -1
        self._update_example_button()

    def _on_example_set_change(self, _event: tk.Event | None = None) -> None:
        """示例集下拉框切换后，立即载入该组第一张并预测。"""
        self._apply_example_set(self.example_set_var.get())
        if not self.example_paths:
            return

        self.example_index = 0
        self._set_image(self.example_paths[0])
        self._update_example_button()
        if self.checkpoint_combo.get():
            self._predict()

    def _update_example_button(self) -> None:
        """按钮上显示当前示例序号和总数。"""
        total = len(self.example_paths)
        if total == 0:
            self.example_button.configure(state="disabled", text="使用示例")
            return

        self.example_button.configure(state="normal")
        if 0 <= self.example_index < total:
            text = f"使用示例 {self.example_index + 1}/{total}"
        else:
            text = f"使用示例 ({total})"
        self.example_button.configure(text=text)

    def _example_position(self, path: Path | None) -> tuple[int, int] | None:
        """返回图片在示例列表中的 (从1开始的序号, 总数)。"""
        if path is None or not self.example_paths:
            return None
        try:
            index = self.example_paths.index(path)
        except ValueError:
            return None
        return index + 1, len(self.example_paths)

    def _example_true_label(self, path: Path | None) -> int | None:
        """从 digit_0.png / digit_8_amb_01.png 推断真实标签。"""
        if path is None or path not in self.example_paths:
            return None
        parts = path.stem.split("_")
        if len(parts) >= 2 and parts[0] == "digit" and parts[1].isdigit():
            return int(parts[1])
        return None

    def _load_example(self) -> None:
        """循环切换到下一张示例，并自动预测。"""
        if not self.example_paths:
            messagebox.showinfo("提示", "当前没有可用示例图片。")
            return

        # 第一次点击从第 1 张开始，之后依次 2/N、3/N……到末尾后回到第 1 张。
        self.example_index = (self.example_index + 1) % len(self.example_paths)
        path = self.example_paths[self.example_index]
        self._set_image(path)
        self._update_example_button()
        if self.checkpoint_combo.get():
            self._predict()

    def _load_example_and_predict(self) -> None:
        """启动时自动载入第一张示例并预测，让界面立即有内容。"""
        if self.current_image_path is not None or not self.example_paths:
            return
        self.example_index = 0
        self._set_image(self.example_paths[0])
        self._update_example_button()
        if self.checkpoint_combo.get():
            self._predict()

    def _set_image(self, path: Path) -> None:
        """显示图片预览，并清空上一次预测结果。"""
        try:
            with Image.open(path) as source:
                source.verify()
        except Exception as error:
            messagebox.showerror("图片读取失败", f"{path}\n\n{error}")
            return

        self.current_image_path = path
        self._display_image(path, self.input_canvas, "input_photo")
        self.digit_label.configure(text="?", fg=ACCENT)
        self.confidence_var.set("等待预测")
        self.top3_var.set("点击“开始预测”")
        self.current_probabilities = None
        self.current_predicted = None
        self._draw_probability_canvas()

        position = self._example_position(path)
        if position is not None:
            index, total = position
            true_label = self._example_true_label(path)
            label_text = (
                f" · 真实标签 {true_label}" if true_label is not None else ""
            )
            set_name = self.example_set_var.get() or "示例"
            self.status_var.set(
                f"{set_name} {index}/{total}{label_text} · 已选择 {path.name}"
            )
        else:
            self.status_var.set(f"已选择图片：{path.name}")

    def _predict(self) -> None:
        """加载权重并预测当前图片。"""
        if self.current_image_path is None:
            messagebox.showwarning("缺少图片", "请先选择一张图片。")
            return

        label = self.checkpoint_combo.get()
        checkpoint_path = self.checkpoint_map.get(label)
        if checkpoint_path is None:
            messagebox.showwarning("缺少权重", "请选择可用的模型权重。")
            return
        if not checkpoint_path.exists():
            messagebox.showerror("权重不存在", str(checkpoint_path))
            return

        invert = INVERT_OPTIONS.get(self.invert_var.get(), "auto")
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            model, checkpoint = self._get_model(checkpoint_path)
            image = preprocess_single_image(self.current_image_path, invert=invert)
            predicted, probabilities = predict_single_image(
                model,
                image,
                self.device,
            )
        except Exception as error:
            messagebox.showerror("预测失败", f"{error}")
            return
        finally:
            self.root.configure(cursor="")

        probability_list = probabilities.tolist()
        confidence = float(probability_list[predicted])
        top_indices = torch.argsort(probabilities, descending=True)[:3].tolist()
        true_label = self._example_true_label(self.current_image_path)
        is_misclassified = true_label is not None and predicted != true_label

        self.current_probabilities = probability_list
        self.current_predicted = predicted
        # 有真实标签时，误判样本用红色突出显示，方便演示模糊样本。
        self.digit_label.configure(
            text=str(predicted),
            fg=DANGER if is_misclassified else ACCENT,
        )
        self.confidence_var.set(f"置信度 {confidence * 100:.3f}%")
        self.top3_var.set(
            "Top-3："
            + "　".join(
                f"{digit}  {probability_list[digit] * 100:.2f}%"
                for digit in top_indices
            )
        )
        self._draw_probability_canvas()

        model_name = checkpoint.get("config", {}).get("model", "?")
        position = self._example_position(self.current_image_path)
        sample_text = ""
        label_text = ""
        if position is not None:
            index, total = position
            set_name = self.example_set_var.get() or "示例"
            sample_text = f"{set_name} {index}/{total} · "
        if true_label is not None:
            label_text = f" · 真实标签 {true_label}"
        result_text = " · 误判" if is_misclassified else ""
        self.status_var.set(
            f"{sample_text}预测完成 · {model_name.upper()}{label_text}{result_text}"
        )

    def _get_model(
        self,
        checkpoint_path: Path,
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        """按设备缓存已加载模型，避免重复读取 checkpoint。"""
        key = (str(checkpoint_path), str(self.device))
        if key not in self.model_cache:
            model, checkpoint = load_model_from_checkpoint(
                checkpoint_path,
                self.device,
            )
            self.model_cache[key] = (model, checkpoint)
        return self.model_cache[key]

    # ------------------------------------------------------------------
    # 图表与图片显示
    # ------------------------------------------------------------------
    def _on_figure_run_change(self, _event: tk.Event | None = None) -> None:
        """切换运行下拉框时刷新当前图表。"""
        self._refresh_figure(notify=False)

    def _on_run_tree_select(self, _event: tk.Event | None = None) -> None:
        """选中逐次运行表的某一行时，同步右侧图表运行。"""
        selection = self.run_tree.selection()
        if not selection:
            return
        label = self.run_iid_map.get(selection[0])
        if label is not None and label in self.run_map:
            self.figure_run_var.set(label)
            self._refresh_figure(notify=False)

    def _show_figure(self, kind: str) -> None:
        """按钮点击：显式切换到某种图表。"""
        self.current_figure_kind = kind
        label = self.figure_run_var.get()
        run = self.run_map.get(label)
        if run is None:
            return

        run_dir = Path(run["run_dir"])
        path = run_dir / f"{kind}.png"
        if not path.exists():
            if kind in ("kernels", "feature_maps") and run.get("model") == "mlp":
                messagebox.showinfo(
                    "提示",
                    "MLP 没有卷积层，因此没有卷积核/特征图。",
                )
            else:
                messagebox.showinfo("提示", f"未找到图片：\n{path}")
            return

        self._refresh_figure(notify=False)

    def _refresh_figure(self, *, notify: bool) -> None:
        """按当前运行和图表类型更新预览。"""
        label = self.figure_run_var.get()
        run = self.run_map.get(label)
        if run is None:
            self._show_canvas_message(self.figure_canvas, "请选择一个运行")
            self.figure_caption_var.set("")
            return

        run_dir = Path(run["run_dir"])
        path = run_dir / f"{self.current_figure_kind}.png"

        # 从 CNN 切到 MLP 时，卷积核图不存在；自动回到曲线，避免弹窗。
        if not path.exists() and self.current_figure_kind != "curves":
            self.current_figure_kind = "curves"
            path = run_dir / "curves.png"

        if not path.exists():
            self._show_canvas_message(
                self.figure_canvas,
                f"未找到 {path.name}",
            )
            self.figure_caption_var.set(str(path))
            if notify:
                messagebox.showinfo("提示", f"未找到图片：\n{path}")
            return

        self._display_image(path, self.figure_canvas, "figure_photo")
        self.figure_caption_var.set(
            f"{self._stage_text(run.get('stage', '?'))} / "
            f"{self._model_text(run.get('model', '?'))} / "
            f"seed {run.get('seed', '?')} · {path.name}"
        )

    def _display_image(
        self,
        path: Path,
        canvas: tk.Canvas,
        photo_attribute: str,
    ) -> None:
        """把图片缩放后画到 Canvas；小图使用最近邻放大以保留像素。"""
        canvas.update_idletasks()
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 10:
            width = int(canvas["width"])
        if height <= 10:
            height = int(canvas["height"])

        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                max_width = max(1, width - 18)
                max_height = max(1, height - 18)

                if max(image.size) <= 64:
                    # MNIST 28x28 图片直接放大，保留像素边缘。
                    scale = min(max_width / image.width, max_height / image.height)
                    resized = (
                        max(1, int(image.width * scale)),
                        max(1, int(image.height * scale)),
                    )
                    image = image.resize(resized, Image.Resampling.NEAREST)
                else:
                    image.thumbnail(
                        (max_width, max_height),
                        Image.Resampling.LANCZOS,
                    )

                photo = ImageTk.PhotoImage(image)
        except Exception as error:
            messagebox.showerror("图片显示失败", f"{path}\n\n{error}")
            return

        canvas.delete("all")
        canvas.create_image(width // 2, height // 2, image=photo)
        setattr(self, photo_attribute, photo)

    def _redraw_input_preview(self) -> None:
        """Canvas 尺寸变化后重新绘制当前输入图片。"""
        if self.current_image_path is not None and self.current_image_path.exists():
            self._display_image(
                self.current_image_path,
                self.input_canvas,
                "input_photo",
            )
        else:
            self._show_canvas_message(self.input_canvas, "尚未选择图片")

    def _show_canvas_message(self, canvas: tk.Canvas, message: str) -> None:
        """在 Canvas 中央显示提示文字。"""
        canvas.update_idletasks()
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 10:
            width = int(canvas["width"])
        if height <= 10:
            height = int(canvas["height"])
        canvas.delete("all")
        canvas.create_text(
            width // 2,
            height // 2,
            text=message,
            fill=MUTED,
            font=(self.font_family, 11),
            tags="placeholder",
        )

    def _draw_probability_canvas(self) -> None:
        """绘制 10 类概率柱状图。"""
        canvas = self.prob_canvas
        canvas.delete("all")
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 10:
            width = 640
        if height <= 10:
            height = 260

        left, right, top, bottom = 58, 18, 24, 34
        plot_width = max(1, width - left - right)
        plot_height = max(1, height - top - bottom)
        base_y = top + plot_height

        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + plot_height * (1.0 - fraction)
            canvas.create_line(
                left,
                y,
                width - right,
                y,
                fill="#e8eaed",
            )
            canvas.create_text(
                left - 8,
                y,
                text=f"{fraction * 100:.0f}%",
                anchor="e",
                fill=MUTED,
                font=(self.font_family, 9),
            )

        if not self.current_probabilities:
            canvas.create_text(
                width // 2,
                top + plot_height // 2,
                text="选择图片并点击“开始预测”",
                fill=MUTED,
                font=(self.font_family, 11),
            )
            return

        slot = plot_width / 10.0
        bar_width = slot * 0.62
        for digit, probability in enumerate(self.current_probabilities):
            x0 = left + slot * digit + (slot - bar_width) / 2.0
            x1 = x0 + bar_width
            y0 = top + plot_height * (1.0 - probability)
            color = DANGER if digit == self.current_predicted else ACCENT
            canvas.create_rectangle(
                x0,
                y0,
                x1,
                base_y,
                fill=color,
                outline="",
            )
            canvas.create_text(
                (x0 + x1) / 2.0,
                base_y + 14,
                text=str(digit),
                fill=TEXT,
                font=(self.font_family, 10),
            )
            if probability >= 0.02:
                canvas.create_text(
                    (x0 + x1) / 2.0,
                    max(top + 6, y0 - 9),
                    text=f"{probability * 100:.0f}%",
                    fill=TEXT,
                    font=(self.font_family, 8),
                )

    # ------------------------------------------------------------------
    # 汇总文件操作
    # ------------------------------------------------------------------
    def _open_summary(self) -> None:
        """用系统默认程序打开 summary.md。"""
        if self.summary_path is None or not self.summary_path.exists():
            messagebox.showinfo(
                "提示",
                "当前对比数据没有对应的 summary.md 文件。",
            )
            return
        try:
            subprocess.Popen(
                ["xdg-open", str(self.summary_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as error:
            messagebox.showinfo(
                "汇总文件",
                f"无法自动打开，请手动查看：\n{self.summary_path}\n\n{error}",
            )


def launch_gui(
    artifacts_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    device: torch.device | None = None,
) -> None:
    """创建并运行 GUI 主循环。"""
    root = tk.Tk()
    DigitRecognitionGUI(root, artifacts_root=artifacts_root, device=device)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
