"""Desktop GUI for DataPolisher — CustomTkinter edition."""

from __future__ import annotations

import os
import random
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk
from PIL import Image, ImageTk

from . import cli as backend
from . import feed_eye
from .red_number_fonts import RED_NUMBER_FAMILY, register_red_number_for_gui


PREVIEW_MAX_WIDTH = 340
PREVIEW_MAX_HEIGHT = 600

_WAITING = "waiting"
_RUNNING = "running"
_DONE = "done"
_FAILED = "failed"

_STATUS_LABEL = {
    _WAITING: "等待中",
    _RUNNING: "处理中",
    _DONE: "完成",
    _FAILED: "失败",
}


def _parse_range_pair(
    lo_raw: str,
    hi_raw: str,
    label: str,
    *,
    lower_floor: int = 0,
) -> tuple[int, int]:
    lo_s, hi_s = lo_raw.strip(), hi_raw.strip()
    if not lo_s or not hi_s:
        raise ValueError(f"{label}：请填写最小值与最大值")
    try:
        lo, hi = int(lo_s), int(hi_s)
    except ValueError as exc:
        raise ValueError(f"{label} 须为整数") from exc
    if lo > hi:
        lo, hi = hi, lo
    if lo < lower_floor or hi < lower_floor:
        raise ValueError(f"{label} 不能小于 {lower_floor}")
    return lo, hi


def _parse_int_range_text(
    raw: str,
    label: str,
    *,
    lower_floor: int = 0,
) -> tuple[int, int]:
    s = raw.strip().replace("－", "-").replace("–", "-").replace("—", "-")
    if not s:
        raise ValueError(f"{label} 不能为空")
    if "-" in s:
        left, right = s.split("-", 1)
        lo_s, hi_s = left.strip(), right.strip()
        if not lo_s or not hi_s:
            raise ValueError(f"{label} 范围格式应为「最小-最大」，例如 100-200")
    else:
        lo_s = hi_s = s.strip()
    return _parse_range_pair(lo_s, hi_s, label, lower_floor=lower_floor)


def resolve_logo_path() -> Optional[Path]:
    """Bundled ``static/logo.png``, or ``DATAPOLISHER_LOGO`` env override."""

    env = (os.environ.get("DATAPOLISHER_LOGO") or "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    bundled = Path(__file__).resolve().parent / "static" / "logo.png"
    if bundled.is_file():
        return bundled
    return None


def logo_png_path() -> Path:
    """Default bundled logo path (may not exist)."""

    return Path(__file__).resolve().parent / "static" / "logo.png"


def load_sidebar_logo_ctk(max_w: int = 260, max_h: int = 96) -> Optional[ctk.CTkImage]:
    path = resolve_logo_path()
    if path is None:
        return None
    try:
        img = Image.open(path).convert("RGBA")
        w0, h0 = img.size
        scale = min(max_w / max(w0, 1), max_h / max(h0, 1), 1.0)
        nw = max(1, int(w0 * scale))
        nh = max(1, int(h0 * scale))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        return ctk.CTkImage(light_image=img, dark_image=img, size=(nw, nh))
    except Exception:
        return None


def _display_scale(widget) -> float:
    try:
        return max(1.0, widget.winfo_fpixels("1i") / 72.0)
    except Exception:
        return 1.0


def preview_pil_and_size(image: Image.Image, widget=None) -> Tuple[Image.Image, int, int]:
    scale = _display_scale(widget) if widget is not None else 1.0
    phys_max_w = int(PREVIEW_MAX_WIDTH * scale)
    phys_max_h = int(PREVIEW_MAX_HEIGHT * scale)
    width, height = image.size
    fit = min(phys_max_w / width, phys_max_h / height, 1.0)
    phys_w = max(1, int(width * fit))
    phys_h = max(1, int(height * fit))
    resized = image.resize((phys_w, phys_h), Image.Resampling.LANCZOS)
    logical_w = max(1, int(phys_w / scale))
    logical_h = max(1, int(phys_h / scale))
    return resized, logical_w, logical_h


def fit_preview(image: Image.Image, widget=None) -> tuple:
    """Legacy Tk PhotoImage helper (unused in CTk path; kept for compatibility)."""

    resized, lw, lh = preview_pil_and_size(image, widget)
    photo = ImageTk.PhotoImage(resized)
    return photo, lw, lh


@dataclass
class TaskItem:
    path: Path
    args: SimpleNamespace
    status: str = _WAITING
    progress: str = ""
    result_image: Optional[Image.Image] = None
    original_image: Optional[Image.Image] = None
    error: Optional[str] = None
    output_path: Optional[Path] = None


class DataPolisherApp(ctk.CTk):
    ACCENT = "#546de8"
    ACCENT_HOVER = "#4559d0"
    SUCCESS = "#10a37f"
    SUCCESS_HOVER = "#0c8f6e"
    PAGE_BG = "#eef0f5"
    CARD = "#ffffff"
    PREVIEW_BG = "#e8ecf4"
    MUTED = "#64748b"

    def __init__(self) -> None:
        super().__init__()
        self.title("DataPolisher")
        self.geometry("1220x820")
        self.minsize(1040, 720)
        self.configure(fg_color=self.PAGE_BG)

        self._tasks: List[TaskItem] = []
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_requested = False

        self._original_tk: Optional[ImageTk.PhotoImage] = None
        self._result_tk: Optional[ImageTk.PhotoImage] = None
        self._logo_ctk: Optional[ctk.CTkImage] = None
        self._digit_font_family: Optional[str] = (
            RED_NUMBER_FAMILY if register_red_number_for_gui() else None
        )

        self._configure_tree_style()
        self._build_layout()

    def _configure_tree_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "DP.Treeview",
            background=self.CARD,
            fieldbackground=self.CARD,
            foreground="#334155",
            borderwidth=0,
            rowheight=32,
            font=("Segoe UI", 11),
        )
        style.configure(
            "DP.Treeview.Heading",
            background="#f1f2f6",
            foreground=self.MUTED,
            relief="flat",
            font=("Segoe UI", 10),
        )
        style.map(
            "DP.Treeview",
            background=[("selected", "#ebeefe")],
            foreground=[("selected", "#4338ca")],
        )

    def _build_layout(self) -> None:
        root = ctk.CTkFrame(self, fg_color=self.PAGE_BG, corner_radius=0)
        root.pack(fill="both", expand=True)

        # ── Sidebar ────────────────────────────────────────────────────────
        sidebar = ctk.CTkFrame(
            root,
            width=318,
            corner_radius=16,
            fg_color=self.CARD,
            border_width=0,
        )
        sidebar.pack(side="left", fill="y", padx=(22, 12), pady=22)
        sidebar.pack_propagate(False)

        inner = ctk.CTkFrame(sidebar, fg_color="transparent", corner_radius=0)
        inner.pack(fill="both", expand=True, padx=18, pady=18)

        try:
            self._logo_ctk = load_sidebar_logo_ctk()
        except Exception:
            self._logo_ctk = None
        if self._logo_ctk is not None:
            ctk.CTkLabel(inner, text="", image=self._logo_ctk).pack(anchor="w", pady=(0, 8))
        else:
            ctk.CTkLabel(
                inner,
                text="DataPolisher",
                font=ctk.CTkFont(size=20, weight="bold"),
                text_color="#1e293b",
            ).pack(anchor="w")

        ctk.CTkLabel(
            inner,
            text="截图数据批量美化",
            font=ctk.CTkFont(size=12),
            text_color=self.MUTED,
        ).pack(anchor="w", pady=(0, 14))

        self.tabs = ctk.CTkTabview(inner, width=274, height=400)
        self.tabs.pack(fill="both", expand=True)

        tab_detail = self.tabs.add("详细数据")
        tab_eye = self.tabs.add("小眼睛")

        # --- 详细数据 ---
        ctk.CTkLabel(
            tab_detail,
            text="指标范围",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#1e293b",
        ).pack(anchor="w", pady=(4, 4))
        ctk.CTkLabel(
            tab_detail,
            text="每张图在区间内随机取整数（曝光须 ≥ 1）。顶部小眼睛与「新观看数」相同。",
            font=ctk.CTkFont(size=11),
            text_color=self.MUTED,
            wraplength=252,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        self.entries: Dict[str, ctk.CTkEntry] = {}
        self._add_range_fields(tab_detail, "exposure", "新曝光数", "900", "1100", lower_floor=1)
        self._add_range_fields(tab_detail, "views", "新观看数", "250", "350", lower_floor=0)
        ctk.CTkLabel(
            tab_eye,
            text="适用于信息流列表截图：只改封面左下角小眼睛旁的浏览数。",
            font=ctk.CTkFont(size=11),
            text_color=self.MUTED,
            wraplength=252,
            justify="left",
        ).pack(anchor="w", pady=(4, 12))
        ctk.CTkLabel(
            tab_eye,
            text="标题关键词",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#1e293b",
        ).pack(anchor="w", pady=(0, 6))

        self.eye_entries: Dict[str, ctk.CTkEntry] = {}
        et = ctk.CTkEntry(tab_eye, height=36, corner_radius=10, border_width=1)
        et.pack(fill="x", pady=(0, 14))
        self.eye_entries["title"] = et

        self._add_range_fields(
            tab_eye,
            "eye_views",
            "浏览量随机范围",
            "80",
            "120",
            lower_floor=0,
            key_lo="eye_views_lo",
            key_hi="eye_views_hi",
        )

        ctk.CTkFrame(inner, height=1, fg_color="#e8eaef", corner_radius=0).pack(fill="x", pady=(18, 14))

        ctk.CTkButton(
            inner,
            text="添加图片…",
            height=42,
            corner_radius=12,
            fg_color=self.ACCENT,
            hover_color=self.ACCENT_HOVER,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.on_add_files,
        ).pack(fill="x")

        ctk.CTkButton(
            inner,
            text="全部开始",
            height=42,
            corner_radius=12,
            fg_color=self.SUCCESS,
            hover_color=self.SUCCESS_HOVER,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.on_start_all,
        ).pack(fill="x", pady=(12, 0))

        ctk.CTkButton(
            inner,
            text="清空列表",
            height=34,
            corner_radius=10,
            fg_color="#f1f2f6",
            hover_color="#e8eaef",
            text_color=self.MUTED,
            font=ctk.CTkFont(size=13),
            command=self.on_clear_tasks,
        ).pack(fill="x", pady=(10, 0))

        ctk.CTkFrame(inner, height=1, fg_color="#e8eaef", corner_radius=0).pack(fill="x", pady=(16, 12))

        self.global_status = ctk.CTkLabel(
            inner,
            text="任务列表为空\n添加图片后开始处理",
            font=ctk.CTkFont(size=11),
            text_color=self.MUTED,
            justify="left",
            anchor="w",
        )
        self.global_status.pack(anchor="w")

        # ── Main：右侧用原生 Tk 承载 Treeview / 预览，避免嵌在 CTkFrame 内导致启动异常
        main = tk.Frame(root, bg=self.PAGE_BG, highlightthickness=0)
        main.pack(side="left", fill="both", expand=True, padx=(0, 22), pady=22)

        list_shell = tk.Frame(main, bg=self.CARD, highlightthickness=0)
        list_shell.pack(fill="x")

        hdr = tk.Frame(list_shell, bg=self.CARD, highlightthickness=0)
        hdr.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(
            hdr,
            text="任务列表",
            bg=self.CARD,
            fg="#1e293b",
            font=("Arial", 15, "bold"),
        ).pack(side="left")

        def _link_btn(txt: str, cmd) -> None:
            tk.Button(
                hdr,
                text=txt,
                command=cmd,
                relief="flat",
                bd=0,
                bg=self.CARD,
                fg=self.ACCENT,
                activeforeground=self.ACCENT_HOVER,
                activebackground=self.CARD,
                cursor="hand2",
                font=("Arial", 11),
                padx=6,
                pady=2,
                highlightthickness=0,
            ).pack(side="right", padx=(8, 0))

        _link_btn("移除选中", self.on_remove_selected)
        _link_btn("批量保存完成项", self.on_batch_save)
        _link_btn("保存选中结果", self.on_save_selected)

        tree_host = tk.Frame(list_shell, bg=self.CARD, highlightthickness=0)
        tree_host.pack(fill="x", padx=12, pady=(4, 14))

        self.tree = ttk.Treeview(
            tree_host,
            columns=("file", "status", "progress"),
            show="headings",
            height=10,
            selectmode="browse",
            style="DP.Treeview",
        )
        self.tree.heading("file", text="文件")
        self.tree.heading("status", text="状态")
        self.tree.heading("progress", text="进度")
        self.tree.column("file", width=320, stretch=True)
        self.tree.column("status", width=76, stretch=False, anchor=tk.CENTER)
        self.tree.column("progress", width=220, stretch=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_task_selected)

        vsb = ttk.Scrollbar(tree_host, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")

        self.tree.tag_configure("waiting", foreground=self.MUTED)
        self.tree.tag_configure("running", foreground=self.ACCENT)
        self.tree.tag_configure("done", foreground=self.SUCCESS)
        self.tree.tag_configure("failed", foreground="#e25555")

        preview_wrap = tk.Frame(main, bg=self.CARD, highlightthickness=1, highlightbackground="#eceef3")
        preview_wrap.pack(fill="both", expand=True, pady=(16, 0))

        tk.Label(
            preview_wrap,
            text="预览 · 点击任务行查看",
            bg=self.CARD,
            fg=self.MUTED,
            font=("Arial", 12, "bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 6))

        pv_row = tk.Frame(preview_wrap, bg=self.CARD, highlightthickness=0)
        pv_row.pack(fill="both", expand=True, padx=12, pady=(0, 14))

        left_p = tk.Frame(pv_row, bg=self.CARD, highlightthickness=0)
        left_p.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(left_p, text="原图", bg=self.CARD, fg=self.MUTED, font=("Arial", 11)).pack(anchor="w")
        self.original_canvas = tk.Label(
            left_p,
            bg=self.PREVIEW_BG,
            text="暂无",
            fg=self.MUTED,
            font=("Arial", 12),
            bd=0,
            highlightthickness=0,
        )
        self.original_canvas.pack(fill="both", expand=True, pady=(6, 0))

        right_p = tk.Frame(pv_row, bg=self.CARD, highlightthickness=0)
        right_p.pack(side="left", fill="both", expand=True, padx=(8, 0))
        tk.Label(right_p, text="美化结果", bg=self.CARD, fg=self.MUTED, font=("Arial", 11)).pack(
            anchor="w"
        )
        self.result_canvas = tk.Label(
            right_p,
            bg=self.PREVIEW_BG,
            text="暂无",
            fg=self.MUTED,
            font=("Arial", 12),
            bd=0,
            highlightthickness=0,
        )
        self.result_canvas.pack(fill="both", expand=True, pady=(6, 0))

    def _add_range_fields(
        self,
        parent,
        prefix: str,
        label: str,
        default_lo: str,
        default_hi: str,
        *,
        lower_floor: int,
        key_lo: Optional[str] = None,
        key_hi: Optional[str] = None,
    ) -> None:
        kl = key_lo or f"{prefix}_lo"
        kh = key_hi or f"{prefix}_hi"
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(
            box,
            text=label,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#334155",
        ).pack(anchor="w", pady=(0, 8))
        row = ctk.CTkFrame(box, fg_color="transparent")
        row.pack(fill="x")
        ek: Dict[str, object] = dict(height=34, corner_radius=10, justify="center")
        if self._digit_font_family:
            ek["font"] = ctk.CTkFont(family=self._digit_font_family, size=15, weight="normal")
        el = ctk.CTkEntry(row, **ek)
        el.insert(0, default_lo)
        el.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(row, text="—", width=20, text_color=self.MUTED).pack(side="left")
        eh = ctk.CTkEntry(row, **ek)
        eh.insert(0, default_hi)
        eh.pack(side="left", fill="x", expand=True)
        self.entries[kl] = el
        self.entries[kh] = eh

    def _parse_metrics(self) -> SimpleNamespace:
        er = _parse_range_pair(
            self.entries["exposure_lo"].get(),
            self.entries["exposure_hi"].get(),
            "新曝光数",
            lower_floor=1,
        )
        vr = _parse_range_pair(
            self.entries["views_lo"].get(),
            self.entries["views_hi"].get(),
            "新观看数",
            lower_floor=0,
        )
        return SimpleNamespace(exposure_range=er, views_range=vr)

    def _make_args(self, path: Path, m: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(
            normal=str(path),
            output=None,
            exposure_range=m.exposure_range,
            views_range=m.views_range,
            likes=0,
            comments=0,
            collects=0,
            shares=0,
            ocr=True,
            inspect=False,
            glyph_atlas=False,
            style_report=False,
            eye_mode=False,
        )

    def _validate_eye_inputs(self) -> None:
        if not self.eye_entries["title"].get().strip():
            raise ValueError("标题关键词不能为空")
        _parse_range_pair(
            self.entries["eye_views_lo"].get(),
            self.entries["eye_views_hi"].get(),
            "浏览量范围",
            lower_floor=0,
        )

    def _make_eye_args(self, path: Path) -> SimpleNamespace:
        self._validate_eye_inputs()
        title = self.eye_entries["title"].get().strip()
        rng = _parse_range_pair(
            self.entries["eye_views_lo"].get(),
            self.entries["eye_views_hi"].get(),
            "浏览量范围",
            lower_floor=0,
        )
        return SimpleNamespace(
            normal=str(path),
            output=None,
            eye_title=title,
            eye_views_range=rng,
            ocr=True,
            glyph_atlas=False,
            eye_mode=True,
        )

    def on_add_files(self) -> None:
        tab_name = self.tabs.get()
        if tab_name == "详细数据":
            try:
                metrics = self._parse_metrics()
            except ValueError as exc:
                messagebox.showerror("输入有误", str(exc))
                return
        else:
            try:
                self._validate_eye_inputs()
            except ValueError as exc:
                messagebox.showerror("输入有误", str(exc))
                return

        paths = filedialog.askopenfilenames(
            title="选择图片（可多选）",
            filetypes=[("图片", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")],
        )
        if not paths:
            return

        for p in paths:
            path = Path(p)
            if tab_name == "详细数据":
                args = self._make_args(path, metrics)
            else:
                args = self._make_eye_args(path)
            task = TaskItem(path=path, args=args)
            self._tasks.append(task)
            self.tree.insert(
                "",
                tk.END,
                iid=str(id(task)),
                values=(path.name, _STATUS_LABEL[_WAITING], ""),
                tags=(_WAITING,),
            )
        self._refresh_global_status()

    def on_remove_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        task = self._task_by_iid(iid)
        if task and task.status == _RUNNING:
            messagebox.showwarning("提示", "任务正在处理中，无法移除。")
            return
        if task:
            self._tasks.remove(task)
        self.tree.delete(iid)
        self._clear_preview()
        self._refresh_global_status()

    def on_clear_tasks(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            if not messagebox.askyesno("确认", "正在处理，确定要清空任务列表并停止吗？"):
                return
            self._stop_requested = True
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self._tasks.clear()
        self._clear_preview()
        self._refresh_global_status()

    def _task_by_iid(self, iid: str) -> Optional[TaskItem]:
        for t in self._tasks:
            if str(id(t)) == iid:
                return t
        return None

    def _update_row(self, task: TaskItem) -> None:
        iid = str(id(task))
        if not self.tree.exists(iid):
            return
        self.tree.item(
            iid,
            values=(task.path.name, _STATUS_LABEL[task.status], task.progress),
            tags=(task.status,),
        )

    def _refresh_global_status(self) -> None:
        total = len(self._tasks)
        done = sum(1 for t in self._tasks if t.status == _DONE)
        running = sum(1 for t in self._tasks if t.status == _RUNNING)
        failed = sum(1 for t in self._tasks if t.status == _FAILED)
        waiting = sum(1 for t in self._tasks if t.status == _WAITING)
        if total == 0:
            self.global_status.configure(text="任务列表为空\n添加图片后开始处理")
        elif running:
            self.global_status.configure(text=f"处理中… {done}/{total} 已完成")
        elif done == total:
            self.global_status.configure(text=f"全部完成 · {done} 张\n可批量保存结果")
        else:
            parts = []
            if waiting:
                parts.append(f"等待 {waiting}")
            if done:
                parts.append(f"完成 {done}")
            if failed:
                parts.append(f"失败 {failed}")
            self.global_status.configure(text=" · ".join(parts))

    def on_start_all(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("提示", "已有任务在处理中，请等待完成。")
            return
        pending = [t for t in self._tasks if t.status == _WAITING]
        if not pending:
            messagebox.showinfo("提示", "没有等待中的任务。\n\n请先添加图片。")
            return
        self._stop_requested = False
        self._worker_thread = threading.Thread(target=self._run_all, args=(pending,), daemon=True)
        self._worker_thread.start()

    def _run_all(self, tasks: list[TaskItem]) -> None:
        for task in tasks:
            if self._stop_requested:
                break
            self.after(0, self._set_task_running, task)
            try:

                def _progress(msg: str, _task=task) -> None:
                    self.after(0, self._set_task_progress, _task, msg)

                original = Image.open(task.args.normal).convert("RGB")
                task.original_image = original

                if getattr(task.args, "eye_mode", False):
                    result = feed_eye.beautify_feed_card_eye(task.args, on_progress=_progress)
                elif hasattr(task.args, "exposure_range"):
                    el, eh = task.args.exposure_range
                    vl, vh = task.args.views_range
                    exp_v = random.randint(el, eh)
                    views_v = random.randint(vl, vh)
                    metrics = backend.calculate_metrics(
                        exposure=exp_v,
                        views=views_v,
                        likes=0,
                        comments=0,
                        collects=0,
                        shares=0,
                    )
                    result = backend.beautify_normal_with_ocr(task.args, metrics, on_progress=_progress)
                else:
                    metrics = backend.calculate_metrics(
                        exposure=task.args.exposure,
                        views=task.args.views,
                        likes=getattr(task.args, "likes", 0),
                        comments=getattr(task.args, "comments", 0),
                        collects=getattr(task.args, "collects", 0),
                        shares=getattr(task.args, "shares", 0),
                    )
                    result = backend.beautify_normal_with_ocr(task.args, metrics, on_progress=_progress)
                self.after(0, self._set_task_done, task, result)
            except Exception as exc:  # noqa: BLE001
                self.after(0, self._set_task_failed, task, exc)

        self.after(0, self._refresh_global_status)

    def _set_task_running(self, task: TaskItem) -> None:
        task.status = _RUNNING
        task.progress = "启动…"
        self._update_row(task)
        self._refresh_global_status()

    def _set_task_progress(self, task: TaskItem, msg: str) -> None:
        task.progress = msg
        self._update_row(task)

    def _set_task_done(self, task: TaskItem, result: Image.Image) -> None:
        task.status = _DONE
        task.progress = "完成 ✓"
        task.result_image = result
        self._update_row(task)
        self._refresh_global_status()
        iid = str(id(task))
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.see(iid)

    def _set_task_failed(self, task: TaskItem, exc: Exception) -> None:
        task.status = _FAILED
        task.progress = str(exc)[:60]
        task.error = str(exc)
        self._update_row(task)
        self._refresh_global_status()

    def _on_task_selected(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        task = self._task_by_iid(sel[0])
        if task is None:
            return
        if task.original_image:
            self._show_preview_tk(self.original_canvas, task.original_image, "_original_tk")
        else:
            self._clear_preview_label(self.original_canvas)
        if task.result_image:
            self._show_preview_tk(self.result_canvas, task.result_image, "_result_tk")
        else:
            msg = task.progress if task.status == _RUNNING else "尚未生成"
            self.result_canvas.configure(image=None, text=msg, fg=self.MUTED)

    def _show_preview_tk(self, label: tk.Label, image: Image.Image, attr: str) -> None:
        pil_img, _lw, _lh = preview_pil_and_size(image, widget=label)
        photo = ImageTk.PhotoImage(pil_img)
        setattr(self, attr, photo)
        label.configure(image=photo, text="", fg="#1e293b")

    def _clear_preview_label(self, label: tk.Label, text: str = "暂无") -> None:
        label.configure(image=None, text=text, fg=self.MUTED)

    def _clear_preview(self) -> None:
        self._original_tk = None
        self._result_tk = None
        self._clear_preview_label(self.original_canvas)
        self._clear_preview_label(self.result_canvas)

    def on_save_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("提示", "请先在任务列表中选中一行。")
            return
        task = self._task_by_iid(sel[0])
        if task is None or task.result_image is None:
            messagebox.showinfo("提示", "所选任务尚未完成，无结果可保存。")
            return
        self._save_task(task)

    def on_batch_save(self) -> None:
        done = [t for t in self._tasks if t.status == _DONE and t.result_image]
        if not done:
            messagebox.showinfo("提示", "没有已完成的任务可保存。")
            return
        folder = filedialog.askdirectory(title="选择保存目录")
        if not folder:
            return
        folder_path = Path(folder)
        saved, errors = 0, []
        for task in done:
            out = folder_path / f"{task.path.stem}-polished.jpg"
            try:
                task.result_image.save(str(out), "JPEG", quality=92)
                task.output_path = out
                task.progress = f"已保存 → {out.name}"
                self._update_row(task)
                saved += 1
            except Exception as exc:
                errors.append(f"{task.path.name}: {exc}")
        msg = f"已保存 {saved} 张到\n{folder}"
        if errors:
            msg += "\n\n失败：\n" + "\n".join(errors)
        messagebox.showinfo("批量保存完成", msg)

    def _save_task(self, task: TaskItem) -> None:
        suggested = f"{task.path.stem}-polished.jpg"
        out = filedialog.asksaveasfilename(
            title="保存美化图",
            defaultextension=".jpg",
            initialfile=suggested,
            filetypes=[("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png")],
        )
        if not out:
            return
        ext = Path(out).suffix.lower()
        try:
            if ext in (".jpg", ".jpeg"):
                task.result_image.save(out, "JPEG", quality=92)
            else:
                task.result_image.save(out)
            task.output_path = Path(out)
            task.progress = f"已保存 → {Path(out).name}"
            self._update_row(task)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))

    def set_status(self, text: str) -> None:
        self.global_status.configure(text=text)


def main() -> int:
    import traceback

    try:
        # macOS：关闭窗口标题栏操控，避免部分环境下启动即崩溃
        if sys.platform == "darwin":
            setattr(ctk.CTk, "_deactivate_macos_window_header_manipulation", True)
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        register_red_number_for_gui()
        app = DataPolisherApp()
        app.mainloop()
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
