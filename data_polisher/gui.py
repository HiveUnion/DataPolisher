"""Desktop GUI for DataPolisher — task-queue edition.

Supports beautifying multiple images in sequence.  Each image is shown as a
row in the task list with its own status / progress.  Clicking a row previews
that image's original and result side by side.
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace
from typing import Callable, List, Optional

from PIL import Image, ImageTk

from . import cli as backend


PREVIEW_MAX_WIDTH = 340
PREVIEW_MAX_HEIGHT = 600

# Task status constants
_WAITING = "waiting"
_RUNNING = "running"
_DONE    = "done"
_FAILED  = "failed"

_STATUS_LABEL = {
    _WAITING: "等待中",
    _RUNNING: "处理中",
    _DONE:    "完成",
    _FAILED:  "失败",
}
_STATUS_COLOR = {
    _WAITING: "#888",
    _RUNNING: "#0070c0",
    _DONE:    "#107c10",
    _FAILED:  "#c50f1f",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _display_scale(widget) -> float:
    try:
        return max(1.0, widget.winfo_fpixels("1i") / 72.0)
    except Exception:
        return 1.0


def fit_preview(image: Image.Image, widget=None) -> tuple:
    scale = _display_scale(widget) if widget is not None else 1.0
    phys_max_w = int(PREVIEW_MAX_WIDTH * scale)
    phys_max_h = int(PREVIEW_MAX_HEIGHT * scale)
    width, height = image.size
    fit = min(phys_max_w / width, phys_max_h / height, 1.0)
    phys_w = max(1, int(width * fit))
    phys_h = max(1, int(height * fit))
    resized = image.resize((phys_w, phys_h), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(resized)
    logical_w = max(1, int(phys_w / scale))
    logical_h = max(1, int(phys_h / scale))
    return photo, logical_w, logical_h


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskItem:
    path: Path
    args: SimpleNamespace
    status: str = _WAITING
    progress: str = ""          # short human-readable step text
    result_image: Optional[Image.Image] = None
    original_image: Optional[Image.Image] = None
    error: Optional[str] = None
    output_path: Optional[Path] = None  # set after auto-save


# ──────────────────────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────────────────────

class DataPolisherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DataPolisher · 数据美化工具")
        self.geometry("1120x760")
        self.minsize(960, 680)

        self._tasks: List[TaskItem] = []
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_requested = False

        # PhotoImage references (must stay alive as long as labels show them)
        self._original_tk: Optional[ImageTk.PhotoImage] = None
        self._result_tk:   Optional[ImageTk.PhotoImage] = None

        self._build_layout()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        # ── Left panel: metrics form ──────────────────────────────────────────
        left = ttk.Frame(self, padding=12)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="指标数据", font=("", 11, "bold")).pack(anchor=tk.W, pady=(0, 8))

        self.entries: dict[str, ttk.Entry] = {}
        self._add_entry(left, "exposure", "新曝光数", "1000")
        self._add_entry(left, "views",    "新观看数", "300")
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(left, text="互动数据（可选）", foreground="#666").pack(anchor=tk.W)
        self._add_entry(left, "likes",    "点赞数", "0")
        self._add_entry(left, "comments", "评论数", "0")
        self._add_entry(left, "collects", "收藏数", "0")
        self._add_entry(left, "shares",   "分享数", "0")

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        ttk.Button(left, text="＋ 添加图片…", command=self.on_add_files).pack(fill=tk.X)
        ttk.Button(left, text="▶ 全部开始",   command=self.on_start_all).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(left, text="✕ 清空列表",   command=self.on_clear_tasks).pack(fill=tk.X, pady=(4, 0))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)
        self.global_status = ttk.Label(
            left, text="", foreground="#888", wraplength=170, justify=tk.LEFT
        )
        self.global_status.pack(anchor=tk.W)

        # ── Right panel: task list + preview ─────────────────────────────────
        right = ttk.Frame(self, padding=(0, 12, 12, 12))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Task list header
        list_header = ttk.Frame(right)
        list_header.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(list_header, text="任务列表", font=("", 11, "bold")).pack(side=tk.LEFT)
        ttk.Button(
            list_header, text="保存选中结果", command=self.on_save_selected
        ).pack(side=tk.RIGHT)
        ttk.Button(
            list_header, text="批量保存完成项", command=self.on_batch_save
        ).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(
            list_header, text="移除选中", command=self.on_remove_selected
        ).pack(side=tk.RIGHT, padx=(0, 6))

        # Treeview for tasks
        tree_frame = ttk.Frame(right)
        tree_frame.pack(fill=tk.X)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("file", "status", "progress"),
            show="headings",
            height=8,
            selectmode="browse",
        )
        self.tree.heading("file",     text="文件名")
        self.tree.heading("status",   text="状态")
        self.tree.heading("progress", text="进度")
        self.tree.column("file",     width=280, stretch=True)
        self.tree.column("status",   width=70,  stretch=False, anchor=tk.CENTER)
        self.tree.column("progress", width=160, stretch=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_task_selected)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,   command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)

        # Row-colour tags
        self.tree.tag_configure("waiting", foreground="#888")
        self.tree.tag_configure("running", foreground="#0070c0")
        self.tree.tag_configure("done",    foreground="#107c10")
        self.tree.tag_configure("failed",  foreground="#c50f1f")

        # ── Preview panels ────────────────────────────────────────────────────
        preview_outer = ttk.LabelFrame(right, text="预览（点击任务行查看）", padding=8)
        preview_outer.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        original_panel = ttk.LabelFrame(preview_outer, text="原图", padding=4)
        original_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.original_canvas = tk.Label(original_panel, background="#f0f0f0",
                                        text="—", foreground="#aaa")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        result_panel = ttk.LabelFrame(preview_outer, text="美化结果", padding=4)
        result_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.result_canvas = tk.Label(result_panel, background="#f0f0f0",
                                      text="—", foreground="#aaa")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

    def _add_entry(self, parent, key: str, label: str, default: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
        entry = ttk.Entry(row, width=10)
        entry.insert(0, default)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entries[key] = entry

    # ── Parsing helpers ───────────────────────────────────────────────────────

    def _parse_metrics(self) -> SimpleNamespace:
        def parse(name: str, label: str, allow_zero: bool = True) -> float:
            raw = self.entries[name].get().strip()
            if not raw:
                if allow_zero:
                    return 0
                raise ValueError(f"{label} 不能为空")
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"{label} 必须是数字") from exc
            if value != int(value):
                raise ValueError(f"{label} 必须是整数")
            if value < 0:
                raise ValueError(f"{label} 不能小于 0")
            return value

        exposure = parse("exposure", "新曝光数", allow_zero=False)
        if exposure <= 0:
            raise ValueError("新曝光数必须大于 0")
        return SimpleNamespace(
            exposure  = exposure,
            views     = parse("views",    "新观看数"),
            likes     = parse("likes",    "点赞数"),
            comments  = parse("comments", "评论数"),
            collects  = parse("collects", "收藏数"),
            shares    = parse("shares",   "分享数"),
        )

    def _make_args(self, path: Path, m: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(
            normal       = str(path),
            output       = None,
            exposure     = m.exposure,
            views        = m.views,
            likes        = m.likes,
            comments     = m.comments,
            collects     = m.collects,
            shares       = m.shares,
            ocr          = True,
            inspect      = False,
            glyph_atlas  = False,
            style_report = False,
        )

    # ── Task list management ──────────────────────────────────────────────────

    def on_add_files(self) -> None:
        try:
            metrics = self._parse_metrics()
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
            args = self._make_args(path, metrics)
            task = TaskItem(path=path, args=args)
            self._tasks.append(task)
            self.tree.insert(
                "", tk.END,
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
        total   = len(self._tasks)
        done    = sum(1 for t in self._tasks if t.status == _DONE)
        running = sum(1 for t in self._tasks if t.status == _RUNNING)
        failed  = sum(1 for t in self._tasks if t.status == _FAILED)
        waiting = sum(1 for t in self._tasks if t.status == _WAITING)
        if total == 0:
            self.global_status.configure(text="任务列表为空")
        elif running:
            self.global_status.configure(text=f"处理中… {done}/{total} 完成")
        elif done == total:
            self.global_status.configure(text=f"全部完成 ✓ ({done} 张)")
        else:
            parts = []
            if waiting: parts.append(f"等待 {waiting}")
            if done:    parts.append(f"完成 {done}")
            if failed:  parts.append(f"失败 {failed}")
            self.global_status.configure(text=" / ".join(parts))

    # ── Processing ────────────────────────────────────────────────────────────

    def on_start_all(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("提示", "已有任务在处理中，请等待完成。")
            return
        pending = [t for t in self._tasks if t.status == _WAITING]
        if not pending:
            messagebox.showinfo("提示", "没有等待中的任务。\n\n请先添加图片。")
            return
        self._stop_requested = False
        self._worker_thread = threading.Thread(
            target=self._run_all, args=(pending,), daemon=True
        )
        self._worker_thread.start()

    def _run_all(self, tasks: list[TaskItem]) -> None:
        for task in tasks:
            if self._stop_requested:
                break
            self.after(0, self._set_task_running, task)
            try:
                metrics = backend.calculate_metrics(
                    exposure=task.args.exposure,
                    views=task.args.views,
                    likes=task.args.likes,
                    comments=task.args.comments,
                    collects=task.args.collects,
                    shares=task.args.shares,
                )

                def _progress(msg: str, _task=task) -> None:
                    self.after(0, self._set_task_progress, _task, msg)

                # Pre-load the original for preview
                original = Image.open(task.args.normal).convert("RGB")
                task.original_image = original

                result = backend.beautify_normal_with_ocr(
                    task.args, metrics, on_progress=_progress
                )
                self.after(0, self._set_task_done, task, result)
            except Exception as exc:  # noqa: BLE001
                self.after(0, self._set_task_failed, task, exc)

        self.after(0, self._refresh_global_status)

    def _set_task_running(self, task: TaskItem) -> None:
        task.status   = _RUNNING
        task.progress = "启动…"
        self._update_row(task)
        self._refresh_global_status()

    def _set_task_progress(self, task: TaskItem, msg: str) -> None:
        task.progress = msg
        self._update_row(task)

    def _set_task_done(self, task: TaskItem, result: Image.Image) -> None:
        task.status       = _DONE
        task.progress     = "完成 ✓"
        task.result_image = result
        self._update_row(task)
        self._refresh_global_status()
        # Auto-select the just-finished task to show its preview
        iid = str(id(task))
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.see(iid)

    def _set_task_failed(self, task: TaskItem, exc: Exception) -> None:
        task.status   = _FAILED
        task.progress = str(exc)[:60]
        task.error    = str(exc)
        self._update_row(task)
        self._refresh_global_status()

    # ── Preview ───────────────────────────────────────────────────────────────

    def _on_task_selected(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        task = self._task_by_iid(sel[0])
        if task is None:
            return
        if task.original_image:
            self._show_preview(self.original_canvas, task.original_image, "_original_tk")
        else:
            self._clear_canvas(self.original_canvas, "_original_tk")
        if task.result_image:
            self._show_preview(self.result_canvas, task.result_image, "_result_tk")
        else:
            msg = task.progress if task.status == _RUNNING else "尚未生成"
            self._clear_canvas(self.result_canvas, "_result_tk", msg)

    def _show_preview(self, label: tk.Label, image: Image.Image, attr: str) -> None:
        photo, lw, lh = fit_preview(image, widget=label)
        setattr(self, attr, photo)
        label.configure(image=photo, width=lw, height=lh, text="")

    def _clear_canvas(self, label: tk.Label, attr: str, text: str = "—") -> None:
        setattr(self, attr, None)
        label.configure(image="", text=text, foreground="#aaa")

    def _clear_preview(self) -> None:
        self._clear_canvas(self.original_canvas, "_original_tk")
        self._clear_canvas(self.result_canvas,   "_result_tk")

    # ── Save ──────────────────────────────────────────────────────────────────

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
    app = DataPolisherApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
