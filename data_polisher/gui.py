"""Desktop GUI for DataPolisher.

Wraps the OCR-based image beautify pipeline in a Tkinter window so non-developers
can pick a screenshot, enter new metric values, preview the result and export the
final image without touching the command line.
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace
from typing import Optional

from PIL import Image, ImageTk

from . import cli as backend


PREVIEW_MAX_WIDTH = 380
PREVIEW_MAX_HEIGHT = 700


def _display_scale(widget) -> float:
    """Return the HiDPI scale factor (e.g. 2.0 on Retina) for *widget*."""
    try:
        # winfo_fpixels('1i') gives physical pixels per inch.
        # 72 pt/in is Tk's logical baseline; dividing gives the scale factor.
        return max(1.0, widget.winfo_fpixels("1i") / 72.0)
    except Exception:
        return 1.0


def fit_preview(image: Image.Image, widget=None) -> tuple:
    """Return (photo, logical_w, logical_h) for a HiDPI-aware preview.

    On Retina / HiDPI displays the PhotoImage is rendered at the physical
    pixel count so it stays sharp, while the Label widget is sized to the
    logical (point) dimensions so the layout doesn't change.
    """
    scale = _display_scale(widget) if widget is not None else 1.0
    phys_max_w = int(PREVIEW_MAX_WIDTH * scale)
    phys_max_h = int(PREVIEW_MAX_HEIGHT * scale)

    width, height = image.size
    fit = min(phys_max_w / width, phys_max_h / height, 1.0)
    phys_w = max(1, int(width * fit))
    phys_h = max(1, int(height * fit))

    resized = image.resize((phys_w, phys_h), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(resized)

    # Logical size the Label should occupy (points, not physical pixels)
    logical_w = max(1, int(phys_w / scale))
    logical_h = max(1, int(phys_h / scale))
    return photo, logical_w, logical_h


class DataPolisherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DataPolisher · 数据美化工具")
        self.geometry("980x780")
        self.minsize(880, 700)

        self.normal_path: Optional[Path] = None
        self.original_image: Optional[Image.Image] = None
        self.result_image: Optional[Image.Image] = None
        self._original_tk: Optional[ImageTk.PhotoImage] = None
        self._result_tk: Optional[ImageTk.PhotoImage] = None
        self._busy = False

        self._build_layout()

    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self, padding=12)
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="选择截图…", command=self.on_pick_file).pack(side=tk.LEFT)
        self.path_label = ttk.Label(toolbar, text="尚未选择文件", foreground="#666")
        self.path_label.pack(side=tk.LEFT, padx=12)

        body = ttk.Frame(self, padding=(12, 0, 12, 12))
        body.pack(fill=tk.BOTH, expand=True)

        form = ttk.LabelFrame(body, text="新指标数据", padding=12)
        form.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        self.entries = {}
        self._add_entry(form, "exposure", "新曝光数", "1000")
        self._add_entry(form, "views", "新观看数", "300")
        ttk.Separator(form, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(8, 8))
        ttk.Label(form, text="互动数据（可选）", foreground="#666").pack(anchor=tk.W)
        self._add_entry(form, "likes", "点赞数", "0")
        self._add_entry(form, "comments", "评论数", "0")
        self._add_entry(form, "collects", "收藏数", "0")
        self._add_entry(form, "shares", "分享数", "0")

        button_row = ttk.Frame(form)
        button_row.pack(fill=tk.X, pady=(16, 0))
        self.generate_button = ttk.Button(button_row, text="生成美化图", command=self.on_generate)
        self.generate_button.pack(side=tk.LEFT)
        self.save_button = ttk.Button(button_row, text="另存为…", command=self.on_save, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(8, 0))

        self.status_label = ttk.Label(form, text="", foreground="#888", wraplength=220, justify=tk.LEFT)
        self.status_label.pack(anchor=tk.W, pady=(12, 0))

        preview = ttk.Frame(body)
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        original_panel = ttk.LabelFrame(preview, text="原图", padding=8)
        original_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.original_canvas = tk.Label(original_panel, background="#f3f3f3")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        result_panel = ttk.LabelFrame(preview, text="美化结果", padding=8)
        result_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self.result_canvas = tk.Label(result_panel, background="#f3f3f3")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

    def _add_entry(self, parent: ttk.LabelFrame, key: str, label: str, default: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
        entry = ttk.Entry(row, width=14)
        entry.insert(0, default)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entries[key] = entry

    def on_pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 normal 截图",
            filetypes=[("图片", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            self.original_image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("无法打开图片", str(exc))
            return
        self.normal_path = Path(path)
        self.path_label.configure(text=str(self.normal_path), foreground="#222")
        self._show_preview(self.original_canvas, self.original_image, "_original_tk")
        self.result_image = None
        self._set_canvas_blank(self.result_canvas, "_result_tk")
        self.save_button.configure(state=tk.DISABLED)
        self.set_status("已加载图片，可填写数据后点击生成。")

    def on_generate(self) -> None:
        if self._busy:
            return
        if not self.normal_path:
            messagebox.showwarning("缺少截图", "请先选择一张 normal 截图。")
            return

        try:
            args = self._build_args()
        except ValueError as exc:
            messagebox.showerror("输入有误", str(exc))
            return

        self._busy = True
        self.generate_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self.set_status("正在 OCR 并生成美化图，首次运行会下载 OCR 模型，请耐心等待…")

        thread = threading.Thread(target=self._run_generation, args=(args,), daemon=True)
        thread.start()

    def _build_args(self) -> SimpleNamespace:
        def parse(name: str, label: str, allow_zero: bool = True, integer: bool = True) -> float:
            raw = self.entries[name].get().strip()
            if not raw:
                if allow_zero:
                    return 0
                raise ValueError(f"{label} 不能为空")
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"{label} 必须是数字") from exc
            if integer and value != int(value):
                raise ValueError(f"{label} 必须是整数")
            if value < 0:
                raise ValueError(f"{label} 不能小于 0")
            return value

        exposure = parse("exposure", "新曝光数", allow_zero=False)
        if exposure <= 0:
            raise ValueError("新曝光数必须大于 0")
        views = parse("views", "新观看数")
        likes = parse("likes", "点赞数")
        comments = parse("comments", "评论数")
        collects = parse("collects", "收藏数")
        shares = parse("shares", "分享数")

        return SimpleNamespace(
            normal=str(self.normal_path),
            output=None,
            exposure=exposure,
            views=views,
            likes=likes,
            comments=comments,
            collects=collects,
            shares=shares,
            ocr=True,
            inspect=False,
            glyph_atlas=False,
            style_report=False,
        )

    def _run_generation(self, args: SimpleNamespace) -> None:
        try:
            metrics = backend.calculate_metrics(
                exposure=args.exposure,
                views=args.views,
                likes=args.likes,
                comments=args.comments,
                collects=args.collects,
                shares=args.shares,
            )
            image = backend.beautify_normal_with_ocr(args, metrics)
        except Exception as exc:  # noqa: BLE001 - surface every failure to user
            self.after(0, self._on_generation_error, exc)
            return
        self.after(0, self._on_generation_done, image)

    def _on_generation_error(self, exc: Exception) -> None:
        self._busy = False
        self.generate_button.configure(state=tk.NORMAL)
        self.set_status("")
        messagebox.showerror("生成失败", str(exc))

    def _on_generation_done(self, image: Image.Image) -> None:
        self._busy = False
        self.generate_button.configure(state=tk.NORMAL)
        self.result_image = image
        self._show_preview(self.result_canvas, image, "_result_tk")
        self.save_button.configure(state=tk.NORMAL)
        self.set_status("生成完成，可点击「另存为…」导出。")

    def on_save(self) -> None:
        if self.result_image is None:
            return
        suggested = "polished.jpg"
        if self.normal_path:
            suggested = f"{self.normal_path.stem}-polished.jpg"
        path = filedialog.asksaveasfilename(
            title="保存美化图",
            defaultextension=".jpg",
            initialfile=suggested,
            filetypes=[("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png")],
        )
        if not path:
            return
        try:
            extension = Path(path).suffix.lower()
            if extension in (".jpg", ".jpeg"):
                self.result_image.save(path, "JPEG", quality=92)
            else:
                self.result_image.save(path)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.set_status(f"已保存：{path}")

    def _show_preview(self, label: tk.Label, image: Image.Image, attr: str) -> None:
        photo, lw, lh = fit_preview(image, widget=label)
        setattr(self, attr, photo)
        label.configure(image=photo, width=lw, height=lh, text="")

    def _set_canvas_blank(self, label: tk.Label, attr: str) -> None:
        setattr(self, attr, None)
        label.configure(image="", text="尚未生成", foreground="#888")

    def set_status(self, text: str) -> None:
        self.status_label.configure(text=text)


def main() -> int:
    app = DataPolisherApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
