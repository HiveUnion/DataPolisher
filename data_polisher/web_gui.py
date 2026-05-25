"""WebView desktop UI for DataPolisher."""

from __future__ import annotations

import base64
import json
import random
import sys
import threading
import traceback
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from PIL import Image

from . import cli as backend
from . import feed_eye


PREVIEW_MAX_WIDTH = 820
PREVIEW_MAX_HEIGHT = 980

_WAITING = "waiting"
_RUNNING = "running"
_DONE = "done"
_FAILED = "failed"

_IMAGE_FILE_TYPES = ("Images (*.jpg;*.jpeg;*.png)",)
_SAVE_FILE_TYPES = ("JPEG (*.jpg;*.jpeg)", "PNG (*.png)")


def _parse_range_pair(
    lo_raw: str,
    hi_raw: str,
    label: str,
    *,
    lower_floor: int = 0,
) -> tuple[int, int]:
    lo_s, hi_s = str(lo_raw).strip(), str(hi_raw).strip()
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
    s = str(raw).strip().replace("－", "-").replace("–", "-").replace("—", "-")
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


@dataclass
class WebTask:
    task_id: str
    path: Path
    mode: str
    result_image: Optional[Image.Image] = None
    original_image: Optional[Image.Image] = None
    output_path: Optional[Path] = None


class DataPolisherWebApi:
    def __init__(self) -> None:
        self.window = None
        self.webview = None
        self._tasks: Dict[str, WebTask] = {}
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._lock = threading.Lock()

    def bind(self, window, webview_module) -> None:
        self.window = window
        self.webview = webview_module

    def select_images(self) -> Dict[str, object]:
        if self.window is None or self.webview is None:
            return {"ok": False, "error": "窗口尚未就绪"}
        try:
            paths = self.window.create_file_dialog(
                self.webview.FileDialog.OPEN,
                allow_multiple=True,
                file_types=_IMAGE_FILE_TYPES,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"打开文件选择器失败：{exc}"}
        if not paths:
            return {"ok": True, "files": []}
        files = []
        for raw in paths:
            if not raw:
                continue
            path = Path(raw)
            if not path.is_file():
                continue
            item = {
                "path": str(path),
                "name": path.name,
                "preview": self._preview_for_path(path),
            }
            files.append(item)
        return {"ok": True, "files": files}

    def start_tasks(self, task_specs: List[Dict[str, object]]) -> Dict[str, object]:
        if self._worker_thread and self._worker_thread.is_alive():
            return {"ok": False, "error": "已有任务在处理中，请等待完成。"}
        try:
            prepared = [self._prepare_task_spec(spec) for spec in task_specs]
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        if not prepared:
            return {"ok": False, "error": "没有等待中的任务。"}

        self._stop_requested = False
        self._worker_thread = threading.Thread(
            target=self._run_tasks,
            args=(prepared,),
            daemon=True,
        )
        self._worker_thread.start()
        return {"ok": True}

    def remove_task(self, task_id: str) -> Dict[str, object]:
        with self._lock:
            self._tasks.pop(str(task_id), None)
        return {"ok": True}

    def clear_tasks(self) -> Dict[str, object]:
        self._stop_requested = True
        with self._lock:
            self._tasks.clear()
        return {"ok": True}

    def save_task(self, task_id: str) -> Dict[str, object]:
        task = self._tasks.get(str(task_id))
        if task is None or task.result_image is None:
            return {"ok": False, "error": "所选任务尚未完成，无结果可保存。"}
        if self.window is None or self.webview is None:
            return {"ok": False, "error": "窗口尚未就绪"}

        suggested = f"{task.path.stem}-polished.jpg"
        try:
            target = self.window.create_file_dialog(
                self.webview.FileDialog.SAVE,
                save_filename=suggested,
                file_types=_SAVE_FILE_TYPES,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"打开保存对话框失败：{exc}"}
        out = self._first_dialog_path(target)
        if out is None:
            return {"ok": True, "cancelled": True}

        try:
            out = self._save_image(task.result_image, out)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        task.output_path = out
        self._emit(
            {
                "type": "saved",
                "id": task.task_id,
                "progress": f"已保存 → {out.name}",
            }
        )
        return {"ok": True, "path": str(out)}

    def batch_save(self, task_ids: List[str]) -> Dict[str, object]:
        if self.window is None or self.webview is None:
            return {"ok": False, "error": "窗口尚未就绪"}
        try:
            folder_raw = self.window.create_file_dialog(self.webview.FileDialog.FOLDER)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"打开文件夹选择器失败：{exc}"}
        folder = self._first_dialog_path(folder_raw)
        if folder is None:
            return {"ok": True, "cancelled": True}

        saved = 0
        errors: List[str] = []
        for raw_id in task_ids:
            task = self._tasks.get(str(raw_id))
            if task is None or task.result_image is None:
                continue
            out = folder / f"{task.path.stem}-polished.jpg"
            try:
                out = self._save_image(task.result_image, out)
                task.output_path = out
                saved += 1
                self._emit(
                    {
                        "type": "saved",
                        "id": task.task_id,
                        "progress": f"已保存 → {out.name}",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task.path.name}: {exc}")

        return {
            "ok": not errors,
            "saved": saved,
            "folder": str(folder),
            "errors": errors,
            "error": "\n".join(errors) if errors else "",
        }

    def _prepare_task_spec(self, spec: Dict[str, object]) -> Dict[str, object]:
        task_id = str(spec.get("id") or "").strip()
        if not task_id:
            raise ValueError("任务缺少 ID")
        path = Path(str(spec.get("path") or ""))
        if not path.is_file():
            raise ValueError(f"图片不存在：{path}")
        mode = str(spec.get("mode") or "detail")
        config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
        if mode == "eye":
            self._validate_eye_config(config)
        elif mode == "detail":
            self._validate_detail_config(config)
        else:
            raise ValueError(f"未知模式：{mode}")
        with self._lock:
            existing = self._tasks.get(task_id)
            if existing is None:
                existing = WebTask(task_id=task_id, path=path, mode=mode)
                self._tasks[task_id] = existing
            existing.path = path
            existing.mode = mode
        return {"id": task_id, "path": path, "mode": mode, "config": config}

    def _validate_detail_config(self, config: Dict[str, object]) -> None:
        _parse_range_pair(
            str(config.get("exposureLo", "")),
            str(config.get("exposureHi", "")),
            "新曝光数",
            lower_floor=1,
        )
        _parse_range_pair(
            str(config.get("viewsLo", "")),
            str(config.get("viewsHi", "")),
            "新观看数",
            lower_floor=0,
        )

    def _validate_eye_config(self, config: Dict[str, object]) -> None:
        if not str(config.get("title", "")).strip():
            raise ValueError("标题关键词不能为空")
        _parse_range_pair(
            str(config.get("eyeViewsLo", "")),
            str(config.get("eyeViewsHi", "")),
            "浏览量范围",
            lower_floor=0,
        )

    def _run_tasks(self, prepared: List[Dict[str, object]]) -> None:
        for spec in prepared:
            if self._stop_requested:
                break
            task_id = str(spec["id"])
            path = Path(spec["path"])
            mode = str(spec["mode"])
            config = spec["config"]

            task = self._tasks.get(task_id)
            if task is None:
                continue

            self._emit({"type": "running", "id": task_id, "progress": "启动…"})
            try:

                def _progress(msg: str, _task_id=task_id) -> None:
                    self._emit({"type": "progress", "id": _task_id, "progress": msg})

                original = Image.open(path).convert("RGB")
                task.original_image = original

                if mode == "eye":
                    args = self._make_eye_args(path, config)
                    result = feed_eye.beautify_feed_card_eye(args, on_progress=_progress)
                else:
                    args, metrics = self._make_detail_args(path, config)
                    result = backend.beautify_normal_with_ocr(args, metrics, on_progress=_progress)

                task.result_image = result
                self._emit(
                    {
                        "type": "done",
                        "id": task_id,
                        "progress": "完成",
                        "originalPreview": self._image_to_data_url(original),
                        "resultPreview": self._image_to_data_url(result),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self._emit(
                    {
                        "type": "failed",
                        "id": task_id,
                        "progress": str(exc)[:80],
                        "error": str(exc),
                    }
                )

        self._emit({"type": "finished"})

    def _make_detail_args(self, path: Path, config: Dict[str, object]) -> Tuple[SimpleNamespace, Dict[str, object]]:
        exposure_range = _parse_range_pair(
            str(config.get("exposureLo", "")),
            str(config.get("exposureHi", "")),
            "新曝光数",
            lower_floor=1,
        )
        views_range = _parse_range_pair(
            str(config.get("viewsLo", "")),
            str(config.get("viewsHi", "")),
            "新观看数",
            lower_floor=0,
        )
        exposure = random.randint(*exposure_range)
        views = random.randint(*views_range)
        metrics = backend.calculate_metrics(
            exposure=exposure,
            views=views,
            likes=0,
            comments=0,
            collects=0,
            shares=0,
        )
        args = SimpleNamespace(
            normal=str(path),
            output=None,
            exposure_range=exposure_range,
            views_range=views_range,
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
        return args, metrics

    def _make_eye_args(self, path: Path, config: Dict[str, object]) -> SimpleNamespace:
        views_range = _parse_range_pair(
            str(config.get("eyeViewsLo", "")),
            str(config.get("eyeViewsHi", "")),
            "浏览量范围",
            lower_floor=0,
        )
        return SimpleNamespace(
            normal=str(path),
            output=None,
            eye_title=str(config.get("title", "")).strip(),
            eye_views_range=views_range,
            ocr=True,
            glyph_atlas=False,
            eye_mode=True,
        )

    def _preview_for_path(self, path: Path) -> Optional[str]:
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            return None
        return self._image_to_data_url(image)

    def _image_to_data_url(self, image: Image.Image) -> str:
        preview = image.copy()
        preview.thumbnail((PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.Resampling.LANCZOS)
        buf = BytesIO()
        preview.save(buf, format="JPEG", quality=86, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _save_image(self, image: Image.Image, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        ext = out.suffix.lower()
        if ext in (".jpg", ".jpeg", ""):
            if not out.suffix:
                out = out.with_suffix(".jpg")
            image.save(str(out), "JPEG", quality=92)
        else:
            image.save(str(out))
        return out

    def _first_dialog_path(self, value) -> Optional[Path]:
        if not value:
            return None
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        if not value:
            return None
        return Path(str(value))

    def _emit(self, payload: Dict[str, object]) -> None:
        if self.window is None:
            return
        script = (
            "window.DataPolisher && "
            f"window.DataPolisher.onNativeEvent({json.dumps(payload, ensure_ascii=False)});"
        )
        try:
            self.window.evaluate_js(script)
        except Exception:
            pass


def main() -> int:
    try:
        import webview  # type: ignore
    except ImportError:
        print(
            "pywebview is not installed. Run: pip install pywebview",
            file=sys.stderr,
        )
        return 1

    try:
        static_root = Path(__file__).resolve().parent / "static_web"
        index = static_root / "index.html"
        api = DataPolisherWebApi()
        window = webview.create_window(
            "DataPolisher",
            url=index.as_uri(),
            js_api=api,
            width=1220,
            height=820,
            min_size=(1040, 720),
        )
        api.bind(window, webview)
        webview.start(debug=False)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
