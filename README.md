# DataPolisher

DataPolisher 把 normal 截图里的数据指标按你输入的新值替换掉，使用本地 OCR 识别原文字位置、复用原图字符像素来组合新数字，输出视觉接近原图的截图。

## 截图与图标

* 仅支持当前样例风格的 normal 截图。
* OCR 自动识别曝光数 / 观看数 / 封面点击率 / 互动率 / 顶部小眼睛观看数。
* 新数字优先复用原图同行字符像素；缺字或需要稳定统一字形时，会回退到内置字体校准渲染。
* 不上传任何图片到云端，全部本地处理。

## 计算口径

* 封面点击率 = `观看数 / 曝光数`
* 互动率 = `(点赞数 + 评论数 + 收藏数 + 分享数) / 观看数`

## 本地开发

```bash
cd /Users/MVEN/iMoney/Tutor/DataPolisher

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# 启动 GUI
PYTHONPATH=. python -m data_polisher

# 或者 CLI
PYTHONPATH=. python -m data_polisher.cli \
  --normal /path/to/normal.jpg \
  --output /path/to/output.jpg \
  --exposure 1000 --views 300 \
  --likes 20 --comments 3 --collects 5 --shares 2 \
  --ocr
```

第一次运行会下载 PaddleOCR 模型到 `~/.paddlex/`，之后会复用缓存。

## 字体、定位与替换逻辑

### 字体使用位置

项目内置字体在 `data_polisher/static/fonts/`：

* `REDNumber-Bold.otf`：详细数据区的大号指标数字，如曝光数、观看数、点击率、互动率。用于替代原先的截图字形，保证详情区数字统一、接近小红书数据页样式。
* `REDNumber-Regular.ttf` / `REDNumber-Medium.ttf`：顶部小眼睛观看数的候选字体。顶部栏数字比详情区小、更轻，逻辑会优先用 Regular / Medium，而不是 Bold。
* `DIN-OT-Medium.ttf`：百分号 `%` 单独用 DIN 渲染，再和 RED Number 数字拼接，避免 RED Number 没有合适百分号或百分号过粗。
* `jlm_cmss10.ttf`：信息流封面左下角“小眼睛”白色浏览数字。按样本 `40` 逐字体比对后，它的窄身轻字重比 DIN / RED Number 更贴近原图。

字体选择入口主要在 `data_polisher/cli.py`：

* `BODY_NATIVE_FONT_PATH`：详细数据区主字体，优先 `REDNumber-Bold.otf`。
* `build_header_views_forced_font()`：顶部小眼睛观看数，优先 Regular / Medium，并带 `HEADER_VIEWS_FONT_ADJUST = -2` 的小号修正。
* `red_number_forced_font_for_standalone_patch(..., overlay_views_ink=True)`：信息流封面小眼睛浏览数，优先 `jlm_cmss10.ttf`，并使用按样张微调过的字号、位移和 alpha。
* `choose_feed_overlay_font_size()`：信息流小眼睛专用字号选择；对已校准的 `jlm_cmss10.ttf` 固定使用视觉确认后的字号。

### 详细数据 Tab 如何定位

详细数据 Tab 使用 OCR 文本标签定位，主流程在 `beautify_normal_with_ocr()`：

1. OCR 识别整张图，得到 `{text, rect}` 列表。
2. 用 `find_value_below_label()` 找中文标签下方的数值：
   `曝光数 / 观看数 / 封面点击率 / 互动率`。
3. 候选数值必须在标签下方约 80px 内，且水平中心不能离标签太远。
4. 顶部小眼睛观看数单独用 `find_header_view_value()` 找顶部统计行，从左到右取第一个数字框，并排除日期样式 OCR，如 `05-07`。

### 小眼睛 Tab 如何定位

小眼睛 Tab 只改信息流封面左下角的小眼睛浏览数字，主流程在 `data_polisher/feed_eye.py`：

1. 用用户输入的标题关键词，通过 `pick_best_title_item()` 在 OCR 文本中模糊匹配标题。
2. 根据标题位置推断所属左右列和封面缩略图区域：`_infer_thumbnail_rect()`。
3. 在封面底部左侧推断半透明胶囊条带：`_overlay_strip_roi()`。
4. 用 `_pick_overlay_item()` 优先挑条带内纯数字 OCR 框；如果 OCR 把数字和旁边文字合成宽框，也允许短高度、合理宽度的合并框。
5. 如果全图 OCR 没找到，会对条带裁图做二次 OCR。
6. 真正渲染前再用 `localize_feed_overlay_views_ink()` 找亮色数字墨迹框。它不是找黑字，而是按列统计高亮像素相对背景的亮度差，适合白字叠在半透明胶囊上的场景。

### 替换怎么做

所有数值最终都进入 `patch_ocr_rect_with_glyphs()`：

1. 先确定实际墨迹框：
   * 普通黑字用 `get_ink_rect()`。
   * 信息流封面白字用 `localize_feed_overlay_views_ink()`。
2. 擦除旧字：
   * 普通指标用边缘平均色填充。
   * 信息流小眼睛先遮掉旧数字亮色笔画，再用同一胶囊里的干净灰色材质羽化补回，避免把眼睛图标、耳机线或胶囊外背景复制进数字区域。
3. 渲染新字：
   * 如果有可用 row atlas，优先用原图同行字符像素拼新数字。
   * 详情区默认强制 RED Number Bold，并用原图墨迹高度校准字号。
   * 顶部小眼睛用更轻的小号 RED Number。
   * 信息流封面小眼睛用 `jlm_cmss10.ttf`，并走专用的细 alpha 白字渲染，避免遮挡小眼睛图标。
4. 最后按原文字颜色、alpha 分布、边缘风格做 mask 匹配，让新字的锯齿和透明度尽量贴近原图。

### 范围输入与位数限制

小眼睛 Tab 的范围输入不会在 GUI 层提前随机。逻辑会先定位原图数字，再根据原数字位数收窄范围：

* 原图是 `40` 这种两位数时，输入 `80-120` 会自动收窄为 `80-99`，避免把三位数塞进两位数胶囊里。
* 如果 OCR 把 `40` 漏识别成 `4`，会用亮字墨迹框宽高推断位数，尽量仍按两位数处理。
* 相关逻辑在 `choose_feed_overlay_views_for_slots()` 和 `clamp_feed_overlay_views_to_digit_slots()`。

## 单元测试

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

## 打包成桌面软件

### 在当前操作系统上构建

```bash
pip install -r requirements-dev.txt
python build_app.py
```

* macOS：产出 `dist/DataPolisher.app`
* Windows：产出 `dist/DataPolisher/DataPolisher.exe`

> PaddleOCR / paddlepaddle 体积较大，构建后的安装包通常 800MB 以上，属于正常现象。

### 通过 GitHub Actions 一次构建 Mac + Windows

仓库已经带 `.github/workflows/build.yml`：

1. 把项目推到 GitHub。
2. 推一个 `vX.Y.Z` 标签或在 Actions 页面手动 `workflow_dispatch`。
3. 等待 `macos-latest` 和 `windows-latest` 两个 job 完成。
4. 在 workflow run 的 Artifacts 区下载 `DataPolisher-macOS` 与 `DataPolisher-Windows`。

### 在 macOS 上手工生成 Windows 版

PyInstaller 不支持跨平台打包。如果不希望走 GitHub Actions，可以在一台 Windows 机器上：

```powershell
git clone <你的仓库>
cd DataPolisher
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python build_app.py
```

## 目录结构

```
DataPolisher/
├── data_polisher/        # 业务代码
│   ├── core.py           # 计算 + 像素特征
│   ├── ocr.py            # PaddleOCR 适配
│   ├── template.py       # 模板坐标兜底
│   ├── cli.py            # 命令行 + 自校准 + row atlas 主流程
│   ├── gui.py            # Tkinter GUI
│   └── __main__.py       # python -m data_polisher 入口
├── tests/test_core.py    # 单元测试
├── requirements.txt      # 运行依赖
├── requirements-dev.txt  # 打包/测试依赖
├── pyproject.toml        # 包元数据
├── build_app.py          # 跨平台 PyInstaller 构建脚本
└── .github/workflows/build.yml
```

## 已知限制

* 截图样式必须接近 normal 模板，OCR 通过中文标签 `曝光数 / 观看数 / 封面点击率 / 互动率` 锚定位置。
* 缺少同行原始字符的稀有数字（例如原图里没有出现 `7`）会触发 row atlas 失败，此时 cli 会回退到字体校准模式。
* 应用只用于内部素材排版校正，请勿用于伪造对外公示数据。
