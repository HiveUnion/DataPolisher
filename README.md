# DataPolisher

DataPolisher 把 normal 截图里的数据指标按你输入的新值替换掉，使用本地 OCR 识别原文字位置、复用原图字符像素来组合新数字，输出视觉接近原图的截图。

## 截图与图标

* 仅支持当前样例风格的 normal 截图。
* OCR 自动识别曝光数 / 观看数 / 封面点击率 / 互动率 / 顶部小眼睛观看数。
* 新数字直接复用原图同行字符像素，因此粗细、颜色、锯齿、模糊与原图一致。
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
