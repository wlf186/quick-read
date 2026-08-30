# Sandevistan-Read

本地部署、资料严格溯源的 NotebookLM 类研究工作台。上传 PDF、EPUB、DOCX、PPTX、TXT、Markdown 或 HTML 后，可获得引用到页码/章节/幻灯片/段落的摘要与问答，并生成双人音频、单选 Quiz 和 Flashcards。

## 快速开始

要求：Python 3.11–3.13、Node.js 20+、Corepack、Git、curl。引导脚本会下载经过 SHA-256 校验的 FFmpeg 9.0 LGPL 静态构建与 LibreOffice 26.2.5，并解压到项目内；不会调用系统包管理器或安装桌面组件。

Linux/macOS：

```bash
chmod +x scripts/*.sh
./scripts/bootstrap.sh
./scripts/start.sh
```

Windows 11 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
.\scripts\start.ps1
```

当前部署可在 `runtime/config.toml` 配置为监听 `0.0.0.0:20830`（需访问密钥）；本机浏览器打开 <http://127.0.0.1:20830>。停止服务运行 `scripts/stop.sh` 或 `scripts\stop.ps1`。

首次启动会从 `config.example.toml` 创建 `runtime/config.toml`。默认开发 Provider 是 `http://iollama:11434` 的 `qwen3.5:2b`，TTS 是 `http://localhost:20810`。Kimi、DeepSeek 或其它服务可在网页设置中添加为 OpenAI-compatible Provider。完整 API 文档位于 `/api/docs`。

## Provider 配置

设置页按角色限制可选协议：

| 角色 | 支持的 Provider 类型 |
| --- | --- |
| MAIN、VLM | Ollama、OpenAI-compatible |
| TTS | Sandevistan TTS、OpenAI TTS |

推荐流程是“连接并读取模型 → 选择或手填模型 → 验证并启用”。连接检查会读取服务实时公开的模型、设备和音色清单；若兼容服务不提供清单，可手填模型并执行深度验证。服务地址可直接粘贴带 `/v1`、`/api` 或 `/api/v1` 的常见形式，保存时会规范化为服务根地址。

连接检查不保存配置。深度验证只发送内置的极短测试文本或测试图片，不会发送 Notebook 资料；使用云端模型时仍可能产生少量 API 费用。未通过验证的配置可以保存为未启用状态，启用失败不会替换当前同角色的活跃 Provider。

程序化配置可调用 `POST /api/providers/inspect`，`mode="catalog"` 只读取实时清单，`mode="deep"` 会执行对应角色的最小真实调用。请求和响应模型以 `/api/docs` 为准。

## 本地数据边界

所有可删除的项目状态都在工作目录：

- `.venv/`：Python 环境
- `.tools/`：项目级工具
- `frontend/node_modules/`：前端依赖
- `runtime/data/`：SQLite、原始文档、渲染页、索引、音频、加密密钥
- `runtime/data/job-work/`：任务专属临时文件；删除任务时只清理这里的任务残留
- `runtime/data/backups/`：数据库结构升级前的一致性备份
- `runtime/models/`：项目内多语言向量模型（安装时下载一次，运行时离线）
- `runtime/cache/`、`runtime/logs/`、`runtime/tmp/`

卸载时停止服务并删除整个项目目录即可。注意：启用云端 LLM/VLM/TTS Provider 时，问题上下文或页面图片会发送给用户主动配置的服务；除此之外应用不调用外部业务服务。默认绑定 localhost；如果改为局域网地址，必须同时设置 `security.access_key`。

项目内媒体工具位于：

- `.tools/ffmpeg/bin/ffmpeg`（Windows 为 `ffmpeg.exe`）
- `.tools/ffmpeg/bin/ffprobe`
- `.tools/libreoffice/program/soffice`（Windows 为 `soffice.exe`）

原始安装包保留在 `runtime/cache/downloads/`，可安全清理；需要修复时再次运行 bootstrap 即可。LibreOffice 的无界面用户配置使用 `runtime/tmp/libreoffice-profiles/`，不会写入用户主目录。若只需离线开发，可在引导时设置 `SANDEVISTAN_SKIP_TOOLS=1` 跳过工具下载。

FFmpeg 使用 LGPL 静态构建，LibreOffice 使用 The Document Foundation 官方二进制包。第三方程序保留其各自许可证；本项目不修改或重新链接它们。

## Grounding 规则

检索范围固定为当前勾选资料的修订快照；回答要求逐项使用 `[S1]` 引用，服务端只返回真实检索片段对应的引用元数据。模型不可用时返回相关原文摘录，不使用未上传的外部知识。点击 UI 引用可查看文件名、定位与原文证据。

## Podcast V2

双人音频采用“全文证据地图 → 节目主题 → 分章对话 → 事实审校 → 逐轮 TTS”的本地流水线。默认自动生成 12–25 分钟，也可选择 5、10、20 或 30 分钟；首次默认输出简体中文，并在浏览器中记住上次选择。每一轮实质内容都关联真实文档引用，引用只显示在同步文字稿中，不会被朗读。

Sandevistan TTS Provider 可在设置中探测实时能力。自动模式选择已安装的最高质量模型：该模型支持 GPU 时优先 GPU，否则使用同一模型的 CPU；不会为了速度静默降低模型质量。高质量 CPU 合成通常比音频实时长度慢数倍，任务支持取消和项目内断点续跑。最终音频优先输出为经过响度统一的 AAC/M4A，旧版 WAV 产物保持兼容。

## 任务与 Notebook 管理

网页顶部的 `TASKS` 提供全局任务记录、搜索、类型/状态筛选、分页、阶段事件、真实计数、排队位置与本地历史 ETA。执行中的任务可安全终止；终止后删除任务只移除任务记录和任务专属临时文件，不会删除正式资料或已完成的 Studio 产物。ETA 少于 5 个同类本地样本时显示“学习中”，不会使用静态倍速冒充预测。

`NOTEBOOKS` 提供统一的资料库管理与空间统计。删除前必须输入完整名称；系统会先取消关联任务，再清理文档、索引、对话、产物和登记的本地资源。清理操作写入 SQLite，可在服务重启后继续，失败时可在界面重试。
