# Sandevistan-Read

本地部署、资料严格溯源的 NotebookLM 类研究工作台。上传 PDF、EPUB、DOCX、PPTX、TXT、Markdown 或 HTML 后，可获得引用到页码/章节/幻灯片/段落的摘要与问答，并生成双人音频、单选 Quiz 和 Flashcards。

## 快速开始

要求：Git、Node.js 20+ 和 Corepack（推荐 Node.js 24 LTS）；Linux/macOS 还需要 curl。引导脚本使用 `uv` 自动准备 Python 3.11–3.13，并下载经过 SHA-256 校验的 FFmpeg 9.0 LGPL 静态构建与 LibreOffice 26.2.5，全部解压到项目内；不会调用系统包管理器或安装桌面组件。

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

Windows 11 x64 支持原生部署；ARM64 当前为实验支持。首次部署、Ollama/云端 Provider/TTS 配置、更新备份和故障处理详见 [Windows 11 原生部署手册](docs/windows-deployment.md)。

当前部署可在 `runtime/config.toml` 配置为监听 `0.0.0.0:20830`（需访问密钥）；本机浏览器打开 <http://127.0.0.1:20830>。停止服务运行 `scripts/stop.sh` 或 `scripts\stop.ps1`。

首次启动会从 `config.example.toml` 创建 `runtime/config.toml`。默认开发 Provider 是 `http://127.0.0.1:11434` 的 `qwen3.5:2b`，TTS 是 `http://127.0.0.1:20810`。Kimi、DeepSeek 或其它服务可在网页设置中添加为 OpenAI-compatible Provider。完整 API 文档位于 `/api/docs`。

## Provider 配置

设置页按角色限制可选协议：

| 角色 | 支持的 Provider 类型 |
| --- | --- |
| MAIN、VLM | Ollama、OpenAI-compatible |
| TTS | Sandevistan TTS、OpenAI TTS |

推荐流程是“连接并读取模型 → 选择或手填模型 → 验证并启用”。连接检查会读取服务实时公开的模型、设备和音色清单；若兼容服务不提供清单，可手填模型并执行深度验证。服务地址可直接粘贴带 `/v1`、`/api` 或 `/api/v1` 的常见形式，保存时会规范化为服务根地址。

MAIN/VLM 会同时管理模型的总上下文窗口、最大输入（服务提供时）和最大输出。Ollama 优先采用当前运行实例的 `context_length`，其次采用 Modelfile 的 `num_ctx`，不会把模型理论最大窗口直接当作运行窗口；OpenAI-compatible 服务则读取模型清单或详情中公开的限制。无法探测时使用 4096 tokens 安全回退并显示警告，最大输出默认按有效窗口的 25% 推导且不超过 4096。设置页可人工覆盖上下文窗口和最大输出；人工值优先，但不能超过模型明确报告的理论最大值。也可按 Provider 覆盖 temperature；留空时使用各任务默认值，运行记录会标明实际温度及来源。

问答、摘要、闪卡和播客脚本会预留 20% 安全余量，并按输入与输出预算裁剪证据。若 Provider 仍报告上下文溢出，系统会以 100%、50%、25% 三档自动缩小后重试；网络连接或超时错误则在同一预算下重试一次，不消耗溢出降档。发生裁剪、输出受限或回退时，聊天或 Studio 产物会显示“已按模型窗口调整证据量”。

连接检查不保存配置。深度验证只发送内置的极短测试文本或测试图片，不会发送 Notebook 资料；使用云端模型时仍可能产生少量 API 费用。未通过验证的配置可以保存为未启用状态，启用失败不会替换当前同角色的活跃 Provider。

程序化配置可调用 `POST /api/providers/inspect`，`mode="catalog"` 只读取实时清单，`mode="deep"` 会执行对应角色的最小真实调用；已保存的 MAIN/VLM 可调用 `POST /api/providers/{id}/probe` 重新探测并持久化窗口能力。请求和响应模型以 `/api/docs` 为准。

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

## Quiz 与 Flashcards

学习内容采用“知识蓝图 → 候选生成 → 证据与质量校验 → 去重”的流水线。系统综合模型参数量、有效上下文窗口和最大输出窗口自动选择 `full` 或 `lite` 档，也可在 Provider 配置中覆盖；`lite` 档不生成困难题。题卡只有通过逐项引用、证据一致性和结构质量检查后才会保留，数量不足时会明确返回部分产物，不使用模板填充凑数。

Quiz 按题作答，提交前不会通过产物 API 暴露答案、解释或引用；作答后才返回该题的答案、解释和原文证据，并支持提示、错题重练和整组重练。Flashcards 支持四档反馈（再来、困难、良好、简单），使用 FSRS 持久化安排到期复习，同时支持到期/全部/本轮队列、洗牌、移除和 CSV 导出。旧版 Quiz 尝试与闪卡复习记录会在数据库 v4 升级后保留，并在需要时参与新调度状态的恢复。

## Podcast V4

双人音频 V4 采用“多样化证据 → 主张表 → editorial acts → 携带前文的长段续写 → 客观门禁 → 整集审校 → 逐轮 TTS → 本地 ASR 验收”流水线。两位主持人共享解释、质疑、追问与综合职责，每一轮都要回应真实前文；事实、数字、案例和结论逐轮关联文档引用，纯寒暄或承上启下可以不显示引用。口播限定用自然口语表达，审稿术语与密集的防误读提醒不得进入口播；每个 Act 先立题目再展开。长节目使用约 2.8 轮/分钟的充实对话，并以紧凑模型输出减少结构 token；每个 Act 按剩余时长分配口播字数，欠账最多把下一幕提高到名义预算的 120%。只有输出上限截断时才允许全剧一次小型续写，且它与边界修复共享唯一恢复名额。客观门禁覆盖时长、引用、角色平衡、问句密度、重复轮次，以及审计套话与防误读句式的密度上限（整集、单族、单 Act 三档），失败时在整集审校前直接停止，不消耗额外模型调用；整集审校为推理模型预留输出余量，避免隐藏推理挤占答复导致截断。脚本未达到门槛时会在 TTS 前停止，不再用模板问答补足长度。音频完成后会先验证真实时长，再检查字/词错误率、双主持人分离和异常静音；异常静音会归属到具体轮次并并入重合成集合，只有少量坏轮次时最多重合成一次，GPU 因显存或设备问题失败时，TTS/ASR 各自只回退 CPU 一次，不修改 Provider 配置。

默认自动生成 12–25 分钟，也可选择 5、10、20 或 30 分钟；首次默认输出简体中文，并在浏览器中记住上次选择。MAIN Provider 的上下文和输出窗口较小时，系统使用更短场景与压缩剧集记忆；大窗口模型会生成更长场景并执行同一质量门槛。已有 V2 产物保持兼容。

可使用 `.venv/bin/python scripts/evaluate_podcast.py --notebook-id <ID> --minutes 5 --baseline-artifact <旧播客ID>` 生成 V4 文字稿并与旧产物盲评，默认不会调用 TTS。追加 `--reference-audio <样本.m4a>` 时，会通过当前本地 Sandevistan ASR 转写样本并加入匿名盲评；也可用 `--candidate-json <已通过的candidate.json>` 跳过 MAIN，复用同一候选做参考 ASR、盲评或 `--render-candidate` 端到端验收。`--tts-model`/`--tts-device` 可对单次渲染覆盖模型与设备（如 0.6B 快速预览），不写数据库、不修改已保存的 Provider。GPU 资源错误时自动单次转 CPU。评测结果保存在本地 `runtime/evals/`，不会进入 Git。

多样本冻结验收使用 `scripts/evaluate_podcast_suite.py prepare --manifest <runtime-manifest.json> --mode frozen`。它会为每个样本生成新候选、执行 TTS/ASR、按音频 ASR 生成匿名 A/B 包，并按音频 SHA 缓存参考 ASR；开发模式允许在 manifest 中提供已通过门禁的 `candidate_json`，避免因音频或评审问题重复调用 MAIN。`--sample <id>` 可重复指定以只跑 manifest 子集（选样集合参与套件身份哈希），`--tts-model/--tts-device` 同样支持单次覆盖；中断后 resume 会保留已生成的匿名映射，缺失时用不变的 seed 确定性重建。独立评审完成 `scores-template.json` 后，以 `finalize --run-dir <目录> --scores <评分.json>` 解盲并执行六维分数、音频质量及 45k MAIN token 硬门槛。

Sandevistan TTS Provider 可在设置中探测实时能力。自动模式选择已安装的最高质量模型：该模型支持 GPU 时优先 GPU，否则使用同一模型的 CPU；不会为了速度静默降低模型质量。高质量 CPU 合成通常比音频实时长度慢数倍，任务支持取消和项目内断点续跑。最终音频优先输出为经过响度统一的 AAC/M4A，旧版 WAV 产物保持兼容。

## 任务与 Notebook 管理

网页顶部的 `TASKS` 提供全局任务记录、搜索、类型/状态筛选、分页、阶段事件、真实计数、排队位置与本地历史 ETA。执行中的任务可安全终止；终止后删除任务只移除任务记录和任务专属临时文件，不会删除正式资料或已完成的 Studio 产物。ETA 少于 5 个同类本地样本时显示“学习中”，不会使用静态倍速冒充预测。

`NOTEBOOKS` 提供统一的资料库管理与空间统计，并支持勾选当前页中的多个 ACTIVE Notebook 批量删除。单项删除前必须输入完整名称，批量删除必须输入固定短语“批量删除”；系统会先取消关联任务，再清理文档、索引、对话、产物和登记的本地资源。清理操作写入 SQLite，可在服务重启后继续，失败时可在界面重试。
