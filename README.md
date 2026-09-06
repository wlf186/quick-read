# Sandevistan-Read

本地部署、资料严格溯源的 NotebookLM 类研究工作台。上传 PDF、EPUB、DOCX、PPTX、TXT、Markdown、HTML 或 PNG/JPEG/WebP 图片后，可获得引用到页码/章节/幻灯片/段落/原图的摘要与问答，并生成双人音频、单选 Quiz 和 Flashcards。

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

Windows 11 x64 支持原生部署；ARM64 当前为实验支持。首次部署、Ollama/云端 Provider/AUDIO 配置、更新备份和故障处理详见 [Windows 11 原生部署手册](docs/windows-deployment.md)。

当前部署可在 `runtime/config.toml` 配置为监听 `0.0.0.0:20830`（需访问密钥）；本机浏览器打开 <http://127.0.0.1:20830>。停止服务运行 `scripts/stop.sh` 或 `scripts\stop.ps1`。

首次启动会从 `config.example.toml` 创建 `runtime/config.toml`。默认开发 Provider 是 `http://127.0.0.1:11434` 的 `qwen3.5:2b`，AUDIO 服务是 `http://127.0.0.1:20810`。旧配置中的 `development.tts_url` 仍可读取，新配置使用 `development.audio_url`。Kimi、DeepSeek 或其它服务可在网页设置中添加为 OpenAI-compatible Provider。交互式 API 文档位于 `/api/docs`，机器可读定义位于 `/api/openapi.json`；认证接口另见 `/docs`。

## Provider 配置

设置页按 MAIN、VLM、AUDIO 三种职责分开管理。MAIN 是必需能力；VLM 和 AUDIO 可以临时暂停而不删除当前选择，暂停只影响之后创建的任务。Podcast 的 TTS、首次 ASR 验收及音频修复后的复验始终使用任务绑定的 AUDIO Provider，不随角色切换改变；绑定配置不存在时任务明确失败。缺少绑定信息的旧任务在开始执行时解析一次 AUDIO，并在本次执行中持续使用。图片处理默认按 `VLM → MAIN → 本地 RapidOCR` 依次兜底，任一步得到结果即停止；可调整参与步骤、全局关闭，或在每次上传确认时临时覆盖。图片策略仅应用于新上传，不会自动重建已有索引。云端 VLM/MAIN 参与视觉处理时，对应原图会发送给所选服务。

设置页按角色限制可选协议：

| 角色 | 支持的 Provider 类型 |
| --- | --- |
| MAIN、VLM | Ollama、OpenAI-compatible |
| AUDIO | Sandevistan Audio（TTS + ASR） |

推荐流程是“连接并读取模型 → 从完整清单选择模型 → 验证并启用”。模型搜索与当前选中值相互独立，重新展开仍会显示完整清单；仅当兼容服务不提供清单或目标模型未公开时才切换为手动模型 ID。修改地址、Key 或类型后需要重新读取清单；只修改模型或能力参数不会丢失已读取的清单，保存时会重新验证。服务地址可直接粘贴带 `/v1`、`/api` 或 `/api/v1` 的常见形式，保存时会规范化为服务根地址。

AUDIO 配置同时读取 TTS、ASR、设备、预置音色和可用声纹人员；只有 TTS 与 ASR 都满足 Podcast 验收要求时才能启用，深度验证会分别合成两位主持人的短句，再执行一次 TTS→ASR 闭环。主持人可使用不同的预置音色，或从 Audio Intel 声纹库中选择已有且具有可用样本的人员；系统锁定该人员最新的可用样本，不提供手工声纹 ID 或临时上传入口，两位主持人不能选择相同音色或人员。预置音色支持模型公开的基础表达指令，并会在整集追加稳定语速、音高和情绪范围约束；声纹克隆由固定参考样本控制，不发送模型不支持的表达指令。升级前保存的 Sandevistan TTS 会自动迁移为 AUDIO；OpenAI TTS 会保留为停用的 TTS-only 记录，但不能再用于 Podcast。

MAIN/VLM 会同时管理模型的总上下文窗口、最大输入（服务提供时）和最大输出。Ollama 优先采用当前运行实例的 `context_length`，其次采用 Modelfile 的 `num_ctx`，不会把模型理论最大窗口直接当作运行窗口；OpenAI-compatible 服务则读取模型清单或详情中公开的限制。无法探测时使用 4096 tokens 安全回退并显示警告，最大输出默认按有效窗口的 25% 推导且不超过 4096。设置页可人工覆盖上下文窗口和最大输出；人工值优先，但不能超过模型明确报告的理论最大值。也可按 Provider 覆盖 temperature；留空时使用各任务默认值，运行记录会标明实际温度及来源。

问答、摘要、闪卡和播客脚本会预留 20% 安全余量，并按输入与输出预算裁剪证据。若 Provider 仍报告上下文溢出，系统会以 100%、50%、25% 三档自动缩小后重试；网络连接或超时错误则在同一预算下重试一次，不消耗溢出降档。发生裁剪、输出受限或回退时，聊天或 Studio 产物会显示“已按模型窗口调整证据量”。

连接检查不保存配置。深度验证只发送内置的极短测试文本或测试图片，不会发送 Notebook 资料；使用云端模型时仍可能产生少量 API 费用。未通过验证的配置可以保存为未启用状态，启用失败不会替换当前同角色的活跃 Provider。

程序化配置可调用 `POST /api/providers/inspect`，`mode="catalog"` 只读取实时清单，`mode="deep"` 会执行对应角色的最小真实调用；AUDIO catalog 还会返回脱敏的 `voiceprint_library` 和可持久化的 `resolved_audio_config`。角色状态分别通过 `GET /api/provider-roles` 与 `PATCH /api/provider-roles/{role}` 读取和更新，全局图片策略使用 `GET/PUT /api/settings/image-processing`；上传接口可在 multipart `image_policy` 字段中传入单次覆盖。已保存的 MAIN/VLM 可调用 `POST /api/providers/{id}/probe` 重新探测并持久化窗口能力。请求字段及关键响应说明见 `/api/docs`。Provider 配置保留扩展字段；新增布尔开关必须传 JSON 布尔值，不能用字符串或数字代替。AUDIO 的 `config.podcast_sequence_tts` 缺省按开启处理，设为 `false` 会关闭批量合成及脚本/TTS 重叠；该设置保存于 Provider 配置，不属于 `config.toml`。能力清单的 `sequence_jobs`、模型的 `default/checkpoints` 和 `recommended.reason` 可用于解释当前选择。

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

卸载时停止服务并删除整个项目目录即可。注意：启用云端 MAIN/VLM/AUDIO Provider 时，问题上下文、资料片段、页面图片或 Podcast 文本会发送给用户主动配置的服务。声纹克隆时 quick-read 只读取人员与样本元数据并向该 AUDIO 服务提交所选样本 ID，不上传临时参考音频；声纹原始文件由 Audio Intel 自己管理。除此之外应用不调用外部业务服务。默认绑定 localhost；如果改为局域网地址，必须同时设置 `security.access_key`。

项目内媒体工具位于：

- `.tools/ffmpeg/bin/ffmpeg`（Windows 为 `ffmpeg.exe`）
- `.tools/ffmpeg/bin/ffprobe`
- `.tools/libreoffice/program/soffice`（Windows 为 `soffice.exe`）

原始安装包保留在 `runtime/cache/downloads/`，可安全清理。FFmpeg 由 `scripts/tools.lock.json` 固定 Release tag 与资产名，下载时从该 Release 读取并校验 SHA-256；详见 [第三方工具说明](THIRD_PARTY_NOTICES.md)。bootstrap 会复用已存在的媒体工具；需要替换 FFmpeg 构建时，先停止服务并将旧 `.tools/ffmpeg` 目录移到项目内其他位置留作回退，再运行 bootstrap。LibreOffice 的无界面用户配置使用 `runtime/tmp/libreoffice-profiles/`，不会写入用户主目录。若只需离线开发，可在引导时设置 `SANDEVISTAN_SKIP_TOOLS=1` 跳过工具下载。

FFmpeg 使用 LGPL 静态构建，LibreOffice 使用 The Document Foundation 官方二进制包。第三方程序保留其各自许可证；本项目不修改或重新链接它们。

## Grounding 规则

检索范围固定为当前勾选资料的修订快照；回答要求逐项使用 `[S1]` 引用，服务端只返回真实检索片段对应的引用元数据。模型不可用时返回相关原文摘录，不使用未上传的外部知识。点击 UI 引用可查看文件名、定位与原文证据。

切换 Notebook 或开启新对话会重置当前聊天显示与草稿；目标 Notebook 的资料和历史载入完成后才能发送问题。切换前已提交的问题仍由服务端处理并保存在原会话，其迟到的回答或错误不会覆盖新会话。

## Quiz 与 Flashcards

学习内容采用“知识蓝图 → 候选生成 → 证据与质量校验 → 去重”的流水线。系统综合模型参数量、有效上下文窗口和最大输出窗口自动选择 `full` 或 `lite` 档，也可在 Provider 配置中覆盖；`lite` 档不生成困难题。题卡只有通过逐项引用、证据一致性和结构质量检查后才会保留，数量不足时会明确返回部分产物，不使用模板填充凑数。

Quiz 按题作答，提交前不会通过产物 API 暴露答案、解释或引用；作答后才返回该题的答案、解释和原文证据，并支持提示、错题重练和整组重练。Flashcards 支持四档反馈（再来、困难、良好、简单），使用 FSRS 持久化安排到期复习，同时支持到期/全部/本轮队列、洗牌、移除和 CSV 导出。旧版 Quiz 尝试与闪卡复习记录会在数据库 v4 升级后保留，并在需要时参与新调度状态的恢复。

闪卡“移除”会停用该卡片，保留原始题卡、FSRS 状态与历史评分，但从所有学习队列（包括本轮队列与洗牌）中隐藏，并同步当前进度。没有待学习卡片时本轮完成；旧活动会话在恢复时自动校正。对已移除卡片提交评分会返回 409，且不会改写复习记录。

## Podcast V4

双人音频 V4 采用“多样化证据 → 主张表 → editorial acts → 携带前文的长段续写 → 客观门禁 → 整集审校 → 批量 TTS → 本地 ASR 验收”流水线。两位主持人共享解释、质疑、追问与综合职责，每一轮都要回应真实前文；事实、数字、案例和结论逐轮关联文档引用，纯寒暄或承上启下可以不显示引用。口播限定用自然口语表达，审稿术语与密集的防误读提醒不得进入口播；每个 Act 先立题目再展开。长节目使用约 2.8 轮/分钟的充实对话，并以紧凑模型输出减少结构 token；每个 Act 按剩余时长分配口播字数，欠账最多把下一幕提高到名义预算的 120%。输出上限截断时的小型续写、时长不足时的受引用约束扩写、超过目标时长 120% 时的压缩，共享全剧唯一恢复名额；当前不执行整集边界修复。客观门禁覆盖时长、引用、角色平衡、问句密度、重复轮次，以及审计套话与防误读句式的密度上限（整集、单族、单 Act 三档），失败时跳过整集审校；整集审校为推理模型预留输出余量，避免隐藏推理挤占答复导致截断。脚本未达到门槛时任务失败，不发布成品，也不使用模板问答补足长度。关闭脚本/TTS 重叠时，合成在脚本验收后开始；开启重叠时，已完成 Act 可能已调用 AUDIO 并消耗部分合成资源，即使整集最终未通过。音频完成后会先验证真实时长，再检查字/词错误率、双主持人分离和异常静音；异常静音会归属到具体轮次并并入重合成集合，只有少量坏轮次时最多重合成一次，GPU 因显存或设备问题失败且允许回退时，每次 TTS/ASR 调用最多回退同模型 CPU 一次，不修改 Provider 配置。

支持 Audio Intel v1 批量协议时，Podcast 会在一次模型加载中按输入顺序合成多轮并保留逐轮 WAV。首次生成脚本时，若 MAIN 地址不是代码识别的本机/loopback 地址、且 MAIN 与 AUDIO 的 URL hostname 不同，已完成 Act 的合成可与后续脚本生成重叠。这个判定基于地址，不探测实际 GPU 资源；共享算力但使用不同 hostname 的部署可在设置中关闭加速。最终脚本按文本、语言、主持人、音色模式、指令、模型、固定 checkpoint revision、设备和协议版本的精确哈希复核，扩写或配置变化只重做受影响片段。旧服务或能力缺失时使用逐轮合成；可处理的批量 Provider 错误会回退逐轮，取消或未恢复的异常仍会使任务失败。最终 FFmpeg 与全片 ASR 门禁不变。

默认自动生成 12–25 分钟，也可选择 5、10、20 或 30 分钟；首次默认输出简体中文，并在浏览器中记住上次选择。MAIN Provider 的上下文和输出窗口较小时，系统使用更短场景与压缩剧集记忆；大窗口模型会生成更长场景并执行同一质量门槛。已有 V2 产物保持兼容。

评测可用 `.venv/bin/python scripts/evaluate_podcast.py --notebook-id <ID> --minutes 5` 生成文字稿，默认不调用 TTS；复用 `--candidate-json` 并添加 `--render-candidate` 可跳过 MAIN、执行音频验收。评测渲染的 `--tts-mode` 默认为 `single`，显式选择 `sequence` 才使用批量模式；不支持或批量失败时会报告失败，以免将逐轮回退误当作批量评测结果。单次模型/设备覆盖、参考音频盲评和多样本冻结验收见 [Podcast 评测与 TTS 资格维护](docs/podcast-evaluation.md)。评测结果保存在本地 `runtime/evals/`，不会进入 Git。

Sandevistan Audio Provider 可在设置中分别配置 TTS 合成、两位主持人的预置/声纹音色与 ASR 验收。TTS 自动模式优先采用服务默认且符合固定 checkpoint revision 与设备资格表的组合，否则按已安装模型的质量排序回退；回退排序主要参考模型参数量，不代表所有备选都经过双基线资格评测。已有 `auto_select=false` 的人工选择保持不变。自定义表达指令时，若存在支持指令的已安装备选，会优先保留指令能力；不存在此类备选时仍返回原推荐；ASR 自动模式采用服务推荐的模型与设备，两者分别控制 GPU 失败时是否回退同模型 CPU，不会为了速度静默降低模型质量。脚本提示词限制突发愤怒、高亢和夸张标点，TTS 对支持表达指令的预置音色使用整集固定的稳定约束；声纹模式始终使用配置时解析的固定样本。整集仅因时长超过 120% 时可使用一次受引用约束的压缩恢复，未变化轮次继续复用已经并行生成的音频。双说话人分离、时间对齐及 CER/WER 等验收门槛保持固定。高质量 CPU 合成通常比音频实时长度慢数倍，任务支持取消和项目内断点续跑。最终音频优先输出为经过响度统一的 AAC/M4A，旧版 WAV 产物保持兼容。

批量合成默认开启，可在 AUDIO 设置中关闭并保存；开关本身不改变模型、精度或最终文本顺序。通过 `GET /api/artifacts/{id}` 返回的 `payload.performance` 可查看脚本、TTS、总耗时与重叠时长，`payload.provider` 记录批量大小、复用轮次和降级原因。`serial_estimate_seconds` 与 `overlap_gain_ratio` 是本次运行推算值，并非另跑一遍串行流程的实测收益；旧产物可能没有这些字段。

TTS 服务默认组合的资格表绑定 checkpoint revision 与计算设备；GPU 结论不授权 CPU，revision 改变时该组合失去服务默认优先资格。当前 `scripts/qualify_podcast_tts.py` 固定比较 0.6B/GPU 批量候选与 0.6B/GPU、1.7B/CPU 两组逐轮基线；`prepare`、`auto-finalize` 和可选人工 `finalize` 只生成本地评测报告，不修改 Provider 配置或资格表。维护者在核对固定 revision、实际设备和通过的客观/双基线声学报告后，另行更新代码中的资格表。自动声学指标属于代理指标，人工盲听可作为补充。

## 任务与 Notebook 管理

网页顶部的 `TASKS` 提供全局任务记录、搜索、类型/状态筛选、分页、阶段事件、真实计数、排队位置与本地历史 ETA。排队中的任务可立即取消，执行中的任务可安全终止；终止后删除任务只移除任务记录和任务专属临时文件，不会删除正式资料或已完成的 Studio 产物。ETA 少于 5 个同类本地样本时显示“学习中”，不会使用静态倍速冒充预测。

解析任务取消后，尚未完成的资料会显示“解析已取消”（API 状态为 `failed`），并取消勾选；已经完成索引的资料保持可用。服务启动时会校正旧版本遗留的排队状态，仅处理最新解析任务已取消且没有其他活动解析任务的资料，不重新解析或删除原文件。

`NOTEBOOKS` 提供统一的资料库管理与空间统计，并支持勾选当前页中的多个 ACTIVE Notebook 批量删除。单项删除前必须输入完整名称，批量删除必须输入固定短语“批量删除”；系统会先取消关联任务，再清理文档、索引、对话、产物和登记的本地资源。清理操作写入 SQLite，可在服务重启后继续，失败时可在界面重试。
