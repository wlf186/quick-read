# Windows 11 原生部署手册

本文面向从 GitHub 拉取源码、直接在 Windows 11 上自主部署的用户，不要求 WSL、Docker 或预装 Python。项目脚本会在仓库内创建隔离环境，并将运行数据、模型和媒体工具保存在项目目录中。

## 支持范围与资源准备

| 项目 | 建议/状态 |
| --- | --- |
| 系统 | Windows 11 x64（正式支持）；Windows 11 ARM64（实验性，部分组件依赖 x64 模拟） |
| 终端 | Windows PowerShell 5.1 或 PowerShell 7 |
| 必需软件 | Git、Node.js 20+、Corepack；推荐 Node.js 24 LTS |
| Python | 无需单独安装；`uv` 会按项目要求安装 Python 3.11–3.13 |
| 磁盘 | 应用和本地模型建议预留 4–6 GB；再使用 Ollama 时建议至少预留 12 GB |
| 网络 | 首次安装需访问 GitHub、Python/Node 包源和 Hugging Face 模型源 |

建议使用本机 NTFS 磁盘上的短路径，例如 `C:\ai\quick-read`。不要放在 OneDrive 同步目录、网络共享、中文层级很深的目录或权限受控的系统目录中。

## 1. 安装前检查

安装 Git 和 [Node.js 24 LTS](https://nodejs.org/) 后，重新打开 PowerShell，执行：

```powershell
git --version
node --version
corepack --version
```

Node.js 版本应为 20 或更高。如果提示找不到 `corepack`，执行以下命令，然后重新打开 PowerShell：

```powershell
npm install -g corepack
```

企业代理环境可在当前 PowerShell 会话设置代理；请替换为真实地址：

```powershell
$env:HTTPS_PROXY = "http://proxy.example.com:8080"
$env:HTTP_PROXY = $env:HTTPS_PROXY
```

## 2. 克隆并安装

```powershell
git clone https://github.com/wlf186/quick-read.git C:\ai\quick-read
Set-Location C:\ai\quick-read
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
```

`Process` 范围的执行策略只对当前 PowerShell 窗口有效，不会永久修改系统策略。引导脚本会依次：

1. 下载项目内的 `uv` 并创建 `.venv`；
2. 安装锁定的 Python 依赖和本地向量模型；
3. 下载并校验 FFmpeg、LibreOffice，解压到 `.tools`；
4. 安装前端依赖并构建生产页面。

下载中断后可以直接再次运行 `bootstrap.ps1`。已验证的缓存和已安装组件会复用。FFmpeg 的摘要来自对应 GitHub Release 资产，LibreOffice 使用仓库锁定的 SHA-256；不要为了绕过报错而关闭摘要校验。

如果当前只做后端开发、暂时不需要 Office 文档可视化和播客音频规范化，可降级安装：

```powershell
$env:SANDEVISTAN_SKIP_TOOLS = "1"
.\scripts\bootstrap.ps1
Remove-Item Env:SANDEVISTAN_SKIP_TOOLS
```

跳过媒体工具会影响 DOCX/PPTX 等 Office 文档的页面预览，以及最终 AAC/M4A 音频的转码和响度统一，不建议用于完整部署。

## 3. 启动与验证

```powershell
.\scripts\start.ps1
```

脚本会等待服务健康检查通过后才报告成功。浏览器打开 <http://127.0.0.1:20830>，也可以在 PowerShell 验证：

```powershell
Invoke-RestMethod http://127.0.0.1:20830/auth/status
Invoke-RestMethod http://127.0.0.1:20830/api/status
```

第二个接口应返回应用、媒体工具及 Provider 状态。首次启动会从 `config.example.toml` 生成未纳入 Git 的 `runtime/config.toml`。

停止服务：

```powershell
.\scripts\stop.ps1
```

启停日志位于：

- `runtime\logs\server.log`
- `runtime\logs\server-error.log`

## 4. 配置模型 Provider

应用本身可以在没有生成式模型的情况下导入、索引和检索文字资料，并在模型不可用时返回相关原文摘录。完整问答、Studio 内容和播客脚本需要 MAIN；图片默认按 VLM、具备视觉能力的 MAIN、本地 RapidOCR 依次尝试；Podcast 音频需要同时提供 TTS 与 ASR 的 AUDIO。MAIN 是必需角色，VLM 与 AUDIO 可以暂停而不删除已选 Provider，暂停只影响之后创建的任务。

### 4.1 本机 Ollama（最便捷的全本地方案）

从 [Ollama Windows 官方页面](https://docs.ollama.com/windows) 安装并启动 Ollama，然后执行：

```powershell
ollama pull qwen3.5:2b
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

`qwen3.5:2b` 下载约 2.7 GB，是项目默认的轻量多模态模型；内存和显存充足时可以自行选择更大的模型。

在网页 `SETTINGS / Providers` 中：

1. 编辑 `Local Ollama`（MAIN），地址填 `http://127.0.0.1:11434`；
2. 点击“连接并读取模型”，选择模型，再执行“深度验证”并启用；
3. 需要优先使用专用视觉模型时，以同样方式编辑 `Local Vision`（VLM），选择支持视觉的模型；也可以暂停 VLM，让图片策略继续尝试 MAIN 或本地 OCR。

设置页的 `IMAGE PIPELINE` 可调整 VLM、MAIN、OCR 的参与顺序或全局关闭图片处理，每次上传确认时还可临时覆盖。策略只影响新上传资料，不会自动重建已有索引；云端模型参与时原图会发送给对应服务。

设置页会分别显示模型理论最大窗口、当前运行窗口和最大输出。对 Ollama，应用优先采用 `/api/ps` 返回的当前 `context_length`，其次采用 Modelfile 的 `num_ctx`，不会自动把理论最大窗口强制写入运行配置。需要显式调整时，可填写“上下文窗口覆盖”和“最大输出覆盖”；应用调用 Ollama 时会把有效运行窗口作为 `num_ctx` 发送。

如果某个旧部署仍显示 `http://iollama:11434`，这是旧配置数据库中已经保存的地址；在设置页改为 `http://127.0.0.1:11434` 即可。

### 4.2 OpenAI-compatible 云端服务

在 Provider 设置中新增 MAIN，类型选择 `OpenAI-compatible`，填写服务地址、API Key 和模型。地址可以带常见的 `/v1`、`/api` 或 `/api/v1`，保存时会规范化。先“连接并读取模型”，在可搜索的完整清单中选择模型；只有服务不公开清单时才切换为手动模型 ID，再执行“深度验证”。视觉模型可以另建 VLM Provider，也可以由具备视觉能力的 MAIN 作为图片处理后备。

部分兼容服务不公开窗口元数据。此时应用会明确提示并按 4096 tokens 安全运行；最大输出按有效窗口的 25% 推导、上限 4096。可根据服务官方规格人工覆盖两项值。生成过程中还会预留安全余量，并只在服务明确返回上下文溢出时自动缩小证据后重试。

连接检查不会保存配置；深度验证只发送内置的极短测试文本或图片，但云服务仍可能计费。真正使用云端 MAIN/VLM/AUDIO 时，问题上下文、资料片段、页面图片或 Podcast 文本会发送到所配置的服务，请在上传敏感资料前确认服务条款和组织合规要求。

### 4.3 AUDIO 语音服务

Podcast 要求同一个 Provider 同时提供 TTS 合成与 ASR 验收。新配置使用同组织的 [audio-intel](https://github.com/wlf186/audio-intel) 服务；旧版 OpenAI TTS 记录升级后会保留为停用的 TTS-only 配置，但不能用于 Podcast。

```powershell
git clone https://github.com/wlf186/audio-intel.git C:\ai\audio-intel
Set-Location C:\ai\audio-intel
.\service.cmd doctor
.\service.cmd setup all
.\service.cmd start all
Invoke-RestMethod http://127.0.0.1:20810/api/v1/health
```

返回 quick-read，在设置中管理 AUDIO，编辑 `Sandevistan Audio`，地址填 `http://127.0.0.1:20810`，然后：

1. 点击“连接并读取模型”，确认 TTS 与 ASR 模型、设备均可用；
2. 为 Host A 与 Host B 选择不同的预置音色，或选择两个不同的声纹库人员；
3. 保持“Podcast 使用批量合成与安全并行加速”可减少模型重复装载；旧版 Audio Intel 未声明能力时选项不会显示并自动使用逐条合成；
4. 选择“验证并启用”，让系统分别验证两位主持人并完成一次短 TTS→ASR 闭环。

声纹克隆只允许选择 Audio Intel 声纹库中已有且具备可用样本的人员。需要使用时，先打开 <http://127.0.0.1:20810>，在“声纹库”创建人员并导入或录制样本，再回到 quick-read 刷新能力清单。系统会采用该人员最新的可用样本，样本超过 15 秒时由 Audio Intel 按词边界截断；两位主持人不能使用同一个人。预置音色可填写基础表达风格，quick-read 会追加稳定语速、音高和情绪范围的整集约束；克隆模式不发送上游不支持的风格指令。

Audio Intel 模型对磁盘、内存和 GPU 的要求明显高于主应用；完整能力需要 ASR、TTS 和内部 aligner，安装前请阅读 [audio-intel Windows 手册](https://github.com/wlf186/audio-intel/blob/main/docs/WINDOWS.md)。CPU-only 部署可改用 `.\service.cmd setup all --profile cpu`，启动命令仍为 `.\service.cmd start all`。

## 5. 更新、备份和卸载

更新前先停止服务，并备份重要运行数据：

```powershell
Set-Location C:\ai\quick-read
.\scripts\stop.ps1
Copy-Item runtime\data D:\backup\quick-read-data -Recurse
Copy-Item runtime\config.toml D:\backup\quick-read-config.toml
git pull --ff-only
.\scripts\bootstrap.ps1
.\scripts\start.ps1
```

请根据实际情况替换备份位置。不要将 `runtime/config.toml`、`runtime/data`、API Key、上传资料或生成媒体提交到 Git。

卸载时先运行 `stop.ps1`，再删除 quick-read 项目目录即可。Ollama、其模型以及单独部署的 audio-intel 不在该目录内，需要分别卸载或清理。

## 6. 局域网访问与安全

默认只监听本机。需要从可信局域网访问时，先生成强访问密钥：

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

编辑 `runtime/config.toml`：

```toml
[server]
host = "0.0.0.0"
port = 20830

[security]
access_key = "替换为刚生成的随机密钥"
```

重启服务，并仅为 TCP 20830 创建“专用网络/可信局域网”防火墙规则。不要将此端口直接暴露到公网；公网访问应由具备 TLS、身份认证和访问控制的反向代理承载。非 loopback 地址未配置 `security.access_key` 时，应用会拒绝不安全启动。

## 7. 常见问题

| 现象 | 处理建议 |
| --- | --- |
| `running scripts is disabled` | 在当前窗口执行 `Set-ExecutionPolicy -Scope Process Bypass`，不要修改为全局无限制。企业策略禁止时联系管理员。 |
| 找不到 `node` 或版本过低 | 安装 Node.js 24 LTS，关闭并重新打开 PowerShell。 |
| 找不到 `corepack` | 执行 `npm install -g corepack`，重新打开 PowerShell 后验证 `corepack --version`。 |
| GitHub、包源或模型下载超时 | 检查 DNS、代理和证书拦截；设置 `HTTP_PROXY`/`HTTPS_PROXY` 后重跑 bootstrap。 |
| `SHA-256 verification failed` | 不要绕过校验。删除报错中对应的 `runtime\cache\downloads` 单个安装包后重试；反复失败通常是代理缓存或下载被篡改。 |
| Defender/杀毒软件拦截 `.tools` 或 `.venv` | 先核对下载来源和摘要；确认为误报后只对项目目录设置最小范围排除，不要全局关闭防护。 |
| 路径过长、访问被拒绝、文件反复消失 | 移到 `C:\ai\quick-read` 一类本地短路径，避开 OneDrive、网络盘、`Program Files` 和受控文件夹。 |
| 20830 端口被占用 | 执行 `Get-NetTCPConnection -LocalPort 20830 -ErrorAction SilentlyContinue`；停止占用程序，或修改 `runtime/config.toml` 的端口。 |
| 启动脚本超时或服务立即退出 | 查看 `runtime\logs\server-error.log`；确认 bootstrap 完整成功，再重新启动。 |
| Ollama 无法连接 | 确认托盘中的 Ollama 已运行，并检查 `Invoke-RestMethod http://127.0.0.1:11434/api/tags`；设置页地址不要使用容器主机名。 |
| 模型清单为空 | 服务可能不提供清单接口；切换为手动模型 ID 后执行深度验证。Ollama 则先运行 `ollama pull <模型>`。 |
| AUDIO 无法启用 | 在 Audio Intel 执行 `.\service.cmd status`，确认 ASR 与 TTS worker 均已注册；然后重新“连接并读取模型”。 |
| 声纹克隆不可选或人员为空 | 确认所选 TTS 模型支持 voiceprint，并先在 Audio Intel 声纹库创建人员及至少一个完成处理的样本，再刷新 quick-read 的能力清单。 |
| Office 预览或音频转码不可用 | 重新运行 `bootstrap.ps1`，不要设置 `SANDEVISTAN_SKIP_TOOLS=1`，并在 `/api/status` 检查工具状态。 |
| ARM64 安装失败 | ARM64 目前为实验支持；升级 Windows x64 模拟组件，或改用 Windows x64 主机。提交问题时附上系统架构及日志。 |

仍无法解决时，提交 GitHub Issue，并附上 Windows 版本/架构、`node --version`、执行的命令和脱敏后的 `server-error.log`。不要上传 API Key、`runtime/config.toml`、原始资料或数据库。
