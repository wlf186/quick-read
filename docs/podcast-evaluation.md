# Podcast 评测与 TTS 资格维护

本页说明文字稿评测、整集冻结验收与固定模型的 TTS 资格评测。它们使用不同的 manifest，不能互换。结果、输入清单、评分和音频都应放在未纳入 Git 的 `runtime/evals/`，不要使用生产资料作为提交到仓库的测试夹具。

交互生成现在允许将有依据、可播放但未达到全部指标的产物作为 `partial` 交付；原始 `quality.passed`、`audio_quality.passed` 和时长检查不改写为通过。本页的严格脚本评测、冻结套件和 TTS 资格评测仍按原有门槛判定，降级交付不是资格通过。综合功能实测使用 `scripts/evaluate_generation.py`，输入与输出形式见 README。

## 交互生成与严格评测

网页生成的 `partial` 是部分交付，不是严格评测通过。请分别读取脚本 `quality.passed`、ASR `audio_quality.passed` 和实际时长 `audio_quality.duration.passed`；ASR 不可用时部分指标缺省，不能把缺省值当作 0 错误率。界面的音频 VERIFIED 要求 ASR 为 true，且已有时长检查不为 false；旧产物可能没有时长字段。任务 complete、产物 ready、文件可播放均不能独立证明全部质量通过。

产品生成保留可用内容并附质量评级；事实支持待核实的轮次不直接删除。额外 MAIN 审校最多一次，已知连贯性断裂最多在同次审校中做一处局部修复。仍有断裂时保留连续短版或草稿。严格脚本与 TTS 资格评测保持各自既有验收门槛。

### 脚本恢复与音频复用

双人音频 V4 使用证据、主张表和携带真实前文的章节生成。每项任务最多一次额外 MAIN 审校和一次共享技术恢复；大时长所需的正常分批不计作重试。纯提问省略空引用数组、明确复合引用如 `C1|E1` 可本地恢复，未知编号不会挂到无关资料。检查相邻问答、章节衔接和完整结尾，不能把同属一个主题当作已经回答问题。首轮声称已有前文属于明确断裂，审校笼统通过不能覆盖该问题；同次审校最多返回连续六轮局部替换，未经独立二次审校会注明。无法恢复时只保留连续短版或草稿，不跨缺失章节拼接。时长、角色比例、问句比例和措辞风格是软提示，不据此重写整集。

支持 Audio Intel v1 批量协议时，TTS 按输入顺序批量合成并保留逐轮 WAV。产品任务关闭提前合成，等待脚本检查结束后再开始 TTS。精确文本、语言、主持人、音色、模型 checkpoint、设备及协议哈希用于断点复用。合成或媒体处理失败保留脚本；不得跨缺失轮次拼接音频。完成后执行一次 ASR，保留原始指标；不为 ASR 分数或时长再次合成。设备故障的已有同模型 CPU 回退仍由 AUDIO 配置控制。

默认自动选择 12–25 分钟目标时长，也可选择 5、10、20 或 30 分钟；首次默认输出简体中文，并在浏览器中记住上次选择。MAIN Provider 的上下文和输出窗口较小时，系统使用更短场景与压缩剧集记忆；大窗口模型可生成更长场景；产品交付评级与严格资格门槛独立。已有 V2 产物保持兼容。


## 运行环境与调用边界

以下命令从 quick-read 项目根目录运行。Windows PowerShell 将 `.venv/bin/python` 替换为 `.\.venv\Scripts\python.exe`；命令中的 `<...>` 和 `sample-01-*` 等示例路径均需替换为实际值。manifest 内相对路径按**命令工作目录**解析，不按 manifest 所在目录解析；也可以使用绝对路径。

生成文字稿需要已有 Notebook 及 MAIN Provider。渲染需要已启用并通过能力检查的 AUDIO Provider、项目内 FFmpeg，以及可用的 TTS/ASR 模型。`--candidate-json` 跳过 MAIN，`--render-candidate` 调用 TTS 和全片 ASR，`--reference-audio` 调用 ASR。文本或参考音频会发送到所选服务；若配置了云端 Provider，可能产生费用。这些命令是主动集成评测，普通 pytest 与浏览器 smoke 不运行它们。

`--tts-model` 和 `--tts-device` 只覆盖本次渲染，不修改已保存的 Provider。ASR 仍使用当前 AUDIO 配置。允许 GPU→CPU 回退时，设备故障可能使实际设备与请求设备不同；资格评测必须核对输出的 `execution.compute_device` 和 `fallback_used`，不能把 CPU 回退结果当作 GPU 资格证据。整个对比期间固定 AUDIO 服务版本、checkpoint revisions、主持人音色/样本、表达指令、ASR 配置和 FFmpeg 构建。

## 1. 文字稿与单候选音频

先生成通过门禁的候选，默认不会调用 TTS：

```bash
.venv/bin/python scripts/evaluate_podcast.py --notebook-id <NotebookID> --minutes 5 --language zh-CN
```

脚本输出 JSON 中的 `output` 是本次结果目录，形如 `runtime/evals/podcast-v4-<时间戳>`。成功后冻结其中的 `candidate.json`，后续渲染始终复用这一份，避免重新调用 MAIN 导致文字稿变化。需要文字稿对比时可加 `--baseline-artifact <旧播客ID>`，需要参考录音转写盲评时可加 `--reference-audio <样本.m4a>`；英文参考音频配合 `--reference-language English`。

逐轮与批量渲染使用同一候选：

```bash
.venv/bin/python scripts/evaluate_podcast.py --candidate-json <冻结的candidate.json> --render-candidate --tts-model qwen3-tts-0.6b --tts-device gpu --tts-mode single
.venv/bin/python scripts/evaluate_podcast.py --candidate-json <冻结的candidate.json> --render-candidate --tts-model qwen3-tts-0.6b --tts-device gpu --tts-mode sequence
```

每条命令生成独立结果目录，顺序执行并记录各自 `output`。`--tts-mode` 默认为 `single`，与网页 Podcast 默认开启批量合成不同。`sequence` 要求能力清单声明兼容的 v1 协议，且 Provider 的 `podcast_sequence_tts` 未关闭；能力不足或批量失败时评测失败，不回退逐轮冒充批量结果。同模型 GPU→CPU 回退仍由 AUDIO 配置控制。

主要输出如下；失败时部分文件可能不存在，应先看退出状态和报告：

| 文件 | 用途 |
| --- | --- |
| `candidate.json`、`comparison.json` | 冻结文字稿与脚本质量指标 |
| `candidate.m4a` | 经过 FFmpeg 规范化的成品音频 |
| `candidate-parts/0000.wav` 等 | 按轮次保留的原始 PCM16 WAV |
| `candidate-asr.json` | 成品转写与验收所需 ASR 结果 |
| `candidate-audio-quality.json` | 实际时长、音频门禁及 `execution` 模型/设备/批量统计 |
| `blind-review.md`、`mapping.json` | 有对比输入时的匿名文字评审与私有映射 |
| `failure.json` | 新文字稿未通过门禁时的报告 |

## 2. TTS 双基线资格评测

当前 `qualify_podcast_tts.py` 固定评测以下三组，尚不是任意模型/设备的通用资格工具：

| 角色 | 模型 | 设备 | 模式 |
| --- | --- | --- | --- |
| 批量候选（脚本生成） | `qwen3-tts-0.6b` | GPU | `sequence` |
| `baseline_06_dir`（事先准备） | `qwen3-tts-0.6b` | GPU | `single` |
| `baseline_17_dir`（事先准备） | `qwen3-tts-1.7b` | CPU | `single` |

至少准备两个不同样本，每个样本各自冻结一份已通过门禁的文字稿。对每份文字稿执行以下两条命令并记录成功的 `output`，作为该样本的两个基线目录：

```bash
.venv/bin/python scripts/evaluate_podcast.py --candidate-json <该样本冻结的candidate.json> --render-candidate --tts-model qwen3-tts-0.6b --tts-device gpu --tts-mode single
.venv/bin/python scripts/evaluate_podcast.py --candidate-json <该样本冻结的candidate.json> --render-candidate --tts-model qwen3-tts-1.7b --tts-device cpu --tts-mode single
```

两个基线目录都必须含 `candidate.json`、`candidate.m4a`、`candidate-audio-quality.json` 和完整的 `candidate-parts/*.wav`。每个基线的 `candidate.json` 必须与该样本冻结文字稿的 SHA-256 完全一致，语义相同但格式不同也会被拒绝。检查质量报告确实通过，且实际模型、设备、模式符合上表。

在 `runtime/evals/tts-manifest.json` 创建清单；将以下目录替换为前述命令实际输出的目录。样本 ID 必须非空且唯一，建议使用简单目录名：

```json
{
  "samples": [
    {
      "id": "sample-01",
      "candidate_json": "runtime/evals/sample-01-script/candidate.json",
      "baseline_06_dir": "runtime/evals/sample-01-06-single",
      "baseline_17_dir": "runtime/evals/sample-01-17-single"
    },
    {
      "id": "sample-02",
      "candidate_json": "runtime/evals/sample-02-script/candidate.json",
      "baseline_06_dir": "runtime/evals/sample-02-06-single",
      "baseline_17_dir": "runtime/evals/sample-02-17-single"
    }
  ]
}
```

生成批量候选并执行首次客观对比：

```bash
.venv/bin/python scripts/qualify_podcast_tts.py prepare --manifest runtime/evals/tts-manifest.json
```

默认输出到 `runtime/evals/podcast-tts-qualification-<时间戳>`；也可通过 `--output <新目录>` 指定不存在的目录。该命令没有 `--resume` 参数。它复用文字稿，实际调用 AUDIO，输出每个样本的 `sequence/` 音频与质量报告、`objective-result.json`、匿名 `blind/A.m4a`、`B.m4a`、`C.m4a`，并在运行目录输出 `qualification.json`、`private-mapping.json`、`scores-template.json` 和 `blind-review.md`。

首次门禁通过时状态为 `awaiting_scores`；该名称不要求必须人工评分，可以直接进行自动终验。未通过时为 `objective_failed`，退出码为 2。使用同一份清单、保持所有基线和候选文件不变，执行：

```bash
.venv/bin/python scripts/qualify_podcast_tts.py auto-finalize --run-dir <prepare输出目录> --manifest runtime/evals/tts-manifest.json
```

自动终验仅分析本地已有报告与逐轮 WAV，不再次调用 MAIN、TTS 或 ASR。它输出 `automated-result.json`，并更新 `qualification.json`；检查涵盖信号完整性、可懂度、说话人一致性、韵律稳定性、听觉疲劳代理指标及修复率。当前包括错误率不超过 0.08、说话人对齐至少 0.95、修复轮次比例不超过 5%，同时要求双基线对比通过；全部条件以脚本实现为准。状态为 `passed` 或 `failed`，门禁失败退出码为 2；输入错误可能以异常退出，不能只凭目录存在判定成功。

人工盲听是可选补充：评分前不打开 `private-mapping.json`，按 `blind-review.md` 将模板中的五个维度填为 1–5 的整数，另存为评分 JSON。`listening_fatigue` 分数越高表示越不易疲劳。随后执行：

```bash
.venv/bin/python scripts/qualify_podcast_tts.py finalize --run-dir <prepare输出目录> --scores <评分.json>
```

人工终验要求批量候选每个维度至少 4 分，总分不低于任一基线，且首次客观门禁通过；结果写入 `final-result.json` 和 `blind-scores.json`。两种终验都会更新 `qualification.json` 的顶层状态，应分别查看 `automated-result.json` 与 `final-result.json`，避免把最近一次状态误当作另一种评审结论。

### 将评测结论用于自动推荐

以上命令只生成本地报告，不修改数据库、Provider 配置或代码中的推荐资格表。自动声学指标是代理指标，不等同于人工听感结论。

维护者应核对服务报告的 checkpoint revisions、实际推理设备及双基线报告，再另行修改 `src/sandevistan_read/providers.py` 中的 `PODCAST_TTS_QUALIFIED_TARGETS`。当前候选 revision 位于 `PODCAST_TTS_CANDIDATE_REVISIONS`；条目绑定各 checkpoint variant、允许的设备和评测方法。资格脚本不会自动证明运行时 checkpoint 与这张表相同，需要保留对应能力快照及报告作证据。新模型或新设备必须先扩展评测工具及覆盖，不能直接套用当前固定的三组标签。

GPU 通过不代表 CPU 通过；revision 改变后，原条目不再授予该服务默认组合优先资格。服务默认优先资格与按已安装模型排序的回退是不同路径，回退模型不一定在资格表内。修改资格表时同步 Provider 推荐测试，并在变更说明中记录 checkpoint、设备、方法和脱敏结果摘要，原始音频及资料仍留在本地。

## 3. 整集多样本冻结验收

`evaluate_podcast_suite.py` 评估完整脚本与音频流水线，使用包含 Notebook 和参考录音的独立 manifest。例如：

```json
{
  "samples": [
    {
      "id": "episode-01",
      "notebook_id": "替换为NotebookID",
      "source_ids": [],
      "minutes": 5,
      "language": "zh-CN",
      "reference_audio": "runtime/evals/references/episode-01.m4a"
    }
  ]
}
```

```bash
.venv/bin/python scripts/evaluate_podcast_suite.py prepare --manifest runtime/evals/suite-manifest.json --mode frozen
.venv/bin/python scripts/evaluate_podcast_suite.py finalize --run-dir <prepare输出目录> --scores <评分.json>
```

`frozen` 模式为每个样本生成新候选并执行 TTS/ASR；`development` 模式允许样本提供 `candidate_json` 复用通过门禁的候选。`--sample <id>` 可重复指定，选样集合参与套件身份哈希；`--tts-model/--tts-device` 支持单次覆盖。此脚本没有 `--tts-mode`，目前调用默认逐轮渲染；专门的批量对比使用前两节工具。

续跑使用 `prepare` 的 `--output <原目录> --resume`，保持原清单、模式、选样和模型/设备覆盖不变；已有匿名映射会保留。套件按音频 SHA 缓存参考 ASR，并输出基于音频转写的匿名 A/B 包及 `scores-template.json`。独立评分后执行 `finalize`，检查六维分数、音频质量以及 45k MAIN token 上限。它的六维评分模板不适用于上一节 TTS 五维评分。


## 4. 综合功能实测

`evaluate_generation.py` 通过隔离应用的真实 HTTP 接口验证摘要、六轮连续问答、Quiz 作答、Flashcard 和完整 Podcast。先运行 bootstrap，准备自己的 PDF、MAIN 服务以及提供 `qwen3-tts-0.6b`、`qwen3-asr-0.6b` 和 Vivian/Dylan 预置音色的 AUDIO 服务。MAIN 上下文显式设为 30720；AUDIO 使用 GPU 并允许同模型 CPU 回退。此工具不继承正式应用的 Provider 或访问密钥；示例适用于工具可直接访问的服务。

```bash
.venv/bin/python scripts/evaluate_generation.py --output runtime/evals/gemma-baseline --phase baseline --sample /path/to/sample.pdf --main-url http://127.0.0.1:11434 --model gemma4:e4b --audio-url http://127.0.0.1:20810
```

必须替换样本路径及服务地址。脚本的原实验样本默认位于被 Git 忽略的 `.experiment/`，原 MAIN 默认地址仅适用于原实验环境；公开部署应总是显式提供 `--sample` 和 `--main-url`。模型仍需事先安装，30720 上下文并不等于单次输出上限。

| phase | 覆盖 |
| --- | --- |
| baseline | 中文默认参数的五类功能，共 5 个场景 |
| matrix | 中文/英文/自动；题卡最小、默认、最大数量 × 四档难度；播客自动及 5/10/20/30 分钟，共 93 个场景 |
| extras | 输出上限 1024/1536/8192、温度 0/1/2、lite/full、定制要求和逐轮 TTS，共 45 个场景 |

切换 phase 时使用新的 `--output` 目录，并保留上述显式样本/服务参数。`--case` 可重复指定该 phase 的场景 ID（如 `quiz-zh-CN-10-mixed`），`--port` 默认 20831，`--job-timeout` 默认 10800 秒/任务。该矩阵覆盖代表组合，不穷举全部合法参数组合。

`--resume` 要求已有目录与相同样本内容、MAIN 地址/模型、AUDIO 地址、phase 和场景选择；续跑使用原应用源码快照，跳过已经记录的场景，**包括失败场景**。修复代码或重新测试失败项时应另开输出目录，而不是用 resume 重测。聊天逐轮保存，可续接已完成轮次。模型和工具目录通过符号链接共享，Windows 需要开发者模式或创建链接权限。并行实验使用不同端口；MAIN 请求和整项音频任务分别通过跨进程锁串行化。

| 输出 | 用途 |
| --- | --- |
| `manifest.json`、`cases.json`、`source-hashes.json`、`evaluator.py` | 输入身份、场景清单、应用源码哈希与评测器副本 |
| `instance/` | 隔离源码、配置、数据库和生成媒体；不是正式应用目录 |
| `results.json`、逐场景 JSON | 参数、结果、产物及 Quiz 提交后的反馈 |
| `current.json`、`main-calls.jsonl`、`server.log` | 当前任务、真实 MAIN 调用输入输出与服务日志 |

`results.json` 的状态为 passed/degraded/failed：passed 表示自动流程与相应系统检查通过，degraded 表示返回部分或带警告内容，failed 表示任务或评测检查失败。正常结束时只要没有 failed 就返回退出码 0，**即使全部结果都是 degraded**；启动或输入错误也可能以异常退出。只有单独汇总工具才可能额外标记 recovered，不是此脚本的原生状态。

输出文件留在本地 `runtime/evals/`，但执行过程中资料片段、脚本和音频会发送到指定 MAIN/AUDIO，不能将“隔离目录”理解成不联网。原始日志、数据库、样本及媒体不应提交到 Git 或附到公开 Release。公开结论应区分完成、降级和事实/音频质量，并注明固定模型、样本和覆盖范围。

## 5. 上下文策略对照

`scripts/evaluate_context_strategy.py` 将应用源码冻结到独立实例，复用同一份解析索引，对摘要、六轮问答、Quiz、Flashcard 和 Podcast 脚本进行真实 MAIN 调用，默认重复两次。没有 `--audio` 时 Podcast **只验证脚本**，不能将结果视为 TTS/ASR 或完整任务通过；添加该参数才调用真实 AUDIO。

准备阶段不调用 MAIN/VLM，使用明确指定的 PDF/EPUB。以下三个名字是评测场景标识，可映射到自己的短文、长书和第三份资料。`multi` 同时勾选三份，不能作为五本真实书的验证证据。

```bash
.venv/bin/python scripts/evaluate_context_strategy.py --prepare --output runtime/evals/context-fixture --sample bitcoin=/path/to/short.pdf --sample geb=/path/to/long.pdf --sample strange-loop=/path/to/third.epub
.venv/bin/python scripts/evaluate_context_strategy.py --output runtime/evals/context-new --fixture runtime/evals/context-fixture --provider-id <MAIN_PROVIDER_ID>
.venv/bin/python scripts/evaluate_context_strategy.py --output runtime/evals/context-old --fixture runtime/evals/context-fixture --provider-id <MAIN_PROVIDER_ID> --source-root /path/to/frozen-baseline
```

`--provider-id` 从本机配置读取指定 MAIN 的模型、窗口、温度及凭据，不修改正式 Provider。凭据仅经子进程环境传递，并在隔离数据库中加密保存，不写入 manifest 或调用日志。省略该参数使用原实验的 Gemma 地址与 30720 上下文；公开复现应指定自己的 Provider。隔离子进程使用直连，不继承主进程的 HTTP 代理；在隔离配置中显式启用 `balanced`，生产默认仍为 `conservative`。旧源码不识别该策略字段时保持旧行为。

`--corpus bitcoin geb multi`、`--kind summary chat quiz flashcard podcast` 可选择子集；`--language` 支持 zh-CN/en/auto。窗口和题卡参数的完整边界测试仍由原综合评测和离线测试覆盖，本工具不能替代全部参数矩阵。`--audio --kind podcast --corpus bitcoin --language zh-CN` 测试 5 分钟音频，选择 `geb` 测试 30 分钟；需要提供 `--audio-url` 或使用默认本机 AUDIO。

输出目录保存源码哈希、输入清单、逐次 MAIN 用量与响应、逐场景结果。`--resume` 复用已有实例，要求相同场景、Provider ID、模型配置及能力指纹、输入清单、AUDIO 地址、语言和重复次数，跳过已经记录的结果（包括失败）。修复代码后重新验收必须使用新目录；不要覆盖正在运行的源码快照。相同服务和模型的 MAIN 调用默认通过跨进程锁串行化；远端 API 的独立评测可显式使用 `--main-concurrency 2`，共享两个调用槽位，Ollama 仍只允许串行。并发设置写入评测身份，延迟对照须注明运行时并发与排队；发生限流的组需降低并发后单独重跑，不合并失败前后的样本。完整音频任务另有串行锁。因此场景耗时包含排队，不用于独立延迟比较。AUDIO 在隔离库中固定为 Qwen3 TTS/ASR 0.6B，并保存探测能力后执行，实际验收读取持久化音频产物。

新旧版本必须使用同一份索引和模型配置。运行前冻结关键要点与原文定位，对输出做匿名事实和覆盖核验，不能用关键词命中、选材量增加或文件可播放代替质量判断。分别报告执行失败、降级、事实正确性、实际覆盖、时长和资源用量。当前均衡策略的 300000 tokens 是整任务预算，包含重试；严格脚本与 TTS 资格评测继续使用原资格条件。超过真实验证窗口的配置仅能报告预算模拟结果。


### 实际音频时长与摘要核验

完整 Podcast 任务固定 MAIN 配置，均衡上下文路径还固定资料选材快照。产品时长以目标的 80%–125% 为基本达标范围，偏差保留为明确提示；不再执行基于实测时长的 MAIN 改写或重合成。严格脚本资格工具维持原有行为。

摘要使用所选片段的完整原文综合，并独立核验每点的陈述者、示例性质及限定；evidence_v4 不生成摘要预读笔记。引用编号和连续摘录检查不能替代对照审阅，程序返回 passed 不等同于人工事实质量资格。评测记录分别列出 MAIN 排队和推理耗时；不同源码快照的 pilot 不合并为最终资格样本。

上下文对照工具支持 `--count`、`--difficulty easy|medium|hard|mixed` 和 `--minutes 5|10|20|30`，用于数量、难度和时长边界；这些参数进入续跑身份检查。`--reference` 固定评审参考，`--script-file` 可在 `--audio --kind podcast` 下重放同一脚本，但原脚本的质量警告仍须保留并区分。任务退出后清理隔离数据库中的 MAIN 凭据副本；被外部强制终止的旧实例仍需单独检查。新增降级、事实错误或未完成的音频验收都不能计作资格通过；修复后必须重新冻结源码，不能合并不同修订的结果。


### 摘要和问答的独立资格（evidence_v4）

摘要与问答取消静态原文容量上限，遵守 20% 窗口余量、300000 累计 MAIN tokens 和至少 25% 审校／恢复预留；1M / 384K 等超出已部署模型能力的配置只作离线预算模拟。输出目标固定，不能以输出更长代替质量提升。聊天历史预留同时受安全输入和累计任务预算限制；恢复请求按剩余累计预算重新装入原文。容量测试必须分别检查摘要和聊天，确保窗口增长不会使原文预算倒退；触及任务总预算后允许平台期。其它功能不因这次修改取得资格。

同一 Provider 的三个诊断组可分别使用 `--strategy conservative`、`--strategy balanced --audit-mode external` 与 `--strategy balanced --audit-mode native`。external 只用于隔离审校影响，不能获得生产资格。若保守版本在旧任务限额处直接拒绝调用，应另作等预算诊断：三个组都使用 balanced 的累计预算和固定输出，第一组添加 `--selection conservative --audit-mode external` 保留旧选材，第二组只添加 `--audit-mode external`，第三组使用 native；这样不会把任务限额阻断混同于选材质量。`--selection conservative` 仅允许 summary 的 external 诊断。`--context-window` 用于同模型容量扫描，不得超过实际配置窗口；不修改远端模型窗口。默认 native。所有参数进入续跑身份。

正式对照前冻结 v0.4.4、候选源码、索引以及逐点绑定原文的参考。每个 Provider × 功能需要 12 对场景（短文／长书／三资料 × 中英 × 两次）；两模型两功能共 96 个场景，问答每场景六轮。逐点审阅遗漏、限定条件、事实及引用错误；关键词或目录命中不能代替参考，自动审阅不得标为人工盲评。

资格报告包括 `provider_fingerprint`、`kind`、`strategy_version`、`audit_mode=native`、`review_method=source_grounded`，以及 reference/baseline/candidate/fixture 的 SHA-256。`pairs` 的每项含 corpus/language/repeat，baseline/candidate 各含非负整数 missing/critical/failed/degraded/fact_errors/citation_errors；critical 是同一参考的关键点数。`artifacts` 指向 reference、baseline、candidate、fixture、review 的本地 path/sha256，其中 candidate 是评测实例的 source-hashes.json。review 必须保存逐点原文依据及输出判定。

```bash
.venv/bin/python scripts/qualify_context_strategy.py --report runtime/evals/<run>/qualification.json --provider-id <MAIN_ID> --kind summary
# 仅对通过完整审阅的报告登记，显式 conservative 配置仍优先：
.venv/bin/python scripts/qualify_context_strategy.py --report runtime/evals/<run>/qualification.json --provider-id <MAIN_ID> --kind summary --apply
```

登记要求长资料关键遗漏相对减少至少 20%、候选没有事实／引用错误、短文不退化、观察到的失败及降级总数不增加，且部署源码与候选快照一致。资格只对该 Provider × 功能生效；模型、容量、温度、思考模式等配置变化会使旧资格不匹配。有限样本通过不保证所有资料、参数或远端未报告的模型更新均通过；发现模型更新应重新验证。未通过的组合保留 conservative，不能据退出码 0 或预算模拟自动登记。

摘要按来源轮转，并按开头／结尾／中部递归补齐区间，选材集合随预算增长保持嵌套；送入模型前恢复原文顺序。明确的索引／书目页标志会应用到该来源页面的全部分片，避免同页其它分片绕过过滤。短定义与数值证据表仍保留。公式提取异常显示待核实提示；不因措辞或引用格式偏差反复重写可用回答。

### 优先交付策略评测

产品生成的自动评级与严格模型/TTS 资格验收是独立概念。`quality_assessment` 报告实际抽查数量及逐项问题；未完成审校不应记作质量通过。`delivery_status=script_only/draft_only` 不计入完整音频成功率。产品时长容忍区间为 80%–125%；严格资格工具继续使用各自既有门槛。

比较成本时同时报告 MAIN 调用、输入/输出 tokens、耗时、实际题卡数量或音频时长；不能把产出减少简单算作同等质量下的效率提升。每次比较冻结源码、样本索引、参考与 Provider 配置。持续 Provider 故障应停止该轮，已取消的旧矩阵不自动恢复。原始提示、样本、数据库及音频只留在本地评测目录，不随 release 发布。

### 连贯性审校预算与覆盖

产品路径在首个 MAIN 调用前保护一次审校调用及 `min(累计 token 上限 × 25%, 模型单次安全输入输出预算)` tokens；预留本身不计费，调用按现有用量规则结算。生成与技术恢复不能占用预留。审校按剩余额度重新装入完整相邻对话，不截断单轮后宣称已检查。`quality_report.episode_audit` 的 `status` 为 `complete/partial/unavailable`，`coverage_mode` 为 `full/sampled`；`reviewed_transitions` 中 i 表示实际检查了 i 到 i+1，`checked_transitions/total_transitions` 以最终保留文本计算。局部修复涉及的轮次不计入独立检查覆盖。服务异常仍可保留音频，但必须显示连贯性未验证；严格资格工具沿用原验收条件。评测须区分真实新生成、固定旧脚本重新审校、仅脚本与完整 AUDIO 运行。

审校在同一次回复中最多请求 8 项带依据的检查，返回时规范为 `transition_checks`，每项包含相邻轮次的原文短引文、connected/broken/uncertain 判断及理由；逐字核对引文后才计入覆盖。仅返回“已检查全部轮次”的列表不能充当具体审校。严格脚本评测的编剧提示与验收条件保持独立。

本地收尾检查开场无前文指代及连续三个纯问题；一轮先给出完整陈述再追问会中断计数。该规则只发现明显结构问题，不声称理解全部语义；命中后保留此前完整连续短版，缺少可用双人前缀时仅留草稿。
