# Podcast 评测与 TTS 资格维护

本页说明文字稿评测、整集冻结验收与固定模型的 TTS 资格评测。它们使用不同的 manifest，不能互换。结果、输入清单、评分和音频都应放在未纳入 Git 的 `runtime/evals/`，不要使用生产资料作为提交到仓库的测试夹具。

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
