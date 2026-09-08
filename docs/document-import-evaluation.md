# 资料导入评测

`scripts/evaluate_imports.py` 冻结指定源码，在隔离数据库中通过 HTTP 上传接口接收文件，执行真实导入任务，记录正文、索引、引用定位、视觉处理和完整入库耗时。不会修改现用 Provider 或自动重建已有资料。样本和服务地址必须显式提供；服务会收到页面图片，可选问答会向 MAIN 发送原文。

```bash
.venv/bin/python scripts/evaluate_imports.py \
  --output runtime/evals/office-run \
  --sample /path/to/example.docx --sample /path/to/example.pptx \
  --sample /path/to/example.xlsx --sample /path/to/example.pdf \
  --main-url http://127.0.0.1:11434 --model gemma4:e4b \
  --vlm-url http://127.0.0.1:11434 --vlm-model your-vision-model
```

默认分别逐个上传及整批上传；`--layout single|batch|both` 控制场景，`--repeats 3` 重复运行。`--source-root` 指定基线代码快照，旧版不支持 XLSX 时应保留真实的 415 结果。每次使用新的输出目录，不能将不同源码的结果当作同一版本续跑。

MAIN 固定 30720 上下文和 4096 最大输出；图片处理固定使用 VLM → MAIN → OCR 顺序。可通过 `--question` 添加有来源约束的问答；问答时间不计入入库耗时。常规浏览器 smoke 不进行这些上传或模型调用。

产物包括源码哈希、样本校验和、`results.json`、逐来源提取文本与定位、`provider-calls.jsonl`。这些内容位于本地评测目录，不得上传至 Release。退出码 0 仅表示每份文件就绪且有索引，不保证所有视觉内容、公式缓存或问答均通过质量检查；检查 warnings、searchable、quality_candidates 及问答引用。

性能对照保持相同服务、模型、图片策略、设备和样本。先区分冷启动与预热，再交替运行新旧版本并比较至少三次的完整入库中位数。解析、转换和各处理器耗时存在包含关系，不重复相加。批量场景另外记录排队耗时；不能把仅正文解析耗时或提前就绪作为完整性能提升。

内容验收需核对：每份文件可检索；工作表没有缺失；单元格数值、日期、百分比与公式缓存一致；图表类别与数值配对正确；扫描页的关键内容仍可获取；引用定位指向对应资料。使用合成文件覆盖转换工具缺失、超时、非零退出、编码异常、合并单元格和公式缓存缺失，用户样本不加入自动化测试仓库。
