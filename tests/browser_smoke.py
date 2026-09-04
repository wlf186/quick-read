from pathlib import Path
import json
import re
import tomllib

from playwright.sync_api import Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:20830"
QA_NOTEBOOK = "QA 2026-08-28 · ALL 4"


def authenticate(page: Page) -> None:
    page.goto(BASE_URL, wait_until="domcontentloaded")
    if page.get_by_text("访问授权", exact=True).is_visible():
        with (ROOT / "runtime/config.toml").open("rb") as handle:
            access_key = tomllib.load(handle)["security"]["access_key"]
        page.get_by_label("访问密钥").fill(access_key)
        page.get_by_role("button", name="授权访问").click()
        page.wait_for_selector(".shell")
    page.wait_for_function(
        "document.querySelector('.notebook-switch b') && "
        "document.querySelector('.notebook-switch b').textContent !== '选择或新建 Notebook'"
    )


def select_qa_notebook(page: Page) -> None:
    page.locator(".notebook-switch").click()
    option = page.locator(".notebook-options > button").filter(has_text=QA_NOTEBOOK)
    if option.count():
        option.click()
        page.wait_for_timeout(800)
    else:
        page.locator(".notebook-switch").click()


def assert_no_horizontal_overflow(page: Page) -> None:
    metrics = page.evaluate("({scroll: document.documentElement.scrollWidth, viewport: innerWidth})")
    assert metrics["scroll"] <= metrics["viewport"], metrics


def main() -> None:
    console_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
        page = context.new_page()
        api_requests: list[str] = []
        page.on("request", lambda request: api_requests.append(request.url) if "/api/" in request.url else None)
        page.set_default_timeout(8000)
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        authenticate(page)
        select_qa_notebook(page)
        page.wait_for_function("document.querySelectorAll('.source-row').length === 4")
        page.wait_for_function("document.querySelectorAll('.artifact').length === 4")

        page.wait_for_timeout(1000)
        api_requests.clear()
        if page.locator(".activity .job").count() == 0:
            page.wait_for_timeout(6500)
            assert api_requests == [], f"idle workspace still requested: {api_requests}"

        page.locator(".upload-zone input").set_input_files({"name": "diagram.png", "mimeType": "image/png", "buffer": b"visual-preview"})
        upload_dialog = page.get_by_role("dialog", name="确认上传")
        upload_dialog.wait_for()
        assert upload_dialog.get_by_text("VLM → MAIN → OCR", exact=False).is_visible()
        page.screenshot(path="/tmp/sandevistan-read-upload-policy-desktop.png")
        upload_dialog.get_by_role("button", name="关闭").click()

        assert page.title() == "Sandevistan-Read"
        assert page.get_by_role("heading", name="向资料提问").is_visible()
        assert page.locator(".source-row").count() == 4
        assert page.locator(".artifact").count() == 4
        assert_no_horizontal_overflow(page)
        page.screenshot(path="/tmp/sandevistan-read-ui-desktop.png")

        prompt = page.get_by_role("button", name="提炼三个核心结论")
        if prompt.is_visible():
            prompt.click()
            assert "三个结论" in page.get_by_label("向已选资料提问").input_value()
            page.get_by_label("向已选资料提问").fill("")

        page.get_by_role("button", name="设置").click()
        settings = page.get_by_role("dialog", name="Provider 配置")
        assert settings.is_visible()
        assert settings.evaluate("element => element.contains(document.activeElement)")
        assert settings.get_by_role("button", name="管理 MAIN").is_visible()
        page.screenshot(path="/tmp/sandevistan-read-provider-roles-desktop.png")

        inspection_requests: list[dict] = []

        def inspect_provider(route) -> None:
            request = json.loads(route.request.post_data or "{}")
            inspection_requests.append(request)
            selected = request.get("model", "")
            if request.get("role") == "audio":
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "status": "passed",
                            "connection_ok": True,
                            "activation_eligible": True,
                            "latency_ms": 42,
                            "catalog_supported": True,
                            "models": [{"id": "voice-1.7b", "name": "Voice 1.7B", "installed": True, "voice_modes": ["preset", "voiceprint"], "controls": {"instruction_voice_modes": ["preset"]}, "devices": [{"id": "gpu", "available": True, "precision": "BF16"}]}],
                            "capabilities": {
                                "voices": [{"id": "Vivian", "native_language": "zh-CN"}, {"id": "Dylan", "native_language": "zh-CN"}],
                                "asr": {
                                    "models": [{"id": "qwen3-asr-0.6b", "name": "Qwen3 ASR 0.6B", "installed": True, "devices": [{"id": "gpu", "available": True, "precision": "BF16"}]}],
                                    "recommended": {"model": "qwen3-asr-0.6b", "compute_device": "gpu"},
                                },
                            },
                            "voiceprint_library": {
                                "status": "ready",
                                "people": [
                                    {"id": "person-a", "name": "声纹甲", "eligible_sample_count": 1, "latest_sample": {"id": "sample-a", "language": "Chinese", "duration": 9.5, "created_at": "2026-09-01"}},
                                    {"id": "person-b", "name": "声纹乙", "eligible_sample_count": 1, "latest_sample": {"id": "sample-b", "language": "Chinese", "duration": 18.0, "created_at": "2026-09-02"}},
                                ],
                            },
                            "resolved_audio_config": {"host_a_voice_mode": "preset", "host_b_voice_mode": "preset", "host_a": "Vivian", "host_b": "Dylan"},
                            "recommended": {"model": "voice-1.7b", "compute_device": "gpu"},
                            "warning": None,
                            "error": None,
                        }
                    ),
                )
                return
            models = [{"id": "alpha-chat", "name": "Alpha Chat"}, {"id": "beta-vision", "name": "Beta Vision"}]
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "passed" if selected else "warning",
                        "connection_ok": True,
                        "activation_eligible": bool(selected),
                        "latency_ms": 42,
                        "catalog_supported": True,
                        "models": models,
                        "capabilities": {
                            "token_limits": {
                                "model_context_tokens": 32768,
                                "effective_context_tokens": 8192,
                                "max_output_tokens": 2048,
                                "context_source": "provider_metadata",
                                "output_source": "provider_metadata",
                            }
                        },
                        "recommended": None,
                        "warning": None if selected else "连接成功，请选择或填写模型",
                        "error": None,
                    }
                ),
            )

        page.route("**/api/providers/inspect", inspect_provider)
        assert settings.get_by_text("IMAGE PIPELINE", exact=True).is_visible()
        settings.get_by_role("button", name="管理 AUDIO").click()
        settings.get_by_role("button", name="添加 AUDIO Provider").click()
        assert settings.get_by_role("heading", name="添加 Provider").is_visible()
        assert settings.get_by_label("角色").is_disabled()
        assert settings.get_by_label("类型").locator("option").count() == 1
        settings.get_by_label("名称").fill("QA Audio")
        settings.get_by_label("服务地址").fill("http://localhost:20810")
        settings.get_by_role("button", name="连接并读取模型").click()
        settings.get_by_text(re.compile("1 个模型 · 42 ms")).wait_for()
        assert settings.get_by_role("heading", name="TTS 合成").is_visible()
        assert settings.get_by_role("heading", name="ASR 验收").is_visible()
        assert "Voice 1.7B" in settings.get_by_role("combobox", name=re.compile("TTS 模型")).inner_text()
        assert "Qwen3 ASR 0.6B" in settings.get_by_role("combobox", name=re.compile("ASR 模型")).inner_text()
        assert settings.get_by_label("ASR 设备").input_value() == "gpu"
        host_a_voice = settings.get_by_role("region", name="Host A 音色")
        host_b_voice = settings.get_by_role("region", name="Host B 音色")
        host_a_voice.get_by_role("button", name="声纹克隆").click()
        host_a_voice.get_by_label("声纹人员").select_option("person-a")
        host_b_voice.get_by_role("button", name="声纹克隆").click()
        duplicate_person = host_b_voice.get_by_label("声纹人员").locator('option[value="person-a"]')
        assert duplicate_person.get_attribute("disabled") is not None, duplicate_person.evaluate("element => element.outerHTML")
        host_b_voice.get_by_label("声纹人员").select_option("person-b")
        assert host_b_voice.get_by_text(re.compile("合成时截断")).is_visible()
        settings.get_by_role("heading", name="ASR 验收").scroll_into_view_if_needed()
        assert_no_horizontal_overflow(page)
        page.screenshot(path="/tmp/sandevistan-read-audio-provider-desktop.png")
        settings.get_by_role("button", name="返回角色配置").click()
        discard = page.get_by_role("dialog", name="放弃未保存的 Provider 配置？")
        assert discard.is_visible()
        discard.get_by_role("button", name="放弃修改").click()
        settings.get_by_role("button", name="返回角色概览").click()
        settings.get_by_role("button", name="管理 MAIN").click()
        settings.get_by_role("button", name="添加 MAIN Provider").click()
        settings.get_by_label("名称").fill("QA Provider")
        settings.get_by_label("服务地址").fill("https://api.example.com/v1")
        settings.get_by_role("button", name="连接并读取模型").click()
        settings.get_by_text("连接成功，请选择或填写模型").wait_for()
        assert settings.get_by_text("连接成功，请选择或填写模型").is_visible()
        model_picker = settings.get_by_role("combobox", name="模型")
        model_picker.click()
        settings.get_by_role("button", name=re.compile("Alpha Chat")).click()
        assert settings.get_by_text(re.compile("模型清单仍可用")).is_visible()
        model_picker.click()
        assert settings.get_by_role("button", name=re.compile("Alpha Chat")).is_visible()
        assert settings.get_by_role("button", name=re.compile("Beta Vision")).is_visible()
        settings.get_by_label("搜索模型").fill("beta")
        assert settings.get_by_role("button", name=re.compile("Beta Vision")).is_visible()
        assert settings.get_by_role("button", name=re.compile("Alpha Chat")).count() == 0
        page.keyboard.press("Escape")
        assert settings.get_by_label("搜索模型").count() == 0
        assert page.get_by_role("dialog", name="放弃未保存的 Provider 配置？").count() == 0
        assert settings.get_by_role("button", name="验证并启用").is_enabled()
        settings.get_by_label("Temperature 覆盖").fill("1")
        settings.get_by_label("上下文窗口覆盖（tokens）").fill("16384")
        settings.get_by_label("最大输出覆盖（tokens）").fill("2048")
        settings.get_by_role("button", name="连接并读取模型").click()
        settings.get_by_text(re.compile("2 个模型 · 42 ms")).wait_for()
        assert inspection_requests[-1]["config"]["temperature"] == 1
        assert settings.get_by_label("Temperature 覆盖").input_value() == "1"
        assert settings.get_by_role("button", name="验证并启用").is_enabled()
        assert_no_horizontal_overflow(page)
        page.screenshot(path="/tmp/sandevistan-read-provider-desktop.png")
        settings.get_by_role("button", name="返回角色配置").click()
        discard = page.get_by_role("dialog", name="放弃未保存的 Provider 配置？")
        assert discard.is_visible()
        discard.get_by_role("button", name="放弃修改").click()
        assert settings.get_by_role("button", name="添加 MAIN Provider").is_visible()
        page.keyboard.press("Escape")
        assert settings.count() == 0

        page.locator(".studio-cards button").filter(has_text="Quiz 题库").click()
        study_create = page.get_by_role("dialog", name="生成理解型 Quiz")
        assert study_create.is_visible()
        assert study_create.get_by_label("数量").is_visible()
        assert study_create.get_by_label("难度").is_visible()
        assert study_create.get_by_label("输出语言").is_visible()
        assert study_create.get_by_label("定制要求").is_visible()
        page.keyboard.press("Escape")
        assert study_create.count() == 0

        page.locator(".studio-cards button").filter(has_text="双人音频").click()
        podcast_create = page.get_by_role("dialog", name="生成双人深度播客")
        assert podcast_create.is_visible()
        assert podcast_create.get_by_text("PODCAST V4 // EDITORIAL ACTS", exact=True).is_visible()
        assert podcast_create.get_by_text(re.compile("成品再由本地 ASR 验收")).is_visible()
        assert page.get_by_label("目标时长").is_visible()
        page.screenshot(path="/tmp/sandevistan-read-podcast-v4-desktop.png")
        page.keyboard.press("Escape")
        assert podcast_create.count() == 0

        audio_guard = {"mode": "missing"}
        guarded_audio_provider = {
            "id": "qa-audio",
            "name": "QA Audio",
            "role": "audio",
            "kind": "sandevistan_audio",
            "base_url": "http://localhost:20810",
            "model": "voice-1.7b",
            "active": 1,
            "has_api_key": False,
            "capabilities": {},
            "config": {
                "compute_device": "gpu",
                "asr_model": "qwen3-asr-0.6b",
                "asr_compute_device": "gpu",
            },
        }

        def guarded_providers(route) -> None:
            providers = [] if audio_guard["mode"] == "missing" else [guarded_audio_provider]
            route.fulfill(status=200, content_type="application/json", body=json.dumps(providers))

        def guarded_status(route) -> None:
            ready = audio_guard["mode"] == "ready"
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "providers": {
                            "main": {"ok": True, "message": "QA Main"},
                            "audio": {"ok": ready, "message": "所选 ASR 设备不可用"},
                        }
                    }
                ),
            )

        guard_page = context.new_page()
        guard_page.set_viewport_size({"width": 1440, "height": 900})
        guard_page.route("**/api/providers", guarded_providers)
        guard_page.route("**/api/status", guarded_status)
        authenticate(guard_page)
        select_qa_notebook(guard_page)
        guard_page.wait_for_function("document.querySelectorAll('.source-row').length === 4")
        guarded_podcast = guard_page.locator(".studio-cards button").filter(has_text="双人音频")
        assert guarded_podcast.is_disabled()
        assert guard_page.get_by_text("请先配置并启用 AUDIO Provider", exact=True).is_visible()
        assert guard_page.locator(".studio-cards button").filter(has_text="Quiz 题库").is_enabled()
        guard_page.get_by_role("button", name="配置 AUDIO Provider").click()
        assert guard_page.get_by_role("dialog", name="Provider 配置").is_visible()
        assert guard_page.get_by_role("dialog", name="生成双人深度播客").count() == 0
        guard_page.screenshot(path="/tmp/sandevistan-read-podcast-disabled-desktop.png")
        guard_page.keyboard.press("Escape")

        audio_guard["mode"] = "unhealthy"
        guard_page.reload(wait_until="domcontentloaded")
        guard_page.wait_for_function("document.querySelectorAll('.source-row').length === 4")
        guard_page.get_by_text("所选 ASR 设备不可用", exact=True).wait_for()
        assert guard_page.locator(".studio-cards button").filter(has_text="双人音频").is_disabled()
        guard_page.screenshot(path="/tmp/sandevistan-read-podcast-unhealthy-desktop.png")

        audio_guard["mode"] = "ready"
        guard_page.reload(wait_until="domcontentloaded")
        guard_page.wait_for_function("document.querySelectorAll('.source-row').length === 4")
        guard_page.wait_for_function(
            "!Array.from(document.querySelectorAll('.studio-cards button')).find(button => button.textContent.includes('双人音频')).disabled"
        )
        guard_page.locator(".studio-cards button").filter(has_text="双人音频").click()
        assert guard_page.get_by_role("dialog", name="生成双人深度播客").is_visible()
        guard_page.close()

        page.locator(".artifact").filter(has_text="资料摘要").click()
        artifact_dialog = page.get_by_role("dialog", name="资料摘要")
        artifact_dialog.wait_for()
        artifact_dialog.locator(".citation-index summary").click()
        artifact_dialog.locator(".citation").first.click()
        citation_dialog = page.get_by_role("dialog", name=re.compile(r"引用 S\d+"))
        assert citation_dialog.is_visible()
        assert citation_dialog.evaluate("element => document.elementFromPoint(innerWidth - 20, innerHeight / 2)?.closest('[role=dialog]') === element")
        page.keyboard.press("Escape")
        assert citation_dialog.count() == 0 and artifact_dialog.is_visible()
        page.keyboard.press("Escape")
        assert artifact_dialog.count() == 0

        study_kind = {"value": "quiz"}
        citation = {"id": "S1", "filename": "qa-source.pdf", "locator": {"page": 1}, "quote": "可核验的原文证据。", "source_id": "qa-source"}

        def study_session(route) -> None:
            kind = study_kind["value"]
            item = (
                {
                    "id": "q1",
                    "question": "根据资料，哪个选项正确描述了核心关系？",
                    "options": ["正确关系", "相反关系", "无关关系", "过度概括"],
                    "hint": "留意因果方向。",
                    "difficulty": "medium",
                    "cognitive_level": "understand",
                    "learning_objective": "区分资料中的核心因果关系",
                }
                if kind == "quiz"
                else {
                    "id": "f1",
                    "front": "资料中的核心概念是什么？",
                    "back": "核心概念的准确说明。",
                    "explanation": "资料原文直接支持该定义。",
                    "difficulty": "medium",
                    "card_type": "concept",
                    "citation_details": [citation],
                }
            )
            route.fulfill(status=200, content_type="application/json", body=json.dumps({
                "id": f"qa-{kind}", "artifact_id": f"artifact-{kind}", "kind": kind,
                "mode": "all" if kind == "quiz" else "due", "status": "active", "items": [item],
                "progress": {"current": 0, "total": 1}, "created_at": "2026-08-31T00:00:00Z", "updated_at": "2026-08-31T00:00:00Z",
            }))

        def quiz_answer(route) -> None:
            item = {
                "id": "q1", "question": "根据资料，哪个选项正确描述了核心关系？",
                "options": ["正确关系", "相反关系", "无关关系", "过度概括"],
                "hint": "留意因果方向。", "difficulty": "medium", "cognitive_level": "understand",
                "learning_objective": "区分资料中的核心因果关系",
                "result": {"correct": True, "selected_index": 0, "answer_index": 0, "explanation": "资料明确给出了这一关系。", "citation_details": [citation]},
            }
            session = {"id": "qa-quiz", "artifact_id": "artifact-quiz", "kind": "quiz", "mode": "all", "status": "complete", "items": [item], "progress": {"current": 1, "total": 1}, "created_at": "2026-08-31T00:00:00Z", "updated_at": "2026-08-31T00:01:00Z"}
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"result": item["result"], "session": session}))

        def flashcard_review(route) -> None:
            item = {"id": "f1", "front": "资料中的核心概念是什么？", "back": "核心概念的准确说明。", "explanation": "资料原文直接支持该定义。", "difficulty": "medium", "card_type": "concept", "citation_details": [citation], "review": {"rating": "good", "due": "2026-09-07T00:00:00Z"}}
            session = {"id": "qa-flashcard", "artifact_id": "artifact-flashcard", "kind": "flashcard", "mode": "due", "status": "complete", "items": [item], "progress": {"current": 1, "total": 1}, "created_at": "2026-08-31T00:00:00Z", "updated_at": "2026-08-31T00:01:00Z"}
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"session": session}))

        page.route(re.compile(r".*/api/artifacts/[^/]+/study-sessions$"), study_session)
        page.route("**/api/study-sessions/qa-quiz/quiz-answer", quiz_answer)
        page.route("**/api/study-sessions/qa-flashcard/flashcard-review", flashcard_review)

        quiz_artifact = page.locator(".artifact").filter(has_text=re.compile("Quiz|测验|题库"))
        assert quiz_artifact.count() > 0
        quiz_artifact.first.click()
        quiz_dialog = page.get_by_role("dialog", name=re.compile("Quiz|测验|题库"))
        quiz_dialog.locator(".quiz-options button").first.click()
        quiz_dialog.get_by_role("button", name="提示").click()
        assert quiz_dialog.get_by_text("留意因果方向。").is_visible()
        quiz_dialog.get_by_role("button", name="检查答案").click()
        quiz_dialog.get_by_text("回答正确").wait_for()
        assert quiz_dialog.get_by_text("资料明确给出了这一关系。").is_visible()
        assert quiz_dialog.get_by_text("完成 · 1/1").is_visible()
        page.screenshot(path="/tmp/sandevistan-read-quiz-desktop.png")
        page.keyboard.press("Escape")

        study_kind["value"] = "flashcard"
        page.locator(".artifact").filter(has_text="闪卡组").click()
        card = page.locator(".flash-card")
        card.wait_for()
        before = card.inner_text()
        card.click()
        assert card.inner_text() != before
        page.get_by_role("button", name="良好").click()
        page.get_by_text("本轮复习完成").wait_for()
        page.screenshot(path="/tmp/sandevistan-read-flashcard-desktop.png")
        page.keyboard.press("Escape")

        page.goto(f"{BASE_URL}/#jobs", wait_until="domcontentloaded")
        page.get_by_role("heading", name="任务记录").wait_for()
        assert page.get_by_role("heading", name="任务记录").is_visible()
        page.locator(".record-open").first.click()
        job_dialog = page.get_by_role("dialog", name=re.compile("任务详情"))
        assert job_dialog.is_visible()
        assert job_dialog.get_by_text("阶段事件").is_visible()
        page.keyboard.press("Escape")
        assert job_dialog.count() == 0
        page.screenshot(path="/tmp/sandevistan-read-ui-jobs-desktop.png")

        notebook_items = [
            {"id": "nb_batch_a", "title": "Batch QA A", "description": "第一项", "state": "active", "source_count": 2, "source_bytes": 2048, "artifact_count": 1, "active_jobs": 0},
            {"id": "nb_batch_b", "title": "Batch QA B", "description": "第二项", "state": "active", "source_count": 3, "source_bytes": 4096, "artifact_count": 2, "active_jobs": 1},
            {"id": "nb_batch_busy", "title": "Batch QA Deleting", "description": "不可选择", "state": "deleting", "source_count": 1, "source_bytes": 1024, "artifact_count": 0, "active_jobs": 0},
        ]

        def notebook_management(route) -> None:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"items": notebook_items, "page": 1, "page_size": 20, "total": len(notebook_items), "pages": 1}),
            )

        page.route("**/api/notebook-management?*", notebook_management)
        page.goto(f"{BASE_URL}/#notebooks", wait_until="domcontentloaded")
        page.get_by_role("button", name="新建 Notebook").wait_for()
        page.get_by_role("button", name="新建 Notebook").click()
        create_dialog = page.get_by_role("dialog", name="新建 Notebook")
        assert create_dialog.is_visible()
        page.keyboard.press("Escape")
        assert create_dialog.count() == 0

        first_batch_checkbox = page.get_by_role("checkbox", name="选择 Batch QA A")
        second_batch_checkbox = page.get_by_role("checkbox", name="选择 Batch QA B")
        unavailable_checkbox = page.get_by_role("checkbox", name="选择 Batch QA Deleting")
        assert unavailable_checkbox.is_disabled()
        first_batch_checkbox.check()
        second_batch_checkbox.check()
        assert page.get_by_role("button", name="删除 2", exact=True).is_enabled()
        assert page.get_by_role("checkbox", name="选择本页可删除 Notebook").is_checked()
        page.get_by_role("button", name="删除 2", exact=True).click()
        batch_dialog = page.get_by_role("dialog", name="批量删除 2 个 Notebook")
        assert batch_dialog.is_visible()
        assert batch_dialog.get_by_text("Batch QA A", exact=True).is_visible()
        assert batch_dialog.get_by_text("Batch QA B", exact=True).is_visible()
        batch_confirm = batch_dialog.get_by_role("button", name="删除 2 个 Notebook")
        assert batch_confirm.is_disabled()
        batch_dialog.locator("input").fill("批量删除")
        assert batch_confirm.is_enabled()
        page.screenshot(path="/tmp/sandevistan-read-ui-notebooks-batch-desktop.png")
        page.keyboard.press("Escape")
        assert batch_dialog.count() == 0
        page.get_by_role("checkbox", name="选择本页可删除 Notebook").uncheck()

        first_row = page.locator(".notebook-grid.manage-row").first
        notebook_title = first_row.locator(".record-title b").inner_text()
        first_row.get_by_role("button", name=f"删除 {notebook_title}").click()
        delete_dialog = page.get_by_role("dialog", name=f"删除 {notebook_title}")
        delete_dialog.locator("input").fill(notebook_title)
        assert delete_dialog.get_by_role("button", name="删除 Notebook").is_enabled()
        delete_dialog.get_by_role("button", name="取消").click()
        first_row.get_by_role("button", name=f"删除 {notebook_title}").click()
        delete_dialog = page.get_by_role("dialog", name=f"删除 {notebook_title}")
        assert delete_dialog.locator("input").input_value() == ""
        assert delete_dialog.get_by_role("button", name="删除 Notebook").is_disabled()
        page.keyboard.press("Escape")

        tablet = context.new_page()
        tablet.set_viewport_size({"width": 900, "height": 900})
        audio_guard["mode"] = "missing"
        tablet.route("**/api/providers", guarded_providers)
        tablet.route("**/api/status", guarded_status)
        tablet.goto(f"{BASE_URL}/#workspace", wait_until="domcontentloaded")
        tablet.locator(".notebook-menu").wait_for()
        assert tablet.locator(".notebook-menu").is_visible()
        assert not tablet.locator(".workspace > .studio").is_visible()
        tablet.locator(".tablet-studio-trigger").click()
        tablet_studio = tablet.get_by_role("dialog", name="Studio")
        assert tablet_studio.is_visible()
        assert tablet_studio.locator(".studio-cards button").count() == 4
        assert tablet_studio.locator(".studio-cards button").filter(has_text="双人音频").is_disabled()
        assert_no_horizontal_overflow(tablet)
        tablet.screenshot(path="/tmp/sandevistan-read-podcast-disabled-tablet.png")
        tablet_studio.get_by_role("button", name="配置 AUDIO Provider").click()
        assert tablet_studio.count() == 0
        assert tablet.get_by_role("dialog", name="Provider 配置").is_visible()
        tablet.keyboard.press("Escape")

        mobile = context.new_page()
        mobile.set_viewport_size({"width": 390, "height": 844})
        mobile.route("**/api/providers", guarded_providers)
        mobile.route("**/api/status", guarded_status)
        mobile.goto(f"{BASE_URL}/#workspace", wait_until="domcontentloaded")
        mobile.locator(".notebook-menu").wait_for()
        assert_no_horizontal_overflow(mobile)
        assert mobile.locator(".notebook-menu").is_visible()
        mobile.get_by_role("button", name=re.compile("资料")).click()
        assert mobile.locator(".sources").is_visible()
        mobile.get_by_role("button", name="Studio", exact=True).click()
        assert mobile.locator(".workspace > .studio").is_visible()
        assert mobile.locator(".workspace > .studio .studio-cards button").filter(has_text="双人音频").is_disabled()
        assert mobile.locator(".workspace > .studio").get_by_role("button", name="配置 AUDIO Provider").is_visible()
        mobile.screenshot(path="/tmp/sandevistan-read-podcast-disabled-mobile.png")

        mobile.route("**/api/providers/inspect", inspect_provider)
        mobile.get_by_role("button", name="设置").click()
        mobile_settings = mobile.get_by_role("dialog", name="Provider 配置")
        mobile.screenshot(path="/tmp/sandevistan-read-provider-roles-mobile.png")
        mobile_settings.get_by_role("button", name="管理 AUDIO").click()
        mobile_settings.get_by_role("button", name="添加 AUDIO Provider").click()
        assert mobile_settings.get_by_role("heading", name="添加 Provider").is_visible()
        assert mobile_settings.get_by_label("角色").is_disabled()
        mobile_settings.get_by_label("名称").fill("QA Audio Mobile")
        mobile_settings.get_by_label("服务地址").fill("http://localhost:20810")
        mobile_settings.get_by_role("button", name="连接并读取模型").click()
        mobile_settings.get_by_role("heading", name="ASR 验收").wait_for()
        mobile_settings.get_by_role("heading", name="ASR 验收").scroll_into_view_if_needed()
        assert "Qwen3 ASR 0.6B" in mobile_settings.get_by_role("combobox", name=re.compile("ASR 模型")).inner_text()
        assert_no_horizontal_overflow(mobile)
        mobile.screenshot(path="/tmp/sandevistan-read-audio-provider-mobile.png")
        mobile_settings.get_by_role("button", name="返回角色配置").click()
        mobile_discard = mobile.get_by_role("dialog", name="放弃未保存的 Provider 配置？")
        mobile_discard.get_by_role("button", name="放弃修改").click()
        mobile_settings.get_by_role("button", name="返回角色概览").click()
        mobile_settings.get_by_role("button", name="管理 MAIN").click()
        mobile_settings.get_by_role("button", name="添加 MAIN Provider").click()
        mobile_temperature = mobile_settings.get_by_label("Temperature 覆盖")
        assert mobile_temperature.is_visible()
        mobile_temperature.scroll_into_view_if_needed()
        assert_no_horizontal_overflow(mobile)
        panel_metrics = mobile_settings.evaluate("element => ({scroll: element.scrollWidth, client: element.clientWidth})")
        assert panel_metrics["scroll"] <= panel_metrics["client"], panel_metrics
        mobile.screenshot(path="/tmp/sandevistan-read-provider-temperature-mobile.png")
        mobile_temperature.fill("1")
        mobile.keyboard.press("Escape")
        mobile_discard = mobile.get_by_role("dialog", name="放弃未保存的 Provider 配置？")
        assert mobile_discard.is_visible()
        mobile_discard.get_by_role("button", name="放弃修改").click()
        assert mobile_settings.count() == 0

        mobile.goto(f"{BASE_URL}/#jobs", wait_until="domcontentloaded")
        mobile.wait_for_function(
            "document.querySelector('.manage-table')?.getAttribute('aria-busy') === 'false'"
        )
        assert mobile.locator(".record-open").count() > 0
        assert_no_horizontal_overflow(mobile)
        mobile.screenshot(path="/tmp/sandevistan-read-ui-jobs-mobile.png")

        mobile.route("**/api/notebook-management?*", notebook_management)
        mobile.goto(f"{BASE_URL}/#notebooks", wait_until="domcontentloaded")
        mobile.wait_for_function(
            "document.querySelector('.manage-table')?.getAttribute('aria-busy') === 'false' && "
            "document.querySelectorAll('.notebook-grid.manage-row').length === 3"
        )
        assert mobile.get_by_role("button", name="打开 Batch QA A").is_visible()
        assert mobile.locator(".notebook-grid.manage-row").first.get_by_text("2 FILES", exact=True).is_visible()
        assert mobile.locator(".state-pill.active").first.is_visible()
        mobile.get_by_role("checkbox", name="选择 Batch QA A").check()
        mobile.get_by_role("checkbox", name="选择 Batch QA B").check()
        mobile.get_by_role("button", name="删除 2", exact=True).click()
        mobile_batch_dialog = mobile.get_by_role("dialog", name="批量删除 2 个 Notebook")
        assert mobile_batch_dialog.is_visible()
        assert_no_horizontal_overflow(mobile)
        mobile.screenshot(path="/tmp/sandevistan-read-ui-notebooks-batch-mobile.png")
        mobile.keyboard.press("Escape")
        assert mobile_batch_dialog.count() == 0
        browser.close()

    if console_errors:
        raise RuntimeError("Browser console errors: " + " | ".join(console_errors))
    print(json.dumps({"status": "passed", "viewports": ["1440x900", "900x900", "390x844"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
