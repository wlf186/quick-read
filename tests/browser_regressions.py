"""Deterministic UI regressions: every API request is intercepted, including writes."""
from __future__ import annotations

import json
from urllib.parse import urlsplit

from playwright.sync_api import Browser, Page, expect, sync_playwright

BASE_URL = "http://127.0.0.1:20830"


class Fixture:
    def __init__(self, page: Page):
        self.pending = []
        self.requests = []
        self.cards = ["f1", "f2"]
        self.reviews = {}
        self.errors = []
        self.console_errors = []
        self.hold_history = False
        self.pending_history = []
        self.hold_messages = False
        self.pending_messages = []
        self.show_job = False
        self.source_state = "ready"
        self.source_overrides = {}
        self.job_state = "queued"
        self.artifact = None
        self.messages = None
        self.provider = None
        self.qualified_summary = False
        page.on("pageerror", lambda error: self.errors.append(str(error)))
        page.on("console", lambda message: self.console_errors.append(message.text) if message.type == "error" else None)
        page.route("**/auth/status", lambda route: route.fulfill(json={"required": False, "authenticated": True}))
        page.route("**/api/**", self.route)

    def session(self):
        return {"id": "study", "artifact_id": "cards", "kind": "flashcard", "mode": "due",
                "status": "complete" if all(key in self.reviews for key in self.cards) else "active",
                "items": [{"id": key, "front": key, "back": "answer", **({"review": self.reviews[key]} if key in self.reviews else {})} for key in self.cards],
                "progress": {"current": sum(key in self.reviews for key in self.cards), "total": len(self.cards)}}

    def route(self, route):
        request = route.request
        path = urlsplit(request.url).path.removeprefix("/api")
        body = json.loads(request.post_data or "{}")
        self.requests.append((request.method, path, body))
        notebooks = [{"id": key, "title": f"Audit {key.upper()}", "description": ""} for key in ("a", "b")]
        artifact = self.artifact or {"id": "cards", "type": "flashcard", "title": "闪卡组", "status": "ready", "payload": {}, "citations": []}
        job = {"id": "ingest", "kind": "ingest", "notebook_id": "a", "notebook_title": "Audit A", "display_name": "文档解析",
               "state": self.job_state, "stage": "已取消" if self.job_state == "cancelled" else "等待执行", "stage_code": self.job_state,
               "progress": 1 if self.job_state == "cancelled" else 0, "created_at": "2026-09-01T00:00:00Z", "updated_at": "2026-09-01T00:00:00Z",
               "eta": {"status": "learning", "sample_count": 0, "queue_position": 1}}
        if path.endswith("/chat"):
            self.pending.append(route)
            return
        if path == "/notebooks":
            result = notebooks
        elif path == "/providers":
            result = [self.provider] if self.provider else []
        elif path == "/provider-roles":
            result = []
        elif path == "/providers/inspect":
            result = {"status":"passed","connection_ok":True,"models":[],"capabilities":{},"latency_ms":1,"activation_eligible":True}
        elif path == "/providers/context-preview":
            from sandevistan_read.context_budget import TokenLimits, plan_context
            limits=TokenLimits.from_provider({"config":body["config"]})
            result={"strategy":"conservative","plans":[plan_context(limits,kind).as_dict() for kind in ('summary','chat','quiz','flashcard','podcast')],"saved_plans":[],"basis":"按每段约 1000 tokens 估算","assumptions":"预算上限不是质量保证。","candidate_segments":None,"material_tokens":None}
            kinds = ('summary', 'chat', 'quiz', 'flashcard', 'podcast')
            matched = self.qualified_summary and body['config'].get('context_window_tokens') == 30720
            result['strategies'] = {kind: 'balanced' if matched and kind == 'summary' else 'conservative' for kind in kinds}
            result['strategy_reasons'] = {kind: '当前 Provider、配置与功能已通过对照验收' if matched and kind == 'summary' else '当前 Provider、配置与功能尚无匹配的质量资格' for kind in kinds}
        elif path == "/settings/image-processing":
            result = {"mode": "process", "processors": ["ocr"]}
        elif path == "/status":
            result = {"version": "9.8.7-fixture", "providers": {}}
        elif path == "/jobs":
            result = {"items": [job] if self.show_job else [], "page": 1, "page_size": 100, "total": int(self.show_job), "pages": 1}
        elif path == "/jobs/ingest/cancel":
            self.job_state = "cancelled"
            self.source_state = "failed"
            result = {"ok": True}
        elif path == "/workspace-state":
            active = [job] if self.show_job and self.job_state == "queued" else []
            result = {"versions": {"notebook": "a", "sources": self.source_state, "artifacts": "cards"},
                      "active_jobs": active, "failed_jobs": [], "has_active_jobs": bool(active)}
        elif path in ("/notebooks/a", "/notebooks/b"):
            key = path[-1]
            result = {**next(item for item in notebooks if item["id"] == key), "sources": [
                {"id": f"{key}-source", "filename": f"Audit {key.upper()}.txt", "state": self.source_state,
                 "selected": 1 if self.source_state == "ready" else 0, "page_count": 1, "error": "解析已取消" if self.source_state == "failed" else None, **self.source_overrides}]}
        elif path.endswith("/conversations"):
            if self.hold_history:
                self.pending_history.append(route)
                return
            result = [{"id": "history"}] if self.hold_messages or self.messages is not None else []
        elif path == "/conversations/history/messages":
            if self.hold_messages:
                self.pending_messages.append(route)
                return
            result = self.messages or []
        elif path.endswith("/artifacts"):
            result = [artifact]
        elif path == "/artifacts/cards":
            result = artifact
        elif path in ("/artifacts/cards/study-sessions", "/study-sessions/study"):
            result = self.session()
        elif path.startswith("/artifacts/cards/flashcards/") and request.method == "DELETE":
            self.cards.remove(path.rsplit("/", 1)[-1])
            route.fulfill(status=204)
            return
        elif path == "/study-sessions/study/flashcard-review":
            self.reviews[body["item_id"]] = {"rating": body["rating"]}
            result = {"session": self.session()}
        else:
            raise AssertionError(f"Unexpected API request: {request.method} {path}")
        route.fulfill(json=result)

    def reply(self, index: int, text: str, conversation: str, status: int = 200):
        self.pending[index].fulfill(status=status, json={"id": f"answer-{index}", "conversation_id": conversation,
                                                       "content": text, "citations": [], "detail": text})


def select_notebook(page: Page, key: str, wait: bool = True):
    page.locator(".notebook-switch").click()
    page.locator(".notebook-options > button").filter(has_text=f"Audit {key.upper()}").click()
    if wait:
        expect(page.locator(".notebook-switch b")).to_have_text(f"Audit {key.upper()}")


def settle(page: Page):
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")


def assert_layout(page: Page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert page.locator("vite-error-overlay").count() == 0


def send(page: Page, fixture: Fixture, question: str):
    count = len(fixture.pending)
    page.get_by_label("向已选资料提问").fill(question)
    page.get_by_label("向已选资料提问").press("Enter")
    expect(page.locator(".message.user").last).to_contain_text(question)
    # A DOM assertion yields to Playwright's network event loop.
    expect(page.locator(".thinking")).to_be_visible()
    assert len(fixture.pending) == count + 1


def run_core_regressions(browser: Browser):
    for width, height in ((1440, 900), (390, 844)):
        context = browser.new_context(viewport={"width": width, "height": height}, reduced_motion="reduce")
        page = context.new_page()
        page.set_default_timeout(8000)
        fixture = Fixture(page)
        page.goto(BASE_URL)
        expect(page.get_by_label("向已选资料提问")).to_be_enabled()
        expect(page.locator("footer")).to_contain_text("BUILD 9.8.7-fixture")
        send(page, fixture, "question A")
        select_notebook(page, "b")
        fixture.reply(0, "OLD ANSWER A", "conversation-a")
        expect(page.locator(".thinking")).not_to_be_visible()
        expect(page.locator(".messages")).not_to_contain_text("OLD ANSWER A")
        send(page, fixture, "question B")
        assert fixture.requests[-1][1] == "/notebooks/b/chat"
        assert fixture.requests[-1][2].get("conversation_id") is None
        fixture.reply(1, "Correct answer B", "conversation-b")
        expect(page.locator(".messages")).to_contain_text("Correct answer B")
        page.screenshot(path=f"/tmp/quick-read-chat-fixed-{width}.png")

        # The old failure must not restore its question or clear the new request's busy state.
        select_notebook(page, "a")
        send(page, fixture, "old request")
        select_notebook(page, "b")
        send(page, fixture, "new request")
        fixture.reply(2, "OLD FAILURE", "conversation-a", status=409)
        settle(page)
        expect(page.locator(".thinking")).to_be_visible()
        expect(page.get_by_label("向已选资料提问")).to_have_value("")
        expect(page.locator(".toast-error")).not_to_be_visible()
        fixture.reply(3, "New response", "conversation-b")
        expect(page.locator(".messages")).to_contain_text("New response")

        # A new conversation in the same notebook invalidates both success and finally.
        send(page, fixture, "discarded conversation")
        page.get_by_role("button", name="新对话", exact=True).click()
        send(page, fixture, "fresh conversation")
        assert fixture.requests[-1][2].get("conversation_id") is None
        fixture.reply(4, "DISCARDED RESPONSE", "discarded")
        settle(page)
        expect(page.locator(".thinking")).to_be_visible()
        expect(page.locator(".messages")).not_to_contain_text("DISCARDED RESPONSE")
        fixture.reply(5, "Fresh response", "fresh")
        expect(page.locator(".messages")).to_contain_text("Fresh response")
        send(page, fixture, "follow up")
        assert fixture.requests[-1][2]["conversation_id"] == "fresh"
        fixture.reply(6, "Follow up response", "fresh")
        expect(page.locator(".messages")).to_contain_text("Follow up response")

        # A -> B -> A must be treated as a new view even though the notebook ID matches.
        select_notebook(page, "a")
        send(page, fixture, "round trip request")
        select_notebook(page, "b")
        select_notebook(page, "a")
        page.get_by_label("向已选资料提问").fill("keep this draft")
        fixture.reply(7, "ROUND TRIP OLD RESPONSE", "old-a")
        settle(page)
        expect(page.locator(".messages")).not_to_contain_text("ROUND TRIP OLD RESPONSE")
        expect(page.get_by_label("向已选资料提问")).to_have_value("keep this draft")

        # Do not allow an Enter submission before the target history has loaded.
        fixture.hold_history = True
        select_notebook(page, "b", wait=False)
        expect(page.get_by_label("向已选资料提问")).to_be_disabled()
        settle(page)
        page.keyboard.press("Enter")
        assert len(fixture.pending) == 8
        fixture.hold_history = False
        for route in fixture.pending_history:
            route.fulfill(json=[])
        expect(page.get_by_label("向已选资料提问")).to_be_enabled()
        assert_layout(page)

        # Two overlapping history loads: the slower one must not erase a new question.
        fixture.hold_messages = True
        select_notebook(page, "a")
        expect(page.get_by_label("向已选资料提问")).to_be_disabled()
        with page.expect_request("**/api/conversations/history/messages"):
            page.evaluate("window.dispatchEvent(new Event('focus'))")
        settle(page)
        assert len(fixture.pending_messages) == 2
        fixture.pending_messages[1].fulfill(json=[])
        expect(page.get_by_label("向已选资料提问")).to_be_enabled()
        send(page, fixture, "question after history")
        fixture.pending_messages[0].fulfill(json=[{"id": "stale", "role": "assistant", "content": "STALE HISTORY"}])
        settle(page)
        expect(page.locator(".message.user")).to_contain_text("question after history")
        expect(page.locator(".messages")).not_to_contain_text("STALE HISTORY")
        fixture.hold_messages = False
        fixture.reply(8, "Current response", "current")
        expect(page.locator(".messages")).to_contain_text("Current response")

        # The drawer must accept the server's smaller queue and its completed state.
        if width < 600:
            page.locator(".workspace-tabs button").filter(has_text="Studio").click()
        page.locator(".artifact").filter(has_text="闪卡组").click()
        drawer = page.get_by_role("dialog", name="闪卡组")
        expect(drawer.locator(".flash-card")).to_contain_text("f1")
        drawer.get_by_role("button", name="移除", exact=True).click()
        expect(drawer.locator(".flash-card")).to_contain_text("f2")
        page.keyboard.press("Escape")
        expect(drawer).not_to_be_visible()
        page.locator(".artifact").filter(has_text="闪卡组").click()
        expect(drawer.locator(".flash-card")).to_contain_text("f2")
        drawer.locator(".flash-card").click()
        drawer.get_by_role("button", name="良好", exact=True).click()
        expect(drawer.get_by_text("本轮复习完成")).to_be_visible()
        assert_layout(page)
        page.screenshot(path=f"/tmp/quick-read-flashcard-fixed-{width}.png")
        drawer.get_by_role("button", name="移除", exact=True).click()
        expect(drawer.get_by_text("当前队列没有待学习内容")).to_be_visible()
        drawer.get_by_role("button", name="学习全部", exact=True).click()
        expect(drawer.get_by_text("当前队列没有待学习内容")).to_be_visible()
        page.keyboard.press("Escape")

        # Confirm a mocked cancellation, then verify the source's terminal explanation.
        fixture.show_job = True
        fixture.source_state = "queued"
        page.evaluate("window.dispatchEvent(new Event('focus'))")
        expect(page.locator(".source-row")).to_contain_text("QUEUED")
        page.goto(f"{BASE_URL}/#jobs")
        page.get_by_role("button", name="终止 文档解析", exact=True).click()
        confirmation = page.get_by_role("dialog", name="终止“文档解析”")
        expect(confirmation).to_be_visible()
        confirmation.get_by_role("button", name="终止任务", exact=True).click()
        expect(page.locator(".manage-row.state-cancelled")).to_be_visible()
        page.evaluate("location.hash='workspace'")
        if width < 600:
            page.locator(".workspace-tabs button").filter(has_text="资料").click()
        expect(page.locator(".source-row")).to_contain_text("解析已取消")
        assert_layout(page)
        page.screenshot(path=f"/tmp/quick-read-cancel-fixed-{width}.png")
        assert not fixture.errors
        assert all("409" in message for message in fixture.console_errors), fixture.console_errors
        context.close()


def run_generation_regressions(browser: Browser) -> None:
    for width, height, asr_ok in ((1440, 900, False), (390, 844, False), (1440, 900, True), (390, 844, True)):
        context = browser.new_context(viewport={"width": width, "height": height}, reduced_motion="reduce")
        page = context.new_page()
        fixture = Fixture(page)
        coverage={"candidate_segments":100,"selected_segments":30,"sent_segments":20,"partially_sent_segments":2,"cited_segments":6,"sources":[{"source_id":"a-source","filename":"Audit A.txt","regions_total":10,"regions_sent":4,"unsent_regions":["页区间：91–100"]}],"plan":{"total_token_limit":100000,"evidence_tokens":20000,"limiting_factor":"有效上下文与输出预算"}}
        fixture.messages = [{"id": "answer", "role": "assistant", "content": "保留有依据的中文回答。", "metadata": {"degraded": True, "warnings": [{"code": "source_formula_unreadable", "stage": "answer", "message": "公式提取不完整，请核对原文件。"}]}}]
        fixture.artifact = {"id": "cards", "type": "podcast", "title": "降级播客", "status": "partial", "media_url": "", "citations": [],
                            "payload": {"version": 4, "degraded": True, "turns": [], "chapters": [], "duration": {"target_minutes": 5, "actual_seconds": 240},
                                        "fact_review": {"status":"partial","total":10,"reviewed":3,"supported":2,"contradicted":1,"uncertain":0}, "source_contributions":[{"source_id":"omitted","included":False,"reason":"本集未涵盖"}], "quality": {"passed": False}, "audio_quality": {"passed": asr_ok, "speaker_method":"word_speaker","speaker_coverage":1.0, "metric": "cer", "error_rate": 0.02 if asr_ok else 0.12, "speaker_alignment": 0.90, "silence_outliers": 1, "duration": {"passed": False}},
                                        "warnings": [{"code": "audio_duration", "stage": "audio", "message": "目标 5 分钟，实际 4 分钟。"}]}}
        coverage['stages']={'act_draft':{'sent_segments':18,'source_tokens':9000},'targeted_repair':{'sent_segments':5,'source_tokens':2000}}
        fixture.artifact['payload']['context_usage']={'coverage':coverage}
        page.goto(BASE_URL)
        expect(page.get_by_label("向已选资料提问")).to_be_enabled()
        expect(page.locator(".messages").get_by_label("生成结果说明")).to_contain_text("公式提取不完整，请核对原文件")
        if width < 600:
            page.locator(".workspace-tabs button").filter(has_text="Studio").click()
        page.locator(".artifact").filter(has_text="降级播客").click()
        drawer = page.get_by_role("dialog", name="降级播客")
        expect(drawer.get_by_label("生成结果说明")).to_contain_text("目标 5 分钟，实际 4 分钟")
        expect(drawer.locator(".podcast-meta span").filter(has_text="AUDIO").locator("b")).to_have_text("VERIFIED" if asr_ok else "UNVERIFIED")
        expect(drawer.locator(".podcast-meta")).to_contain_text("偏离目标，仅作提示")
        expect(drawer.locator(".podcast-meta")).to_contain_text("2.0%" if asr_ok else "12.0%")
        expect(drawer.locator(".podcast-meta")).not_to_contain_text("PASSED")
        expect(drawer).to_contain_text("原文核验 3/10 项")
        expect(drawer).to_contain_text("本集未涵盖 1 份已选资料")
        drawer.locator('.coverage-details > summary').click()
        expect(drawer.locator('.coverage-details')).to_contain_text('成功调用完整送入 20 段')
        expect(drawer.locator('.coverage-details')).to_contain_text('章节写作 18 段 · 技术恢复 5 段')
        drawer.locator('.coverage-details details > summary').click()
        expect(drawer.locator('.coverage-details')).to_contain_text('页区间：91–100')
        assert_layout(page)
        page.screenshot(path=f"/tmp/quick-read-generation-warnings-{width}.png")
        page.keyboard.press("Escape")
        expect(drawer).not_to_be_visible()
        assert not fixture.errors and not fixture.console_errors
        context.close()


def run_context_regressions(browser: Browser) -> None:
    for width in (1440,390):
        context=browser.new_context(viewport={'width':width,'height':900})
        page=context.new_page();fixture=Fixture(page)
        fixture.provider={'id':'main','name':'Fixture Main','role':'main','kind':'openai','base_url':'https://example.invalid','model':'fixture','active':True,'selected':True,'config':{'context_window_tokens':30720,'max_output_tokens':4096},'capabilities':{}}
        page.goto(BASE_URL)
        page.get_by_role('button',name='设置',exact=True).click()
        page.get_by_role('button',name='管理 MAIN',exact=True).click()
        page.get_by_role('button',name='编辑',exact=True).click()
        panel=page.get_by_label('上下文容量预估')
        expect(panel.locator('tbody tr')).to_have_count(5)
        expect(panel).to_contain_text('分批选材后综合')
        expect(panel).to_contain_text('先完成短稿，再逐章深化')
        expect(panel).to_contain_text('个核心章节')
        if width == 390:
            expect(panel.get_by_text('左右滑动查看完整容量对照')).to_be_visible()
            table_region = panel.get_by_role('region', name='功能容量对照表，可横向滚动')
            assert table_region.evaluate('(element) => element.scrollWidth > element.clientWidth')
            assert panel.locator('tbody tr').first.locator('td').nth(2).evaluate('(element) => element.clientWidth >= 160')
            table_region.focus()
            page.keyboard.press('ArrowRight')
            expect(table_region).to_be_focused()
            page.wait_for_function('document.querySelector(".context-capacity-table").scrollLeft > 0')
            table_region.evaluate('(element) => element.scrollLeft = 0')
        expect(panel).to_contain_text('当前使用保守策略')
        original=panel.locator('tbody tr').first.text_content()
        expect(panel).to_contain_text('当前 Provider、配置与功能尚无匹配的质量资格')
        page.get_by_label('上下文窗口覆盖（tokens）').fill('1000000')
        page.get_by_label('最大输出覆盖（tokens）').fill('384000')
        expect(panel.locator('tbody tr').first).not_to_have_text(original)
        expect(panel.locator('tbody tr').first).to_contain_text('1,250,000')
        expect(panel).not_to_contain_text('长任务最高 30 万')
        expect(panel.locator('tbody tr').first).to_contain_text('分区预读后综合')
        expect(panel.locator('tbody tr').first).to_contain_text('条笔记')
        expect(panel.locator('tbody tr').first).to_contain_text('摘要容量约')
        page.get_by_label('上下文窗口覆盖（tokens）').fill('30720')
        page.get_by_label('最大输出覆盖（tokens）').fill('4096')
        expect(panel.locator('tbody tr').first).to_have_text(original)
        fixture.qualified_summary = True
        page.get_by_label('上下文窗口覆盖（tokens）').fill('30721')
        page.get_by_label('上下文窗口覆盖（tokens）').fill('30720')
        expect(panel.locator('tbody tr').first).to_contain_text('已通过对照验收')
        expect(panel.locator('tbody tr').nth(1)).to_contain_text('尚无匹配的质量资格')
        page.get_by_label('上下文窗口覆盖（tokens）').fill('30722')
        expect(panel.locator('tbody tr').first).to_contain_text('尚无匹配的质量资格')
        assert_layout(page)
        panel.scroll_into_view_if_needed()
        page.screenshot(path=f'/tmp/quick-read-context-capacity-{width}.png')
        if width == 390:
            panel.locator('.context-capacity-table').evaluate('(element) => element.scrollLeft = 300')
            page.screenshot(path='/tmp/quick-read-context-capacity-390-details.png')
        assert not fixture.errors and not fixture.console_errors
        assert not any(method=='PATCH' for method,_,_ in fixture.requests)
        context.close()


def run_summary_coverage_regressions(browser: Browser) -> None:
    for width in (1440, 390):
        context = browser.new_context(viewport={'width': width, 'height': 900})
        page = context.new_page()
        fixture = Fixture(page)
        coverage = {'candidate_segments': 100, 'selected_segments': 30, 'sent_segments': 20,
                    'partially_sent_segments': 0, 'cited_segments': 3,
                    'plan': {'total_token_limit': 300000, 'evidence_tokens': 20000, 'limiting_factor': '任务累计预算'},
                    'stages': {'context_prepare': {'sent_segments': 20}, 'summary': {'sent_segments': 10}},
                    'audit': {'supported': 3, 'unsupported': 1, 'unreviewed': 2}}
        fixture.artifact = {'id': 'cards', 'type': 'summary', 'title': '核验摘要', 'status': 'partial',
                            'citations': [], 'payload': {'content': '已保留原文支持的要点。',
                                                        'context_usage': {'coverage': coverage}}}
        page.goto(BASE_URL)
        if width < 600:
            page.locator('.workspace-tabs button').filter(has_text='Studio').click()
        page.locator('.artifact').filter(has_text='核验摘要').click()
        drawer = page.get_by_role('dialog', name='核验摘要')
        drawer.locator('.coverage-details > summary').click()
        expect(drawer).to_contain_text('预读 20 段 · 最终生成 10 段')
        expect(drawer).to_contain_text('模型核验支持 3 项 · 不支持 1 项 · 未完成 2 项')
        assert_layout(page)
        page.screenshot(path=f'/tmp/quick-read-summary-coverage-{width}.png')
        page.keyboard.press('Escape')
        expect(drawer).not_to_be_visible()
        assert not fixture.errors and not fixture.console_errors
        context.close()


def run_import_regressions(browser: Browser) -> None:
    for width, height in ((1440, 900), (390, 844)):
        context = browser.new_context(viewport={"width": width, "height": height}, reduced_motion="reduce")
        page = context.new_page()
        fixture = Fixture(page)
        fixture.source_overrides = {"filename": "资产.xlsx", "page_count": 10, "metadata": {
            "locator_unit": "sheet", "warnings": [{"message": "部分公式没有保存计算结果。"}]}}
        page.goto(BASE_URL, wait_until="domcontentloaded")
        if width < 600:
            page.get_by_role('button', name='资料 1', exact=True).click()
        expect(page.locator('.source').first).to_be_visible()
        expect(page.locator('.source').first).to_contain_text('10 SHEETS')
        expect(page.locator('.source .warning').first).to_contain_text('部分公式')
        upload = page.locator('.upload-zone input')
        assert '.xlsx' in upload.get_attribute('accept')
        upload.set_input_files({"name": "fixture.xlsx", "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "buffer": b"fixture"})
        dialog = page.get_by_role('dialog', name='确认上传')
        expect(dialog).to_be_visible()
        expect(dialog).to_contain_text('确认接入 1 份资料')
        page.keyboard.press('Escape')
        expect(dialog).not_to_be_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not fixture.errors and not fixture.console_errors
        assert not any(method == 'POST' and path.endswith('/sources') for method, path, _ in fixture.requests)
        page.screenshot(path=f'/tmp/quick-read-office-{width}.png', full_page=True)
        context.close()


if __name__ == "__main__":
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
        run_core_regressions(browser)
        run_generation_regressions(browser)
        run_context_regressions(browser)
        run_summary_coverage_regressions(browser)
        run_import_regressions(browser)
        browser.close()
    print("Core UI regressions passed")


def run_delivery_regressions(browser: Browser) -> None:
    """Mocked responses only: expand rating details and inspect a script-only artifact."""
    for width, review_status in [(w, status) for w in (1440,390) for status in (None,"complete","partial","unavailable")]:
        context = browser.new_context(viewport={'width': width, 'height': 900})
        page = context.new_page()
        fixture = Fixture(page)
        quality = {'level':'needs_review','reviewed_units':1,'total_units':2,
                   'issues':[{'unit':'turn_1','code':'evidence_unconfirmed','message':'该轮原文支持待核实。'}]}
        fixture.messages = [{'role':'assistant','content':'保留原始回答。','metadata':{'quality_assessment':quality,'delivery_status':'full'}}]
        fixture.artifact = {'id':'cards','type':'podcast','title':'仅脚本示例','status':'partial','citations':[],
            'payload':{'version':4,'delivery_status':'script_only','quality_assessment':quality,'duration':{'target_minutes':5},
                       'turns':[{'id':'turn_1','speaker':'HOST_A','text':'这是保留的脚本。','quality_issues':quality['issues']}],'chapters':[]}}
        if review_status:
            fixture.artifact["payload"].update(generation_mode="expanded" if review_status=="complete" else "complete_short", narrative_status="complete" if review_status=="complete" else "incomplete" if review_status=="partial" else "unverified")
            fixture.artifact['payload']['quality_report'] = {'episode_audit': {'status':review_status,'coverage_mode':'sampled' if review_status!='complete' else 'full','checked_transitions':2 if review_status=='complete' else 1 if review_status=='partial' else 0,'total_transitions':2,'reviewed_transitions':[]}}
        page.goto(BASE_URL)
        rating = page.locator('.messages details').filter(has_text='质量：待核实')
        rating.locator('summary').focus()
        page.keyboard.press('Enter')
        expect(rating).to_contain_text('自动检查 1 / 2 项')
        if width < 600:
            page.locator('.workspace-tabs button').filter(has_text='Studio').click()
        page.locator('.artifact').filter(has_text='仅脚本示例').click()
        drawer = page.get_by_role('dialog', name='仅脚本示例')
        expect(drawer).to_contain_text('质量：待核实 · 仅脚本')
        expect(drawer.locator('audio')).to_have_count(0)
        expect(drawer.get_by_label('播客连贯性检查')).to_contain_text({'complete':'连贯性检查已完成','partial':'连贯性已部分检查'}.get(review_status,'连贯性未验证'))
        if review_status:
            expect(drawer).to_contain_text('展开版' if review_status=='complete' else '完整短版')
            expect(drawer).to_contain_text({'complete':'收尾检查通过','partial':'完整性不足，仅草稿','unavailable':'收尾待核实'}[review_status])
        expect(drawer).to_contain_text('这是保留的脚本。')
        expect(drawer).to_contain_text('该轮原文支持待核实。')
        assert_layout(page)
        page.screenshot(path=f'/tmp/quick-read-coherence-{review_status or "legacy"}-{width}.png')
        page.keyboard.press('Escape')
        expect(drawer).not_to_be_visible()
        assert not fixture.errors and not fixture.console_errors
        context.close()
