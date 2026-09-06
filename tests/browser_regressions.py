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
        self.job_state = "queued"
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
        artifact = {"id": "cards", "type": "flashcard", "title": "闪卡组", "status": "ready", "payload": {}, "citations": []}
        job = {"id": "ingest", "kind": "ingest", "notebook_id": "a", "notebook_title": "Audit A", "display_name": "文档解析",
               "state": self.job_state, "stage": "已取消" if self.job_state == "cancelled" else "等待执行", "stage_code": self.job_state,
               "progress": 1 if self.job_state == "cancelled" else 0, "created_at": "2026-09-01T00:00:00Z", "updated_at": "2026-09-01T00:00:00Z",
               "eta": {"status": "learning", "sample_count": 0, "queue_position": 1}}
        if path.endswith("/chat"):
            self.pending.append(route)
            return
        if path == "/notebooks":
            result = notebooks
        elif path in ("/providers", "/provider-roles"):
            result = []
        elif path == "/settings/image-processing":
            result = {"mode": "process", "processors": ["ocr"]}
        elif path == "/status":
            result = {"providers": {}}
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
                 "selected": 1 if self.source_state == "ready" else 0, "page_count": 1, "error": "解析已取消" if self.source_state == "failed" else None}]}
        elif path.endswith("/conversations"):
            if self.hold_history:
                self.pending_history.append(route)
                return
            result = [{"id": "history"}] if self.hold_messages else []
        elif path == "/conversations/history/messages":
            if self.hold_messages:
                self.pending_messages.append(route)
                return
            result = []
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


if __name__ == "__main__":
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
        run_core_regressions(browser)
        browser.close()
    print("Core UI regressions passed")
