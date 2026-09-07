from __future__ import annotations

import json

import pytest

from sandevistan_read import services, study, retrieval
from sandevistan_read.context_budget import PromptBudget
from sandevistan_read.database import Database, json_dump
from sandevistan_read.providers import BudgetedCompletion
from sandevistan_read.schemas import FlashcardRequest, PodcastRequest, QuizRequest
from sandevistan_read import podcast


@pytest.fixture
def evidence_db(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    db.initialize()
    db.execute("INSERT INTO notebooks(id,title,created_at,updated_at) VALUES('n','Fixture','now','now')")
    db.execute("INSERT INTO sources(id,notebook_id,revision_id,filename,media_type,size_bytes,sha256,blob_path,state,created_at,updated_at) VALUES('s','n','r','fixture.txt','text/plain',100,'hash','unused','ready','now','now')")
    db.execute("INSERT INTO chunks VALUES('c','s','r',0,?,'{}','[]','hash','now')", ("Photosynthesis converts light energy into chemical energy within chloroplasts.",))
    db.execute("INSERT INTO conversations VALUES('conversation','n','Fixture','now','now')")
    for module in (services, study, retrieval):
        monkeypatch.setattr(module, "DB", db)
    return db


def test_history_is_scoped_and_does_not_reuse_citation_labels(evidence_db):
    scope = services.scope_hash(["s"])
    for key, role, content in (("u", "user", "What does photosynthesis do?"), ("a", "assistant", "It converts light energy [S1]."), ("u2", "user", "Why does it matter?")):
        evidence_db.execute("INSERT INTO messages(id,conversation_id,role,content,scope_hash,created_at) VALUES(?, 'conversation',?,?,?,'now')", (key, role, content, scope if role == "assistant" else None))
    history = services.conversation_context("n", "conversation", ["s"])
    assert "photosynthesis" in history and "light energy" in history
    assert "[S1]" not in history and "Why does it matter" not in history
    assert services.conversation_context("other", "conversation", ["s"]) == ""
    evidence_db.execute("UPDATE sources SET revision_id='new-revision' WHERE id='s'")
    assert services.conversation_context("n", "conversation", ["s"]) == ""


def test_grouped_citations_are_expanded_and_unknown_ids_remain_invalid():
    answer = services.normalize_citation_markers("事实依据 [S1, S2]，进一步说明 [S2、S999]。")
    assert answer == "事实依据 [S1] [S2]，进一步说明 [S2] [S999]。"
    assert "未知引用 S999" in services._grounding_issues(answer, {"S1", "S2"})


@pytest.mark.asyncio
async def test_followup_uses_history_for_retrieval_and_generation(evidence_db, monkeypatch):
    scope = services.scope_hash(["s"])
    evidence_db.execute("INSERT INTO messages(id,conversation_id,role,content,created_at) VALUES('u','conversation','user','What is photosynthesis?','1')")
    evidence_db.execute("INSERT INTO messages(id,conversation_id,role,content,scope_hash,created_at) VALUES('a','conversation','assistant','It converts light energy [S99].',?,'2')", (scope,))
    queries, prompts = [], []
    chunk = {"id": "c", "source_id": "s", "content": "Photosynthesis converts light energy into chemical energy.", "locator": {}}

    def retrieve(notebook, query, ids, **kwargs):
        queries.append(query)
        return [chunk]

    async def chat(build, **kwargs):
        budget = PromptBudget(30720, 12000, 2000, 2048, 1)
        built = build(budget)
        prompts.append(json_dump(built.messages))
        return BudgetedCompletion("Photosynthesis stores energy chemically [S1].", built, budget)

    monkeypatch.setattr(services, "retrieve", retrieve)
    monkeypatch.setattr(services, "budgeted_chat", chat)
    result = await services.grounded_generate("n", "Answer", "Why is it useful?", ["s"], "en", conversation_id="conversation")
    assert queries[0] == "Why is it useful?"
    assert any("photosynthesis" in query.lower() for query in queries)
    assert "photosynthesis" in prompts[0].lower() and "S99" not in prompts[0]
    assert result["citations"][0]["id"] == "S1"


@pytest.mark.asyncio
@pytest.mark.parametrize("supported", [True, False])
async def test_cross_language_cards_require_semantic_review_and_keep_partial(evidence_db, monkeypatch, supported):
    provider = {"config": {"study_generation_tier": "lite"}, "capabilities": {"token_limits": {"effective_context_tokens": 30720, "max_output_tokens": 1024}}}
    monkeypatch.setattr(study, "active_provider", lambda role: provider)
    # Keep one unique, meaningful card instead of accepting repeats to reach 20.
    monkeypatch.setattr(study, "_semantic_unique", lambda candidate, accepted, kind: not accepted)
    audits = []

    async def audit(kind, candidates, evidence, trace):
        audits.append(candidates)
        return ([0] if supported else []), []

    async def chat(build, **kwargs):
        budget = PromptBudget(30720, 10000, 1024, 2048, 1)
        built = build(budget)
        candidate = {"front": "光合作用完成怎样的能量转换？", "back": "将光能转化为化学能。", "explanation": "这一过程发生在叶绿体中。", "citations": ["[S1]"], "difficulty": "hard"}
        return BudgetedCompletion(json_dump({"items": [candidate]}), built, budget)

    monkeypatch.setattr(study, "_audit_candidates", audit)
    monkeypatch.setattr(study, "budgeted_chat", chat)
    if not supported:
        with pytest.raises(ValueError, match="0 个"):
            await study.generate_study_artifact("n", "flashcard", 20, ["s"], "zh-CN", "hard")
    else:
        result = await study.generate_study_artifact("n", "flashcard", 20, ["s"], "zh-CN", "hard")
        assert result["status"] == "partial"
        assert len(result["payload"]["items"]) == 1
        assert result["payload"]["items"][0]["difficulty"] == "hard"
        assert result["payload"]["warnings"][0]["code"] == "count_shortfall"
    assert audits


@pytest.mark.asyncio
async def test_malformed_audit_is_not_a_valid_rejection(monkeypatch):
    async def chat(build, **kwargs):
        return type("Completion", (), {"content": '{"explanation":"incomplete"}'})()

    monkeypatch.setattr(study, "budgeted_chat", chat)
    with pytest.raises(ValueError, match="accepted_indexes"):
        await study._audit_candidates("flashcard", [{"citations": []}], {}, None)


@pytest.mark.parametrize("schema,maximum", [(QuizRequest, 30), (FlashcardRequest, 50)])
def test_all_legal_study_parameters_validate(schema, maximum):
    for count in range(1, maximum + 1):
        for difficulty in ("easy", "medium", "hard", "mixed"):
            for language in ("auto", "zh-CN", "en"):
                assert schema(count=count, difficulty=difficulty, language=language).count == count
    for count in (0, maximum + 1):
        with pytest.raises(ValueError):
            schema(count=count)


def test_podcast_duration_contract():
    for minutes in (5, 10, 20, 30):
        for language in ("auto", "zh-CN", "en"):
            assert PodcastRequest(minutes=minutes, language=language).duration_mode == "fixed"
    assert PodcastRequest().duration_mode == "auto"
    with pytest.raises(ValueError):
        PodcastRequest(duration_mode="fixed")
    with pytest.raises(ValueError):
        PodcastRequest(minutes=7)


def test_podcast_language_contract_shares_the_input_budget():
    from sandevistan_read.context_budget import estimate_messages_tokens

    budget = PromptBudget(30720, 600, 1024, 2048, 1)
    built = podcast._segment_prompt_build(budget, prefix="Chinese planning instructions: " * 5, items=[{"text": "Evidence text. " * 300}], renderer=lambda item: item["text"], language="en")
    assert built.messages[0]["role"] == "system"
    assert "English" in built.messages[0]["content"]
    assert estimate_messages_tokens(built.messages) <= budget.input_tokens


@pytest.mark.asyncio
async def test_blueprint_accepts_bracketed_citations(evidence_db, monkeypatch):
    async def chat(build, **kwargs):
        budget = PromptBudget(30720, 10000, 2000, 2048, 1)
        built = build(budget)
        return BudgetedCompletion(json_dump({"concepts": [{"title": "Energy", "objective": "Understand conversion", "difficulty": "easy", "citations": ["[S1]"]}]}), built, budget)

    monkeypatch.setattr(study, "budgeted_chat", chat)
    chunks = [{"id": "c", "source_id": "s", "content": "Photosynthesis converts energy", "locator": {}}]
    concepts, fallback = await study._build_blueprint(chunks, ["S1"], 1, "easy", "en", "", "full", services.ContextUsage())
    assert not fallback and concepts[0]["citations"] == ["S1"]


@pytest.mark.asyncio
async def test_partial_act_survives_failed_continuation(monkeypatch):
    async def draft(**kwargs):
        return podcast.SceneDraftResult([{"speaker": "HOST_A", "text": "已有的有效对话内容。", "citation_ids": ["E1"], "claim_ids": ["C1"], "dialogue_act": "explain"}], ["有效轮次不足"], "length")

    async def continuation(**kwargs):
        raise RuntimeError("MAIN token 达到任务上限")

    monkeypatch.setattr(podcast, "_draft_scene", draft)
    monkeypatch.setattr(podcast, "_continue_scene", continuation)
    turns, audit = await podcast.create_linked_scene(scene_kind="act", chapter={"title": "Fixture"}, claims=[], cards_by_id={}, memory=podcast.EpisodeMemory("Topic"), existing_turns=[], target=4, language="zh-CN", profile={"recent_turns": 4}, trace=services.ContextUsage(), generation_state=podcast.EpisodeGenerationState(allow_partial=True))
    assert len(turns) == 1 and audit["partial"] and not audit["passed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", [[], [0], ["review"]])
async def test_partial_podcast_preserves_metrics_and_remaps_removed_turns(evidence_db, monkeypatch, unsupported):
    cards = [{"id": "E1", "source_id": "s", "filename": "fixture.txt", "locator": {}, "content": "光合作用将光能转为化学能。"}, {"id": "E2", "source_id": "s", "filename": "fixture.txt", "locator": {}, "content": "叶绿体参与这一转换。"}]
    citations = [{**card, "quote": card["content"], "chunk_id": "c"} for card in cards]
    claims = [{"id": f"C{i}", "text": card["content"], "source_id": "s", "evidence_ids": [card["id"]]} for i, card in enumerate(cards, 1)]
    chapters = [{"id": f"chapter_{i}", "title": "Energy", "purpose": "Explain", "claim_ids": ["C1", "C2"]} for i in (1, 2)]
    monkeypatch.setattr(podcast, "select_podcast_evidence", lambda *args: [])
    monkeypatch.setattr(podcast, "build_evidence_cards", lambda *args: (cards, citations))
    monkeypatch.setattr(podcast, "build_claim_ledger", lambda *args: claims)
    monkeypatch.setattr(podcast, "podcast_generation_profile", lambda: {"scene_turns": 8, "recent_turns": 4})

    async def plan(*args):
        return {"episode_thesis": "Energy", "chapters": chapters}, False

    async def scene(**kwargs):
        index = len(kwargs["existing_turns"])
        return [{"speaker": f"HOST_{speaker}", "text": f"资料解释光合作用中的能量转换机制，阶段 {index + i}。", "claim_ids": ["C1"], "citation_ids": ["E1"], "dialogue_act": "explain"} for i, speaker in enumerate(("A", "B"))], {"passed": True, "partial": True}

    async def expansion(*args):
        raise RuntimeError("MAIN token 达到任务上限")

    async def audit(*args):
        return {"passed": False, "scores": {"grounding": 3 if unsupported else 5}, "unsupported_turns": [] if unsupported == ["review"] else unsupported, "issues": ["Fixture review"]}

    async def recover(*args):
        return {"accepted_indexes": [0, 1], "issues": []}

    monkeypatch.setattr(podcast, "_audit_grounded_subset", recover)
    monkeypatch.setattr(podcast, "create_episode_plan", plan)
    monkeypatch.setattr(podcast, "create_linked_scene", scene)
    monkeypatch.setattr(podcast, "_expand_episode_duration", expansion)
    monkeypatch.setattr(podcast, "_audit_episode", audit)
    result = await podcast.build_podcast_script("n", {"source_ids": ["s"], "minutes": 5, "language": "zh-CN"}, allow_partial=True)
    assert result["degraded"] and not result["quality"]["passed"]
    assert result["duration"]["target_minutes"] == 5
    assert len(result["turns"]) == (2 if unsupported == ["review"] else 4 - len(unsupported))
    assert result["chapters"][-1]["turn_end"] == len(result["turns"]) - 1
    assert all(turn["citation_ids"] == ["S1"] for turn in result["turns"])


@pytest.mark.asyncio
@pytest.mark.parametrize('status,expected_calls', [(500, 2), (503, 2), (400, 1), (401, 1)])
async def test_ollama_transient_retry_is_bounded_and_pinned(monkeypatch, status, expected_calls):
    from sandevistan_read import providers
    provider = {'kind': 'ollama', 'model': 'gemma4:e4b', 'config': {'context_window_tokens': 30720}}
    attempts = []

    async def once(selected, messages, **kwargs):
        attempts.append((selected, kwargs['max_tokens']))
        if len(attempts) == 1:
            raise providers.ProviderError('runner stopped', status=status)
        return providers.ChatCompletion('recovered')

    monkeypatch.setattr(providers, '_chat_once', once)
    trace = services.ContextUsage()
    async def run():
        return await providers.budgeted_chat(lambda budget: providers.PromptBuild([{'role': 'user', 'content': 'Hello'}]), provider_override=provider, trace=trace)
    if expected_calls == 2:
        assert (await run()).content == 'recovered'
        assert trace.requests == 2
        assert attempts[0] == attempts[1]
    else:
        with pytest.raises(providers.ProviderError):
            await run()
    assert len(attempts) == expected_calls


@pytest.mark.asyncio
async def test_small_summary_budget_retains_complete_translated_points(evidence_db, monkeypatch):
    monkeypatch.setattr(services, 'active_provider', lambda role: {'config': {'max_output_tokens': 1024}})
    monkeypatch.setattr(services, 'select_quality_evidence', lambda *args, **kwargs: [{'id': 'c', 'source_id': 's', 'content': 'Photosynthesis converts light energy into chemical energy within chloroplasts.', 'locator': {}}])
    calls = []

    async def chat(build, **kwargs):
        budget = PromptBudget(30720, 10000, 1024, 2048, 1)
        built = build(budget)
        calls.append(built)
        if len(calls) > 1:
            raise RuntimeError('temporary outage')
        point = {'claim': '光合作用利用光能完成能量转换，将其转化为储存的化学能。', 'why_it_matters': '这一过程发生在叶绿体中，使光能转化为可储存的能量。', 'citations': ['S1']}
        return BudgetedCompletion('{"points":[' + json_dump(point) + ',{"claim":"unfinished', built, budget, 'length')

    monkeypatch.setattr(services, 'budgeted_chat', chat)
    result = await services._hierarchical_summary('n', ['s'], 'zh-CN')
    assert result['degraded'] and len(result['points']) == 1
    assert result['points'][0]['claim'].startswith('光合作用')
    assert result['citations'][0]['id'] == 'S1'


def test_history_budget_preserves_recent_question_with_long_answer():
    history = '用户：Old topic\n助手：' + 'Old response. ' * 100 + '\n\n用户：Compare first: work; second: a third party.\n助手：' + 'A long response. ' * 200
    clipped, _ = services._bounded_dialogue(history, 500)
    assert 'Compare first: work; second: a third party.' in clipped


def test_partial_answer_drops_unknown_references_and_uncited_lines():
    answer = 'This valid claim has an existing reference [S1].\nThis unsupported claim cites an unknown source [S1] [S99].\nThis other unsupported claim has no reference.'
    assert services._cited_answer_lines(answer, {'S1'}, 'en') == answer.splitlines()[0]


@pytest.mark.parametrize('explanation', ['选项A和C分别描述了支持与反对。', 'Option A is supported, while C is contradicted.', '第一项描述了正确机制。'])
def test_quiz_balancing_keeps_positional_feedback_aligned(explanation):
    item = {'options': ['Supported fact', 'Distractor two', 'Distractor three', 'Distractor four'], 'answer_index': 0, 'explanation': explanation}
    assert study._balance_answer(item, 3) == item


def test_quiz_balancing_can_move_options_without_positional_references():
    item = {'options': ['Supported fact', 'Distractor two', 'Distractor three', 'Distractor four'], 'answer_index': 0, 'explanation': 'The supported fact matches the source evidence.'}
    balanced = study._balance_answer(item, 3)
    assert balanced['options'][balanced['answer_index']] == 'Supported fact'
    assert balanced['answer_index'] == 3


@pytest.mark.asyncio
async def test_grounding_recovery_only_accepts_explicit_visible_indexes(monkeypatch):
    turns = [{'speaker': 'HOST_A', 'text': 'Light energy is converted.', 'citation_ids': ['E1']}, {'speaker': 'HOST_B', 'text': 'Chemical energy is stored.', 'citation_ids': ['E1']}]
    async def chat(build, **kwargs):
        budget = PromptBudget(30720, 10000, 1200, 2048, 1)
        built = build(budget)
        assert 'Photosynthesis converts light energy' in str(built.messages)
        return BudgetedCompletion('{"accepted_indexes":[0,99,true],"issues":[]}', built, budget)
    monkeypatch.setattr(podcast, 'budgeted_chat', chat)
    result = await podcast._audit_grounded_subset(turns, {'E1': {'content': 'Photosynthesis converts light energy to chemical energy.'}}, services.ContextUsage())
    assert result['accepted_indexes'] == [0]


def test_unreadable_formula_is_not_retained_as_a_grounded_answer():
    answer = 'A slower attacker has a decreasing chance of catching up [S1].\nIf $p \\ne q$, then $q_z=1$ means catching up is impossible [S1].'
    assert services._cited_answer_lines(answer, {'S1'}, 'en', omit_formulas=True) == answer.splitlines()[0]
    assert services._is_refusal('The provided sources do not contain current market prices [No citation].')
    assert not services._has_formula('The gap $z$ increases [S1].')


def test_job_detail_and_list_do_not_expose_quiz_answers(evidence_db, monkeypatch):
    from sandevistan_read import app, observability
    monkeypatch.setattr(app, 'DB', evidence_db)
    monkeypatch.setattr(observability, 'DB', evidence_db)
    result = {'id': 'artifact', 'type': 'quiz', 'status': 'ready', 'payload': {'items': [{'id': 'q1', 'question': 'Energy?', 'options': ['Light', 'Water', 'Air', 'Soil'], 'answer_index': 0, 'explanation': 'Source evidence', 'citations': ['S1']}]}, 'citations': [{'id': 'S1'}]}
    evidence_db.execute("INSERT INTO jobs(id,kind,state,stage,progress,payload_json,notebook_id,result_json,created_at,updated_at) VALUES('j','quiz','complete','完成',1,'{}','n',?,'now','now')", (json_dump(result),))
    for response in (app.job('j'), app.jobs(notebook_id='n', page=1, page_size=20)['items'][0]):
        public = response['result']
        assert public['id'] == 'artifact' and public['status'] == 'ready'
        assert public['payload']['items'][0]['question'] == 'Energy?'
        assert not {'answer_index', 'explanation', 'citations'} & public['payload']['items'][0].keys()
        assert public['citations'] == []
    stored = json.loads(evidence_db.fetchone("SELECT result_json FROM jobs WHERE id='j'")['result_json'])
    assert stored['payload']['items'][0]['answer_index'] == 0
