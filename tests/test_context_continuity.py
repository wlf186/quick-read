import copy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from sandevistan_read import podcast, providers
from sandevistan_read.context_budget import ContextUsage, PromptBudget, TokenLimits, plan_context


def test_million_window_admits_real_evidence_without_inflating_answers():
    def plan(window, material):
        return plan_context(TokenLimits.from_provider({'config': {'context_window_tokens': window, 'max_output_tokens': 131072}}), 'summary', material_tokens=material)
    small, large = plan(262144, 900000), plan(1048576, 900000)
    assert small.evidence_tokens < 262144 < large.evidence_tokens
    assert large.evidence_tokens > 800000
    assert small.output_tokens == large.output_tokens
    assert plan(1048576, 2000).total_token_limit == plan(262144, 2000).total_token_limit
    assert large.evidence_tokens + large.output_tokens + large.final_reserve_tokens < large.total_token_limit


def test_memory_carries_actual_question_not_planned_bridge():
    memory = podcast.EpisodeMemory('A topic')
    chapter = {'title': 'First act', 'bridge_out': 'A question that was never spoken?'}
    statement = {'speaker': 'HOST_A', 'text': 'The source describes ordered records.', 'claim_ids': ['C1'], 'dialogue_act': 'explain'}
    podcast._update_memory(memory, [statement], chapter, 4)
    assert memory.open_hook == ''
    question = {**statement, 'speaker': 'HOST_B', 'text': 'How are these records ordered?', 'dialogue_act': 'question'}
    podcast._update_memory(memory, [question], chapter, 4)
    assert memory.open_hook == question['text']


def test_three_field_turn_preserves_prose_instead_of_speaking_references():
    raw = json.dumps({'turns': [['HOST_A', 'The original supported explanation.', ['C1|E1']],
                                ['HOST_B', 'What does that explain?', []]]})
    turns = podcast._extract_turns(raw)
    assert turns[0]['text'] == 'The original supported explanation.'
    assert turns[0]['claim_ids'] == ['C1|E1']
    assert turns[1]['dialogue_act'] == 'question'
    accepted, issues = podcast.validate_scene_turns([{'speaker': 'HOST_A', 'text': ['C1|E1'], 'claim_ids': []}],
        {}, {}, last_speaker=None, existing_turns=[], language='en', expected_count=1, allow_style_degradation=True)
    assert not accepted and any('口播不是文本' in issue for issue in issues)


@pytest.mark.asyncio
@pytest.mark.parametrize("supplied,target", [(9, 11), (1, 3)])
async def test_long_outline_keeps_valid_model_plan_beyond_eight_acts(monkeypatch, supplied, target):
    claims = [{'id': f'C{i}', 'text': 'An explicitly qualified source claim.', 'source_id': 's', 'filename': 'book', 'evidence_ids': ['E1']} for i in range(11)]
    chapters = [{'title': f'Topic {i}', 'purpose': 'Explain the source argument.', 'claim_ids': [f'C{i}']} for i in range(supplied)]
    async def chat(builder, **kwargs):
        budget = PromptBudget(30720, 18000, 4096, 2048, 1)
        return providers.BudgetedCompletion(json.dumps({'episode_thesis': 'The actual model thesis.', 'chapters': chapters}), builder(budget), budget)
    monkeypatch.setattr(podcast, 'budgeted_chat', chat)
    result, adjusted = await podcast.create_episode_plan(claims, 'en', '', ContextUsage(), target)
    assert result['episode_thesis'] == 'The actual model thesis.'
    assert len(result['chapters']) == target and adjusted and not result['fallback']
    assert {c['title'] for c in result['chapters']} >= {c['title'] for c in chapters}
    assert {claim_id for c in result['chapters'] for claim_id in c['claim_ids']} == {c['id'] for c in claims}
    assert len({c['id'] for c in result['chapters']}) == target


@pytest.mark.asyncio
async def test_typographic_quotes_count_but_uncertainty_is_not_a_pass(monkeypatch):
    turns = [{'speaker': 'HOST_A', 'text': 'Why is “the public record” useful?', 'claim_ids': []},
             {'speaker': 'HOST_B', 'text': 'It preserves the sequence of events.', 'claim_ids': []}]
    async def chat(builder, **kwargs):
        budget = PromptBudget(30720, 18000, 4096, 2048, 1)
        return providers.BudgetedCompletion(json.dumps({'checks': [{'index': 0, 'question_quote': '"the public record"',
            'answer_quote': 'the sequence of events', 'verdict': 'uncertain', 'reason': 'Needs context'}]}), builder(budget), budget)
    monkeypatch.setattr(podcast, 'active_provider', lambda _: {'config': {'context_window_tokens': 30720}})
    monkeypatch.setattr(podcast, 'budgeted_chat', chat)
    result = await podcast._audit_product_episode(turns, [{'turn_start': 0, 'turn_end': 1}], 'Records', 'en', ContextUsage(), {})
    assert result['checked_transitions'] == 1 and result['status'] == 'complete'
    assert result['requested_transitions'] == [0] and result['passed'] is False
    assert podcast._review_text('It does not follow.') != podcast._review_text('It does follow.')
    assert podcast._review_text('x² is positive') != podcast._review_text('x2 is positive')


def test_isolated_model_override_discards_old_limits(monkeypatch):
    spec = importlib.util.spec_from_file_location('context_evaluator', Path(__file__).resolve().parents[1] / 'scripts/evaluate_context_strategy.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    original = {'kind': 'openai', 'model': 'k3-256k', 'base_url': 'https://provider.example', 'api_key': 'fixture',
        'config': {'context_window_tokens': 256000, 'max_output_tokens': 100000},
        'capabilities': {'token_limits': {'model_context_tokens': 262144}}}
    saved = copy.deepcopy(original)
    real_client = httpx.Client
    monkeypatch.setattr(evaluator.httpx, 'Client', lambda **kwargs: real_client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={'data': [{'id': 'k3', 'context_length': 1048576}]})), **kwargs))
    updated = evaluator.override_provider(original, model='k3', context_window=1048576, max_output=131072, reasoning_effort='high')
    assert TokenLimits.from_provider(updated).effective_context_tokens == 1048576
    assert updated['config']['thinking'] == 'enabled'
    assert original == saved


@pytest.mark.asyncio
async def test_reasoning_effort_is_explicit_and_response_model_is_recorded(monkeypatch):
    real_client = httpx.AsyncClient
    sent = []
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={'model': 'k3', 'choices': [{'message': {'content': '{"ok":true}'}, 'finish_reason': 'stop'}], 'usage': {}})
    monkeypatch.setattr(providers.httpx, 'AsyncClient', lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    provider = {'kind': 'openai', 'model': 'k3', 'base_url': 'https://provider.example', 'api_key': '',
                'config': {'context_window_tokens': 1048576, 'max_output_tokens': 131072, 'reasoning_effort': 'high', 'thinking': 'enabled'}}
    result = await providers._chat_once(provider, [{'role': 'user', 'content': 'JSON please'}], json_mode=True, timeout=5, max_tokens=2048, temperature=1)
    assert sent[0]['reasoning_effort'] == 'high'
    assert sent[0]['thinking'] == {'type': 'enabled'}
    assert result.response_model == 'k3'


@pytest.mark.parametrize("effort", [[], {}, True, "unknown"])
def test_invalid_reasoning_effort_is_a_validation_error(effort):
    from sandevistan_read.context_budget import validate_token_overrides
    with pytest.raises(ValueError, match="reasoning_effort"):
        validate_token_overrides({"reasoning_effort": effort})


@pytest.mark.asyncio
async def test_review_relocates_unique_adjacent_quotes_without_trusting_free_indexes(monkeypatch):
    turns = [{'speaker': 'HOST_A' if i % 2 == 0 else 'HOST_B', 'text': text, 'claim_ids': []}
             for i, text in enumerate(['The record has a definite order.', 'That order is publicly visible.',
                'Why does the chain resist changes?', 'The network broadcasts new transactions.'])]
    async def chat(builder, **kwargs):
        budget = PromptBudget(30720, 18000, 4096, 2048, 1)
        response = {'checks': [{'index': 0, 'question_quote': turns[2]['text'],
                    'answer_quote': turns[3]['text'], 'verdict': 'broken', 'reason': 'Broadcasting does not explain resistance.'}],
                    'breaks': [0], 'broken_at': 0}
        return providers.BudgetedCompletion(json.dumps(response), builder(budget), budget)
    monkeypatch.setattr(podcast, 'active_provider', lambda _: {'config': {'context_window_tokens': 30720}})
    monkeypatch.setattr(podcast, 'budgeted_chat', chat)
    result = await podcast._audit_product_episode(turns, [{'turn_start': 0, 'turn_end': 3}], 'Records', 'en', ContextUsage(), {})
    assert result['reviewed_transitions'] == [2]
    assert result['breaks'] == [3] and result['broken_at'] == 3


def test_spoken_text_omits_claim_and_evidence_labels_without_changing_words():
    assert podcast._normalize_text('A: The record [C12] is public [C1|E2] [S3, S4].') == 'The record is public .'
    assert podcast._normalize_text('Vitamin C12 and [Ca2+] are literal text.') == 'Vitamin C12 and [Ca2+] are literal text.'


@pytest.mark.asyncio
async def test_statement_followed_by_question_is_not_a_hard_coherence_break(monkeypatch):
    turns = [{'speaker': 'HOST_A', 'text': 'The source proposes peer-to-peer payments.', 'claim_ids': []},
             {'speaker': 'HOST_B', 'text': 'How do those peer-to-peer payments work?', 'claim_ids': []}]
    async def chat(builder, **kwargs):
        budget = PromptBudget(30720, 18000, 4096, 2048, 1)
        response = {'checks': [{'index': 0, 'question_quote': turns[0]['text'], 'answer_quote': turns[1]['text'],
                               'verdict': 'broken', 'reason': 'The first turn states a difference, the second asks a question.'}],
                    'breaks': [1], 'broken_at': 1}
        return providers.BudgetedCompletion(json.dumps(response), builder(budget), budget)
    monkeypatch.setattr(podcast, 'active_provider', lambda _: {'config': {'context_window_tokens': 30720}})
    monkeypatch.setattr(podcast, 'budgeted_chat', chat)
    result = await podcast._audit_product_episode(turns, [{'turn_start': 0, 'turn_end': 1}], 'Payments', 'en', ContextUsage(), {})
    assert result['broken_at'] is None and result['breaks'] == []
    assert result['transition_checks'][0]['verdict'] == 'uncertain'
    assert not result['passed'] and result['checked_transitions'] == 1


def test_trimmed_episode_does_not_display_checks_for_removed_turns():
    audit = {'passed': False, 'reviewed_indexes': [0, 1, 3, 4], 'reviewed_transitions': [0, 3],
             'requested_transitions': [0, 3], 'transition_checks': [{'index': 0, 'verdict': 'connected'},
                                                                {'index': 3, 'verdict': 'broken'}]}
    podcast._refresh_episode_review(audit, 3)
    assert audit['transition_checks'] == [{'index': 0, 'verdict': 'connected'}]
    assert audit['requested_transitions'] == [0]
    assert audit['checked_transitions'] == 1 and audit['total_transitions'] == 2
    assert not audit['passed']
