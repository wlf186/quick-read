"""Regression contracts for complete exchanges and bounded source synthesis."""
import json
from sandevistan_read import podcast


def test_truncated_exchange_never_leaks_a_question_into_audio():
    raw = '{"exchanges":[{"turns":[["A","Q","Why does the record persist?",["C1"]],["B","X","Proof of work protects the record.",["C1"]]]},{"turns":[["A","Q","What comes next?",[]]'
    turns = podcast._extract_turns(raw)
    assert len(turns) == 2
    assert turns[0]['exchange_start'] and not turns[1]['exchange_start']
    assert turns[-1]['text'] == 'Proof of work protects the record.'


def test_broken_exchange_keeps_independent_later_chapter_and_remaps_review():
    turns = [{'speaker': 'HOST_A' if i % 2 == 0 else 'HOST_B', 'text': text,
              'exchange_id': f'unit{i // 2}', 'exchange_start': i % 2 == 0}
             for i, text in enumerate(['Why does work resist changes?', 'Payments are broadcast.',
                                      'Privacy depends on unlinking identities.', 'Public keys need not identify their owners.',
                                      'Therefore this follows.', 'That conclusion repeats the point.'])]
    audit = {'breaks': [1], 'broken_at': 1, 'reviewed_indexes': [0, 1, 2, 3],
             'reviewed_transitions': [0, 2], 'transition_checks': [{'index': 0}, {'index': 2}], 'requested_transitions': [0, 2]}
    result, chapters, status = podcast.retain_product_exchanges(turns, [{'id': 'one', 'turn_start': 0, 'turn_end': 1}, {'id': 'two', 'turn_start': 2, 'turn_end': 5}], audit, [])
    assert status == 'partial' and result[0]['text'].startswith('Privacy')
    assert chapters[0]['turn_start'] == 0 and chapters[0]['turn_end'] == 3
    assert audit['reviewed_transitions'] == [0]
    assert audit['transition_checks'] == [{'index': 0}]
    assert len(audit['discarded_exchanges']) == 1


def test_dependent_exchange_after_removed_unit_is_not_spliced_in():
    turns = [{'speaker': 'HOST_A' if i % 2 == 0 else 'HOST_B', 'text': text,
              'exchange_id': str(i // 2), 'exchange_start': i % 2 == 0}
             for i, text in enumerate(['Why is the record secure?', 'Transactions are public.',
                                      'This explains the earlier point.', 'That point is central.',
                                      'Privacy uses anonymous public keys.', 'Users may keep identities separate.'])]
    result, _, status = podcast.retain_product_exchanges(turns, [{'turn_start': 0, 'turn_end': 5}], {'broken_at': 1}, [])
    assert status == 'partial' and len(result) == 2
    assert result[0]['text'].startswith('Privacy')


def test_scene_validation_keeps_exchange_metadata():
    turns = podcast._extract_turns(json.dumps({'exchanges': [{'turns': [['A', 'X', 'The record is public.', ['C1']], ['B', 'X', 'Public announcements establish the record.', ['C1']]]}]}))
    accepted, _ = podcast.validate_scene_turns(turns, {'C1': {'id': 'C1', 'text': 'The record is public.', 'evidence_ids': ['E1']}}, {'E1': {'id': 'E1', 'text': 'The record is public.'}}, last_speaker=None, existing_turns=[], language='en', expected_count=2, scene_kind='act', allow_style_degradation=True)
    assert len(accepted) == 2
    assert accepted[0]['exchange_id'] == accepted[1]['exchange_id']
    assert accepted[0]['exchange_start'] is True and accepted[1]['exchange_start'] is False


def test_summary_capacity_allocates_every_source_before_more_detail():
    from sandevistan_read.context_budget import TokenLimits, plan_context
    from sandevistan_read.services import summary_point_quotas
    ids = [str(i) for i in range(5)]
    plans = [plan_context(TokenLimits.from_provider({'config': {'context_window_tokens': 204800, 'max_output_tokens': output}}), 'summary', material_tokens=180000, source_count=5) for output in (4096, 16384)]
    for plan in plans:
        quota = summary_point_quotas(ids, plan.output_items, plan.overview_items)
        assert all(1 <= quota[source] <= 4 for source in ids)
        assert sum(quota.values()) == plan.output_items
        assert 1 <= plan.preparation_batches <= 4
    assert plans[1].output_items > plans[0].output_items
    assert plans[1].note_capacity > plans[0].note_capacity


def test_synthesis_keeps_a_source_without_accepted_notes():
    from sandevistan_read.context_budget import TokenLimits, plan_context, ContextUsage
    from sandevistan_read.generation_context import CURRENT, GenerationContext, synthesis_evidence
    provider = {'config': {'context_window_tokens': 30720, 'max_output_tokens': 4096}}
    rows = [{'id': f'c{i}', 'source_id': str(i), 'content': 'A source argument with its necessary limiting condition.'} for i in range(5)]
    state = GenerationContext(provider, plan_context(TokenLimits.from_provider(provider), 'summary', source_count=5), rows, [], ContextUsage())
    state.preparation_attempted = True
    state.notes = [{'chunk_id': 'c0'}]
    token = CURRENT.set(state)
    try:
        assert {r['source_id'] for r in synthesis_evidence(rows)} == {str(i) for i in range(5)}
    finally:
        CURRENT.reset(token)


import pytest


@pytest.mark.asyncio
async def test_preparation_reserves_synthesis_audit_and_recovery(monkeypatch):
    from sandevistan_read import providers
    from sandevistan_read.context_budget import TokenLimits, plan_context, ContextUsage, prompt_budget
    from sandevistan_read.generation_context import CURRENT, GenerationContext, prepare_evidence
    provider = {'config': {'context_window_tokens': 30720, 'max_output_tokens': 4096}}
    limits = TokenLimits.from_provider(provider)
    row = {'id': 'c', 'source_id': 's', 'ordinal': 0, 'content': 'A precise source claim with its necessary limiting condition.'}
    plan = plan_context(limits, 'summary', material_tokens=90000, source_count=3)
    state = GenerationContext(provider, plan, [row], [], ContextUsage(request_limit=plan.preparation_batches + 3))
    calls = []
    async def chat(builder, **kwargs):
        calls.append(kwargs['stage'])
        budget = prompt_budget(limits, kwargs['max_tokens'], 128, 1.0)
        return providers.BudgetedCompletion(json.dumps({'notes': [{'chunk_id': 'c', 'claim': 'A qualified claim.', 'quote': row['content']}]}), builder(budget), budget)
    monkeypatch.setattr(providers, 'budgeted_chat', chat)
    token = CURRENT.set(state)
    try:
        notes = await prepare_evidence([row], 'en')
        assert len(notes) == 1 and calls == ['context_prepare']
        assert state.preparation['attempted_batches'] == 1
        assert await prepare_evidence([row], 'en') == notes
        assert len(calls) == 1
    finally:
        CURRENT.reset(token)


def test_ollama_model_override_reads_show_and_respects_explicit_windows(monkeypatch):
    import importlib.util
    from pathlib import Path
    import httpx
    spec = importlib.util.spec_from_file_location('gemma_evaluator', Path(__file__).resolve().parents[1] / 'scripts/evaluate_context_strategy.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    requests = []
    def handler(request):
        requests.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={'model_info': {'gemma4.context_length': 262144}})
    client = httpx.Client
    monkeypatch.setattr(evaluator.httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    original = {'kind': 'ollama', 'base_url': 'http://localhost:11434', 'model': 'gemma4:e4b', 'config': {'context_window_tokens': 30720, 'max_output_tokens': 4096}, 'capabilities': {}}
    result = evaluator.override_provider(original, model='gemma4:12b', context_window=204800, max_output=16384)
    assert requests == [('POST', '/api/show', {'model': 'gemma4:12b'})]
    assert result['config']['context_window_tokens'] == 204800
    assert result['config']['max_output_tokens'] == 16384
    assert original['model'] == 'gemma4:e4b'


def test_hints_never_precede_a_truncated_original(monkeypatch):
    from sandevistan_read import services
    from sandevistan_read.context_budget import TokenLimits, plan_context, ContextUsage, PromptBudget, estimate_messages_tokens
    from sandevistan_read.generation_context import CURRENT, GenerationContext
    provider = {'config': {'context_window_tokens': 30720, 'max_output_tokens': 4096}}
    row = {'id': 'c', 'source_id': 's', 'filename': 'book', 'locator': {}, 'content': 'Original source material. ' * 1000}
    state = GenerationContext(provider, plan_context(TokenLimits.from_provider(provider), 'summary'), [row], [], ContextUsage())
    state.notes = [{'chunk_id': 'c', 'claim': 'A classified source hint.', 'qualification': 'Only under a condition.'}]
    token = CURRENT.set(state)
    try:
        budget = PromptBudget(30720, 2000, 4096, 2048, 1)
        built = services._evidence_prompt_build(budget, chunks=[row], labels=['S1'], prefix='Answer from source. ', include_notes=True)
        assert built.truncated_segments == 1
        assert 'A classified source hint.' not in str(built.messages)
        assert estimate_messages_tokens(built.messages) <= budget.input_tokens
    finally:
        CURRENT.reset(token)


def test_conservative_six_point_summary_still_allocates_source_sections():
    from sandevistan_read.services import summary_point_quotas
    ids = [str(i) for i in range(5)]
    quota = summary_point_quotas(ids, 6, 6)
    assert quota[None] == 1
    assert all(quota[source] == 1 for source in ids)


def test_numbered_dialogue_objects_preserve_numeric_order_and_citations():
    values = {str(i): ['A' if i % 2 else 'B', 'X', f'The source states condition {i}.', ['C1_E1']] for i in range(10, 0, -1)}
    turns = podcast._extract_turns(json.dumps(values))
    assert len(turns) == 10
    assert turns[0]['text'].endswith('1.') and turns[-1]['text'].endswith('10.')
    assert turns[0]['claim_ids'] == ['C1_E1']
    assert podcast._ordered_records({'1': 'one', '3': 'three'}) is None
    assert podcast._ordered_records({'1': 'one', 'metadata': 'other'}) is None


def test_numbered_exchanges_and_turns_are_unambiguous():
    raw = {'exchanges': {'1': {'turns': {'1': ['A', 'Q', 'Why does work matter?', ['C1']], '2': ['B', 'X', 'Work makes changing the record costly.', ['C1']]}}}}
    turns = podcast._extract_turns(json.dumps(raw))
    assert len(turns) == 2 and turns[0]['exchange_start']
    assert turns[0]['exchange_id'] == turns[1]['exchange_id']


def test_single_turn_exchange_wrappers_recover_adjacent_complete_pair():
    raw = {'exchanges': [
        {'turns': [['HOST_A', 'X', 'An isolated introduction.', ['C0']]]},
        {'turns': [['HOST_A', 'Q', 'Why does work matter?', ['C1']]]},
        {'turns': [['HOST_B', 'X', 'Work makes changing the record costly.', ['C1']]]},
        {'turns': [['HOST_A', 'Q', 'What happens next?', []]]},
    ]}
    turns = podcast._extract_turns(json.dumps(raw))
    assert len(turns) == 2
    assert turns[0]['text'] == 'Why does work matter?'
    assert turns[0]['exchange_id'] == turns[1]['exchange_id']
    assert turns[-1]['text'] == 'Work makes changing the record costly.'


def test_single_turn_wrappers_do_not_join_across_invalid_or_unrelated_units():
    raw = {'exchanges': [
        {'turns': [['A', 'Q', 'Why does work matter?', ['C1']]]},
        {'turns': 'invalid'},
        {'turns': [['B', 'X', 'Public keys support privacy.', ['C2']]]},
        {'turns': [['A', 'X', 'Blocks contain transaction hashes.', ['C3']]]},
    ]}
    assert podcast._extract_turns(json.dumps(raw)) is None


def test_compact_turn_with_multiple_trailing_claim_ids_keeps_complete_exchange():
    raw = {'exchanges': [{'turns': [
        ['HOST_A', 'X', 'The first source states the condition.', 'C1'],
        ['HOST_B', 'X', 'The second source qualifies that condition.', 'C1', 'C3'],
    ]}]}
    turns = podcast._extract_turns(json.dumps(raw))
    assert len(turns) == 2 and turns[1]['claim_ids'] == ['C1', 'C3']
    raw['exchanges'][0]['turns'][1][-1] = 'unrecognized extra prose'
    assert podcast._extract_turns(json.dumps(raw)) is None


def test_missing_summary_source_ids_preserve_all_five_cited_sources():
    from sandevistan_read.services import allocate_summary_points, summary_point_quotas
    source_order = ['bitcoin'] * 5 + ['learning'] * 2 + ['strange-loop', 'geb', 'zen']
    points = [{'claim': f'Original claim {i}', 'citations': [f'S{i}']} for i in range(10)]
    citation_sources = {f'S{i}': source for i, source in enumerate(source_order)}
    quotas = summary_point_quotas(['bitcoin', 'geb', 'strange-loop', 'learning', 'zen'], 10, 5)
    retained = allocate_summary_points(points, quotas, citation_sources)
    assert len(retained) == 10
    assert {p['source_id'] for p in retained if p['source_id']} == set(source_order)
    assert [p['claim'] for p in retained] == [p['claim'] for p in points]
    assert sum(p['source_id'] is None for p in retained) == 5


def test_cross_source_and_unresolved_summary_citations_do_not_invent_attribution():
    from sandevistan_read.services import allocate_summary_points
    points = [{'claim': 'Cross-source comparison.', 'source_id': None, 'citations': ['S1', 'S2']},
              {'claim': 'Unresolved claim.', 'citations': ['S99']}]
    retained = allocate_summary_points(points, {None: 2, 'a': 1, 'b': 1}, {'S1': 'a', 'S2': 'b'})
    assert len(retained) == 2 and all(p['source_id'] is None for p in retained)
