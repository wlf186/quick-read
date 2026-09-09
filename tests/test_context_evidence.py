import json
from types import SimpleNamespace

import pytest

from sandevistan_read import providers, retrieval, services, context_qualification
from sandevistan_read.context_budget import ContextUsage, TokenLimits, context_strategy, plan_context
from sandevistan_read.generation_context import CURRENT, GenerationContext, evidence_cost, mark_sent, report


def model(window=256000):
    return {'id': 'provider', 'kind': 'openai', 'model': 'fixture', 'base_url': 'http://localhost:9',
            'config': {'context_window_tokens': window, 'max_output_tokens': 100000 if window > 100000 else 4096}}


def test_direct_capacity_exceeds_old_stage_ceiling_without_exceeding_task_budget():
    p = plan_context(TokenLimits.from_provider(model()), 'summary', material_tokens=150000)
    assert p.path == 'direct' and p.evidence_tokens == 150000 and p.preparation_batches == 0
    assert p.evidence_tokens + p.output_tokens + p.final_reserve_tokens <= p.total_token_limit <= 300000
    large = plan_context(TokenLimits.from_provider(model(1000000)), 'summary', material_tokens=900000)
    assert 64000 < large.evidence_tokens < 300000 and large.path == 'structured'
    assert large.output_tokens == p.output_tokens


@pytest.mark.parametrize('sources', [5, 10])
def test_growing_budget_keeps_existing_sources_regions_and_short_definitions(sources):
    rows = [{'id': f'{source}-{spine}', 'source_id': str(source), 'ordinal': spine,
             'filename': 'book.epub', 'locator_json': json.dumps({'spine': spine, 'section': 'Same title'}),
             'content': 'Only if the system is consistent.', 'embedding_json': '[1,0]'}
            for source in range(sources) for spine in [1, 2, 30]]
    assert all(retrieval.is_context_chunk(row) for row in rows)
    cost = evidence_cost(rows[0])
    small = retrieval.select_context_evidence(rows, cost * sources)
    big = retrieval.select_context_evidence(rows, cost * sources * 3 + 100)
    assert {row['source_id'] for row in small} == {str(source) for source in range(sources)}
    assert {row['id'] for row in small} <= {row['id'] for row in big}
    assert len(big) == len(rows)


def test_coverage_distinguishes_preparation_from_synthesis():
    row = {'id': 'c', 'source_id': 's', 'content': 'A precise source claim with its necessary limiting condition.', 'ordinal': 0}
    state = GenerationContext(model(), plan_context(TokenLimits.from_provider(model()), 'summary'), [row], [], ContextUsage())
    token = CURRENT.set(state)
    try:
        build = providers.PromptBuild([{'role': 'user', 'content': row['content']}], metadata={'chunks': [row]})
        mark_sent(build, 'context_prepare')
        assert report()['stages']['context_prepare']['sent_segments'] == 1
        assert 'summary' not in report()['stages']
        mark_sent(build, 'summary')
        assert report()['stages']['summary']['sent_segments'] == 1
    finally:
        CURRENT.reset(token)


@pytest.mark.asyncio
async def test_missing_audit_retains_unreviewed_without_retry(monkeypatch):
    state = GenerationContext(model(), plan_context(TokenLimits.from_provider(model()), 'summary'), [], [], ContextUsage())
    points = [{'claim': str(index)} for index in range(3)]
    calls = []
    async def batch(values, *args):
        calls.append(values)
        return {0: 'supported', 1: 'unsupported'} if len(calls) == 1 else {}
    monkeypatch.setattr(services, '_audit_summary_batch', batch)
    token = CURRENT.set(state)
    try:
        result = await services._audit_summary_points(points, [], [], state.trace)
        assert result == points and len(calls) == 1
        assert [p["review_status"] for p in result] == ["supported", "unsupported", "unreviewed"]
        assert state.audit == {'total': 3, 'supported': 1, 'unsupported': 1, 'unreviewed': 1, 'supplement_attempted': False}
    finally:
        CURRENT.reset(token)


def test_configuration_bound_qualification_and_explicit_conservative(monkeypatch):
    from sandevistan_read import database
    p = model()
    records = [{'fingerprint': context_qualification.fingerprint(p), 'kind': 'summary', 'passed': True, 'report_sha256': 'a' * 64}]
    monkeypatch.setattr(database, 'DB', SimpleNamespace(fetchone=lambda *args: {'value_json': json.dumps(records)}))
    assert context_strategy(p, 'summary') == 'balanced'
    assert context_strategy(p, 'chat') == 'conservative'
    assert context_strategy(model(1000000), 'summary') == 'conservative'
    assert context_strategy({**p, 'model': 'new-model'}, 'summary') == 'conservative'
    assert context_strategy({**p, 'config': {**p['config'], 'context_strategy': 'conservative'}}, 'summary') == 'conservative'


def test_dialogue_keeps_complete_pairs_when_window_grows():
    state = GenerationContext(model(), plan_context(TokenLimits.from_provider(model()), 'chat'), [], [], ContextUsage())
    history = '用户：first\n助手：' + 'Long earlier answer. ' * 100 + '\n\n用户：second\n助手：Short recent answer.'
    token = CURRENT.set(state)
    try:
        short, truncated = services._bounded_dialogue(history, 100)
        assert short == '用户：second\n助手：Short recent answer.' and truncated
        full, truncated = services._bounded_dialogue(history, 10000)
        assert full == history and not truncated
    finally:
        CURRENT.reset(token)


def review_report(p):
    from itertools import product
    from sandevistan_read.context_budget import CONTEXT_STRATEGY_VERSION
    pairs = []
    for corpus, language, repeat in product(('bitcoin', 'geb', 'multi'), ('zh-CN', 'en'), (1, 2)):
        metrics = dict(critical=10, missing=5, failed=0, degraded=0, fact_errors=0, citation_errors=0)
        pairs.append(dict(corpus=corpus, language=language, repeat=repeat,
                          baseline=metrics, candidate={**metrics, 'missing': 4}))
    return dict(provider_fingerprint=context_qualification.fingerprint(p), kind='summary',
                strategy_version=CONTEXT_STRATEGY_VERSION, audit_mode='native', review_method='source_grounded',
                reference_sha256='a'*64, baseline_sha256='b'*64, candidate_sha256='c'*64,
                fixture_sha256='d'*64, pairs=pairs)


def test_qualification_requires_complete_nonregressing_native_pairs():
    import copy
    p = model()
    valid = review_report(p)
    assert context_qualification.assess(valid, p, 'summary')['passed']
    for mutate in (
        lambda r: r['pairs'].pop(),
        lambda r: r.update(audit_mode='external'),
        lambda r: r.update(review_method='keyword'),
        lambda r: r['pairs'][0]['candidate'].update(citation_errors=1),
        lambda r: r['pairs'][0]['candidate'].update(fact_errors=1),
        lambda r: r['pairs'][0]['candidate'].update(degraded=1),
        lambda r: r['pairs'][0]['candidate'].update(missing=6),
        lambda r: r['pairs'][4]['candidate'].update(missing=5),
    ):
        bad = copy.deepcopy(valid)
        mutate(bad)
        assert not context_qualification.assess(bad, p, 'summary')['passed']
    assert not context_qualification.assess(valid, model(1000000), 'summary')['passed']


@pytest.mark.asyncio
async def test_reserved_budget_is_available_to_recovery_but_not_first_generation(monkeypatch):
    from dataclasses import replace
    from sandevistan_read.providers import ChatCompletion, PromptBuild
    p = model()
    plan = replace(plan_context(TokenLimits.from_provider(p), 'summary'), total_token_limit=24000, final_reserve_tokens=6000)
    trace = ContextUsage(total_token_limit=24000, accounted_tokens=17500)
    state = GenerationContext(p, plan, [], [], trace)
    calls = []
    async def answer(*args, **kwargs):
        calls.append(kwargs)
        return ChatCompletion('Recovered answer', prompt_tokens=100, completion_tokens=100)
    monkeypatch.setattr(providers, '_chat_once', answer)
    token = CURRENT.set(state)
    try:
        builder = lambda _: PromptBuild([{'role': 'user', 'content': 'Source passage.'}])
        with pytest.raises((RuntimeError, providers.ContextOverflowError), match='预算'):
            await providers.budgeted_chat(builder, max_tokens=1200, stage='summary')
        assert calls == []
        trace.requests = 1
        result = await providers.budgeted_chat(builder, max_tokens=1200, stage='summary')
        assert result.content == 'Recovered answer' and len(calls) == 1
        assert trace.accounted_tokens < plan.total_token_limit
    finally:
        CURRENT.reset(token)


@pytest.mark.parametrize('sources', [5, 10])
def test_small_budget_spans_beginning_middle_and_end_without_displacement(sources):
    rows = [{'id': f'{source}-{index}', 'source_id': str(source), 'ordinal': index,
             'filename': 'book.pdf', 'locator_json': json.dumps({'page': index * 10 + 1}),
             'content': 'The conclusion holds only under the stated conditions.', 'embedding_json': '[1,0]'}
            for source in range(sources) for index in range(80)]
    cost = evidence_cost(rows[0])
    small = retrieval.select_context_evidence(rows, cost * sources * 3)
    larger = retrieval.select_context_evidence(rows, cost * sources * 5)
    for source in range(sources):
        assert {f'{source}-{index}' for index in [0, 39, 79]} <= {r['id'] for r in small}
    assert {r['id'] for r in small} <= {r['id'] for r in larger}
    assert [r['ordinal'] for r in larger if r['source_id'] == '0'] == sorted(r['ordinal'] for r in larger if r['source_id'] == '0')


def test_index_footer_filters_the_whole_page_but_not_numeric_evidence_or_definitions():
    def row(key, source, page, text):
        return dict(id=key, source_id=source, content=text, locator_json=json.dumps({'page': page}))
    rows = [row('index-start', 'a', 99, 'Many topic names and page references in a split chunk.'),
            row('index-end', 'a', 99, 'Last index entries.\nIndex\n\n99'),
            row('index-odd', 'a', 97, 'More references.\n97\n\nIndex'),
            row('index-spaced', 'a', 96, 'More references.\nIndex\n9 6'),
            row('definition', 'a', 98, 'An index identifies a position.'),
            row('table', 'a', 8, 'q=0.1\nz=5 P=0.0009137\n\n8'),
            row('other', 'b', 99, 'Only if the system is consistent.')]
    assert {r['id'] for r in retrieval.context_candidates(rows)} == {'definition', 'table', 'other'}


@pytest.mark.parametrize('kind', ['summary', 'chat'])
def test_larger_windows_never_reduce_evidence_at_the_task_ceiling(kind):
    windows = [30720, 64000, 128000, 256000, 280000, 300000, 320000, 512000, 1000000, 2000000, 4194304]
    plans = [plan_context(TokenLimits.from_provider(model(window)), kind, material_tokens=2000000) for window in windows]
    assert [p.evidence_tokens for p in plans] == sorted(p.evidence_tokens for p in plans)
    assert all(p.total_token_limit <= 300000 for p in plans)
    assert plans[-1].evidence_tokens == plans[-2].evidence_tokens


def test_source_coverage_can_add_a_source_outside_both_ranked_pools(monkeypatch):
    rows = [{'id': f'a-{i}', 'source_id': 'a', 'ordinal': i, 'content': 'A source passage.',
             'locator_json': '{}', 'embedding_json': '[1,0]'} for i in range(40)]
    rows.append({'id': 'b', 'source_id': 'b', 'ordinal': 0, 'content': 'An unrelated source passage.',
                 'locator_json': '{}', 'embedding_json': '[0,1]'})
    monkeypatch.setattr(retrieval, 'DB', SimpleNamespace(fetchall=lambda *args: rows))
    monkeypatch.setattr(retrieval, 'EMBEDDINGS', SimpleNamespace(encode=lambda *args, **kwargs: [[1,0]]))
    found = retrieval.retrieve('n', '陌生问题', ['a', 'b'], limit=2, ensure_source_coverage=True)
    assert {r['source_id'] for r in found} == {'a', 'b'}
    assert next(r for r in found if r['source_id'] == 'b')['score'] == 0


def test_query_ranking_is_independent_of_the_requested_capacity(monkeypatch):
    rows = [{'id': str(i), 'source_id': 's', 'ordinal': i,
             'content': 'keyword ' * (i % 11 + 1), 'locator_json': '{}',
             'embedding_json': json.dumps([1, i / 100])} for i in range(100)]
    state = GenerationContext(model(), plan_context(TokenLimits.from_provider(model()), 'chat'), rows, [], ContextUsage())
    monkeypatch.setattr(retrieval, 'EMBEDDINGS', SimpleNamespace(encode=lambda *args, **kwargs: [[1,0]]))
    token = CURRENT.set(state)
    try:
        small = retrieval.retrieve('n', 'keyword', ['s'], limit=3)
        large = retrieval.retrieve('n', 'keyword', ['s'], limit=15)
        assert [r['id'] for r in small] == [r['id'] for r in large[:3]]
    finally:
        CURRENT.reset(token)


@pytest.mark.asyncio
async def test_recovery_repacks_evidence_to_the_remaining_task_budget(monkeypatch):
    from sandevistan_read.providers import ChatCompletion
    from sandevistan_read.context_budget import estimate_messages_tokens
    p = model()
    rows = [dict(id=str(i), source_id='s', ordinal=i, content='A' * 40000, locator={}) for i in range(2)]
    plan = plan_context(TokenLimits.from_provider(p), 'chat', material_tokens=1000000)
    trace = ContextUsage(total_token_limit=300000, accounted_tokens=260000, requests=2)
    state = GenerationContext(p, plan, rows, [dict(id='s', filename='book.txt', revision_id='r')], trace)
    calls = []
    async def answer(provider, messages, **kwargs):
        calls.append(messages)
        assert estimate_messages_tokens(messages) + kwargs['max_tokens'] <= 40000
        return ChatCompletion('Recovered bounded answer [S1].', prompt_tokens=18000, completion_tokens=100)
    monkeypatch.setattr(providers, '_chat_once', answer)
    token = CURRENT.set(state)
    try:
        result = await providers.budgeted_chat(lambda budget: services._evidence_prompt_build(
            budget, chunks=rows, labels=['S1', 'S2'], prefix='Repair using this evidence.\n'),
            max_tokens=1800, stage='answer_repair')
        assert result.content and len(calls) == 1
        assert result.budget.input_tokens <= 38200
        assert result.build.truncated_segments or result.build.included_segments < 2
        assert trace.accounted_tokens <= 300000
    finally:
        CURRENT.reset(token)
