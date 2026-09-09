from __future__ import annotations

import json

import pytest

from sandevistan_read import providers, retrieval
from sandevistan_read.context_budget import ContextUsage, TokenLimits, plan_context
from sandevistan_read.generation_context import CURRENT, GenerationContext, mark_sent, region, report


def provider(context=256000, output=100000):
    return {"kind": "openai", "model": "fixture", "config": {"context_window_tokens": context, "max_output_tokens": output}}


@pytest.mark.parametrize("context,output", [(4096,128),(30720,4096),(131072,8192),(256000,100000),(1000000,384000)])
@pytest.mark.parametrize("kind", ["chat","summary","quiz","flashcard","podcast"])
def test_plan_respects_window_and_total_budget(context, output, kind):
    plan = plan_context(TokenLimits.from_provider(provider(context, output)), kind, material_tokens=1_000_000)
    assert plan.evidence_tokens <= plan.total_token_limit - plan.final_reserve_tokens
    assert plan.total_token_limit <= 300000
    assert plan.final_reserve_tokens >= plan.total_token_limit * .25
    assert plan.output_tokens <= output
    assert plan.preparation_batches <= 4


def test_large_window_changes_capacity_without_filling_small_documents():
    small = plan_context(TokenLimits.from_provider(provider(30720,4096)), "summary", material_tokens=1000000)
    big = plan_context(TokenLimits.from_provider(provider()), "summary", material_tokens=1000000)
    assert big.evidence_tokens > small.evidence_tokens
    tiny = plan_context(TokenLimits.from_provider(provider(1000000,384000)), "summary", material_tokens=2000)
    assert tiny.preparation_batches == 0
    assert tiny.evidence_tokens == 2000
    assert tiny.limiting_factor == "资料已可容纳"


def test_epub_regions_do_not_collapse_repeated_book_titles():
    first = {'locator': {'kind': 'epub', 'section': 'Same book title', 'spine': 2}}
    last = {'locator': {'kind': 'epub', 'section': 'Same book title', 'spine': 39}}
    assert region(first) != region(last)


def test_failed_requests_consume_budget_and_success_settles_reservation():
    trace = ContextUsage(total_token_limit=1000)
    trace.begin_request(estimated_tokens=400)
    trace.record_failure()
    assert trace.accounted_tokens == 400
    trace.begin_request(estimated_tokens=500)
    trace.record(limits=TokenLimits.from_provider(provider()),requested_output=200,output_tokens=200,
                 estimated_prompt=300,actual_prompt=100,actual_completion=100)
    assert trace.accounted_tokens == 600
    with pytest.raises(RuntimeError):
        trace.begin_request(estimated_tokens=401)


@pytest.mark.parametrize("source_count", [5, 10])
def test_diverse_selection_preserves_sources_and_late_regions(source_count):
    rows=[]
    for source in range(source_count):
        for page in (1,11,21,91):
            rows.append({"id": f"{source}-{page}", "source_id": str(source), "ordinal": page,
                         "content": (f"Source {source} region {page} explains the central proposition with an important exception and supporting observations from the original study. " * 3),
                         "locator_json": json.dumps({"page":page}), "embedding_json":"[1,0]"})
    plan=plan_context(TokenLimits.from_provider(provider()),"summary",material_tokens=20000,segment_tokens=500)
    state=GenerationContext(provider(),plan,rows,[],ContextUsage())
    token=CURRENT.set(state)
    try:
        selected=retrieval.select_quality_evidence('n',[str(i) for i in range(source_count)],limit=1)
        assert {r['source_id'] for r in selected}=={str(i) for i in range(source_count)}
        assert any(r['ordinal']==91 for r in selected)
        assert len(selected)>1
    finally:CURRENT.reset(token)


def test_coverage_distinguishes_selection_partial_and_complete():
    text="The original evidence contains a precise limitation and a long explanation of why it applies."
    row={"id":"c","source_id":"s","content":text,"locator":{"page":95}}
    plan=plan_context(TokenLimits.from_provider(provider()),"summary",material_tokens=1000)
    state=GenerationContext(provider(),plan,[row],[{"id":"s","filename":"fixture","revision_id":"r"}],ContextUsage(),selected={"c"})
    token=CURRENT.set(state)
    try:
        build=providers.PromptBuild([{"role":"user","content":text[:50]}],metadata={"chunks":[row]})
        mark_sent(build)
        assert report()['sent_segments']==0
        assert report()['partially_sent_segments']==1
        build.messages[0]['content']=text
        mark_sent(build)
        coverage=report([{"chunk_id":"c"}])
        assert coverage['sent_segments']==coverage['cited_segments']==1
        assert coverage['partially_sent_segments']==0
        assert coverage['sources'][0]['regions_sent']==1
        assert region(row)=='页区间：91–100'
    finally:CURRENT.reset(token)


def test_quiz_coverage_does_not_expose_post_answer_evidence():
    from sandevistan_read.study_sessions import public_artifact
    raw={'type':'quiz','payload':{'items':[],'context_usage':{'coverage':{'sent_segments':10,'cited_segments':2,'sources':[{'unsent_regions':['secret answer region']}]}}}}
    public=public_artifact(raw)
    assert public['payload']['context_usage']['coverage']=={'sent_segments':10}
    assert 'sources' in raw['payload']['context_usage']['coverage']


def test_preview_is_local_and_uses_the_execution_planner(monkeypatch):
    from sandevistan_read import app
    from sandevistan_read.schemas import ContextPreviewRequest
    monkeypatch.setattr(app, 'provider_by_id', lambda _: provider())
    body=ContextPreviewRequest(provider_id='fixture',model='fixture',config={'context_window_tokens':1000000,'max_output_tokens':384000})
    result=app.context_preview(body)
    expected=plan_context(TokenLimits.from_provider(provider(1000000,384000)),'podcast').as_dict()
    assert result['plans'][-1]==expected
    assert result['basis']=='按每段约 1000 tokens 估算'
    assert 'api_key' not in json.dumps(result)


def test_revision_scope_stays_pinned_when_live_selection_changes(monkeypatch):
    import hashlib
    from sandevistan_read import services, study
    state=GenerationContext(provider(),plan_context(TokenLimits.from_provider(provider()),'summary'),[],
                            [{'id':'s','filename':'old','revision_id':'old-revision'}],ContextUsage())
    token=CURRENT.set(state)
    try:
        monkeypatch.setattr(services.DB,'fetchone',lambda *args: (_ for _ in ()).throw(AssertionError('live database read')))
        assert services.source_scope('n',None)==study._source_scope('n',None)==['s']
        assert services.scope_hash(['s'])==hashlib.sha256(b'old-revision').hexdigest()
        assert services.source_scope('n',['other'])==[]
    finally:CURRENT.reset(token)


@pytest.mark.parametrize('minutes', [5,20,30,60,120])
def test_long_duration_keeps_evidence_budget(minutes):
    plan = plan_context(TokenLimits.from_provider(provider()), 'podcast', material_tokens=1000000, minutes=minutes)
    assert plan.evidence_tokens >= 1000
    assert plan.total_token_limit <= 300000


@pytest.mark.asyncio
async def test_preparation_salvages_complete_exact_quotes_only(monkeypatch):
    from types import SimpleNamespace
    from sandevistan_read.generation_context import prepare_evidence
    text = 'This original passage explains a central proposition and its important limitations in detail.'
    row = {'id': 'c', 'source_id': 's', 'content': text, 'ordinal': 0}
    plan = plan_context(TokenLimits.from_provider(provider(30720,4096)), 'quiz', material_tokens=1000000)
    state = GenerationContext(provider(), plan, [row], [], ContextUsage())
    valid = {'chunk_id':'c','claim':'A supported claim','quote':text}
    fabricated = {'chunk_id':'c','claim':'A fabricated claim','quote':'This quotation does not occur anywhere in the document.'}
    invisible = {**valid,'chunk_id':'other'}
    content = '{"notes":[' + ','.join(json.dumps(note) for note in [valid,fabricated,invisible]) + ',{"chunk_id":'
    async def response(*args, **kwargs):
        return SimpleNamespace(content=content, build=SimpleNamespace(metadata={'chunks':[row]}))
    monkeypatch.setattr(providers, 'budgeted_chat', response)
    token = CURRENT.set(state)
    try:
        notes = await prepare_evidence([row], 'en')
        assert len(notes) == 1
        assert notes[0]['quote'] == text
        assert notes[0]['chunk_id'] == 'c'
    finally:
        CURRENT.reset(token)


@pytest.mark.asyncio
async def test_generation_is_opt_in_and_pins_provider(monkeypatch):
    from sandevistan_read.generation_context import adaptive_generation, current
    configured = provider()
    row = {'id':'c','source_id':'s','ordinal':0,'content':'A source-grounded explanation of the key proposition, its evidence and limitations. ' * 8}
    class FixtureDatabase:
        def fetchall(self, sql, params):
            assert 'LEFT JOIN chunks' in sql
            return [{**row, 'snapshot_source_id':'s','snapshot_filename':'fixture','snapshot_revision':'r','snapshot_selected':1}]
    monkeypatch.setitem(globals(), 'DB', FixtureDatabase())
    monkeypatch.setitem(globals(), 'active_provider', lambda _: configured)
    @adaptive_generation('summary')
    async def generate(notebook_id, source_ids=None):
        state = current()
        if state:
            configured['model']='changed-in-live-settings'
            assert providers.active_provider('main')['model']=='fixture'
        return {'context_usage':{},'citations':[{'chunk_id':'c'}]}
    conservative = await generate('n')
    assert 'coverage' not in conservative['context_usage']
    configured['config']['context_strategy']='balanced'
    balanced = await generate('n')
    assert balanced['context_usage']['coverage']['candidate_segments']==1
    assert current() is None


@pytest.mark.asyncio
async def test_preparation_stops_before_spending_final_reserve(monkeypatch):
    from sandevistan_read.context_budget import PromptBudget
    plan=plan_context(TokenLimits.from_provider(provider()),'quiz',material_tokens=1000000)
    trace=ContextUsage(total_token_limit=plan.total_token_limit)
    trace.accounted_tokens=plan.total_token_limit-plan.final_reserve_tokens
    state=GenerationContext(provider(),plan,[],[],trace)
    async def unexpected(*args,**kwargs):
        raise AssertionError('External call after reserve exhausted')
    monkeypatch.setattr(providers,'_chat_once',unexpected)
    token=CURRENT.set(state)
    try:
        with pytest.raises(RuntimeError,match='保留预算'):
            await providers.budgeted_chat(lambda _: providers.PromptBuild([{'role':'user','content':'source'}]),stage='context_prepare')
        assert trace.requests==0
        assert trace.accounted_tokens==plan.total_token_limit-plan.final_reserve_tokens
    finally:
        CURRENT.reset(token)


def test_many_page_regions_do_not_crowd_out_another_selected_book():
    rows=[]
    for source in ('large','other'):
        for index in range(120):
            rows.append({'id':f'{source}-{index}','source_id':source,'ordinal':index,
                         'content':('A detailed proposition with supporting evidence, qualifications and concrete examples. ' * 6),
                         'locator_json':json.dumps({'page':index*10+1} if source=='large' else {'section':'One large EPUB section'}),
                         'embedding_json':'[1,0]'})
    plan=plan_context(TokenLimits.from_provider(provider(30720,4096)),'summary',material_tokens=1000000,segment_tokens=500)
    state=GenerationContext(provider(),plan,rows,[],ContextUsage())
    token=CURRENT.set(state)
    try:
        selected=retrieval.select_quality_evidence('n',['large','other'],limit=1)
        counts={source:sum(row['source_id']==source for row in selected) for source in ('large','other')}
        assert counts['large'] >= 10
        assert abs(counts['large']-counts['other']) <= 1
    finally:
        CURRENT.reset(token)


@pytest.mark.asyncio
async def test_failed_preparation_is_not_repeated_by_recovery(monkeypatch):
    from sandevistan_read.generation_context import prepare_evidence
    plan=plan_context(TokenLimits.from_provider(provider(30720,4096)),'quiz',material_tokens=1000000)
    row={'id':'c','source_id':'s','content':'Valid source passage with detailed supporting facts and qualifications. '*5}
    state=GenerationContext(provider(),plan,[row],[],ContextUsage())
    calls=[]
    async def fail(*args,**kwargs):
        calls.append(1)
        raise RuntimeError('Provider unavailable')
    monkeypatch.setattr(providers,'budgeted_chat',fail)
    token=CURRENT.set(state)
    try:
        assert await prepare_evidence([row],'en')==[]
        assert await prepare_evidence([row],'en')==[]
        assert len(calls)==1
    finally:
        CURRENT.reset(token)


def test_preview_rejects_an_override_above_known_model_capacity(monkeypatch):
    from fastapi import HTTPException
    from sandevistan_read import app
    from sandevistan_read.schemas import ContextPreviewRequest
    known={**provider(),'capabilities':{'token_limits':{'model_context_tokens':262144}}}
    monkeypatch.setattr(app,'provider_by_id',lambda _:known)
    with pytest.raises(HTTPException) as error:
        app.context_preview(ContextPreviewRequest(provider_id='fixture',model='fixture',config={'context_window_tokens':1000000,'max_output_tokens':384000}))
    assert error.value.status_code==422
    assert '262144' in error.value.detail


def test_checkpoint_restores_usage_and_does_not_reset_budget():
    from sandevistan_read.generation_context import restore_trace
    saved = ContextUsage(total_token_limit=300000, request_limit=10)
    saved.begin_request(estimated_tokens=90000)
    saved.record_failure()
    restored = ContextUsage(total_token_limit=100000, request_limit=80)
    restore_trace(restored, saved.as_dict())
    assert restored.accounted_tokens == 90000
    assert restored.total_token_limit == 100000
    assert restored.failed_requests == restored.requests == 1
    assert restored.actual_prompt_tokens == 0
    with pytest.raises(RuntimeError, match='token'):
        restored.begin_request(estimated_tokens=10001)


def test_task_qualification_and_explicit_override_share_resolution(monkeypatch):
    from sandevistan_read import context_budget
    monkeypatch.setattr(context_budget, 'QUALIFIED_BALANCED_TASKS', frozenset({'summary'}))
    assert context_budget.context_strategy({}, 'summary') == 'balanced'
    assert context_budget.context_strategy({}, 'podcast') == 'conservative'
    assert context_budget.context_strategy({'config': {'context_strategy': 'conservative'}}, 'summary') == 'conservative'
    assert context_budget.context_strategy({'config': {'context_strategy': 'balanced'}}, 'podcast') == 'balanced'
