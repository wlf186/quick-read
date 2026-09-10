import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('context_eval_policy',Path(__file__).parents[1]/'scripts/evaluate_context_strategy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
Policy=module.EvaluationFailurePolicy


def test_recovery_resets_request_streak_across_cases(tmp_path):
    p=Policy(tmp_path/'state.json')
    p.request(False,500);p.request(False,500)
    p=Policy(p.path);p.before_request();p.request(True)
    for _ in range(2):
        p.request(False,500);p.request(True)
        p.task({'status':'degraded','result':{'delivery_status':'partial','narrative_status':'unverified'}})
    assert p.state['request_errors']==4
    assert p.state['consecutive_tasks']==0 and p.state['stop_reason'] is None


def test_three_requests_stop_across_instances(tmp_path):
    p=Policy(tmp_path/'state.json');p.request(False,500);p.request(False,500)
    p=Policy(p.path);p.request(False,500)
    with pytest.raises(RuntimeError):p.before_request()
    assert p.state['stop_reason']=='three_consecutive_provider_failures'


@pytest.mark.parametrize('artifact',[{}, {'delivery_status':'draft_only'},{'narrative_status':'incomplete'}])
def test_two_final_failures_stop_but_a_delivered_task_resets(tmp_path,artifact):
    p=Policy(tmp_path/'state.json')
    p.task({'status':'failed'})
    p.task({'status':'degraded','result':{'payload':{'delivery_status':'partial','warnings':['duration']}}})
    assert p.state['consecutive_tasks']==0
    for _ in range(2):p.task({'status':'degraded','result':artifact})
    assert p.state['stop_reason']=='two_consecutive_task_failures'


def test_access_error_stops_immediately(tmp_path):
    p=Policy(tmp_path/'state.json');p.request(False,429)
    assert p.state['stop_reason']=='provider_access_or_quota'
