import pytest

from fraud import eval_final as ef


def test_final_runs_once(tmp_path):
    (tmp_path / "20261104-090000_eval_final-logreg-amt_final0.jsonl").write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        ef.refuse_second_run("20261104-100000", tmp_path)


def test_the_same_eval_may_resume(tmp_path):
    (tmp_path / "20261104-090000_eval_final-logreg-amt_final0.jsonl").write_text("{}")
    ef.refuse_second_run("20261104-090000", tmp_path)


def test_traces_of_other_windows_do_not_block_it(tmp_path):
    (tmp_path / "20261103-235132_eval_tab-all-amt_tabular0.jsonl").write_text("{}")
    ef.refuse_second_run("20261104-090000", tmp_path)


def test_the_setups_are_the_registered_ones_and_no_more_than_three_candidates():
    candidates = [s for s in ef.SETUPS if s not in (ef.BASELINE, ef.ev.APPROVE_ALL)]
    assert candidates == [ef.SIGNAL_AMT, ef.SIGNAL_CUST, ef.RICH_AMT] and len(candidates) <= 3
