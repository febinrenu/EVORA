from eval import latency


def test_percentile_is_nearest_rank_and_defined_for_small_samples():
    assert latency.percentile([5.0], 0.95) == 5.0
    assert latency.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert latency.percentile(list(map(float, range(1, 13))), 0.95) == 12.0   # with 12 points the p95 is the maximum


def test_rows_are_grouped_by_the_planner_route_that_answered():
    rows = [{"source": "fastpath", "ttfa_ms": 50}, {"source": "fastpath", "ttfa_ms": 70},
            {"source": "groq", "ttfa_ms": 900}, {"source": None, "ttfa_ms": 10}]
    out = latency.summarise(rows)
    assert out["fastpath"] == {"n": 2, "p50_ms": 60.0, "p95_ms": 70.0, "max_ms": 70.0}
    assert out["groq"]["n"] == 1 and out["unknown"]["n"] == 1


def test_the_question_set_is_not_all_fast_path_phrasing():
    assert len(latency.FREE_PHRASING) >= 10 and len(set(latency.FREE_PHRASING)) == len(latency.FREE_PHRASING)


def test_stage_summary_reports_each_timed_stage_with_its_own_n():
    rows = [{"stages": {"plan": 10, "retrieve": 100}}, {"stages": {"plan": 30, "retrieve": 300, "verify": 2000}},
            {"stages": {}}]
    out = latency.stage_summary(rows)
    assert out["plan"] == {"n": 2, "p50_ms": 20.0, "p95_ms": 30.0}
    assert out["verify"]["n"] == 1 and out["retrieve"]["p95_ms"] == 300.0


def test_only_the_cloud_planner_counts_as_an_outbound_call():
    rows = [{"source": "fastpath"}, {"source": "cache"}, {"source": "local_llm"}, {"source": "llm"}]
    assert latency.no_network_share(rows) == 0.75
    assert latency.no_network_share([]) is None


def test_the_eval_phrasing_comes_from_the_main_query_file():
    texts = latency.eval_phrasing(5)
    assert len(texts) == 5 and len(set(texts)) == 5 and all(t.endswith("?") for t in texts)


def test_a_question_that_stopped_to_ask_has_no_answer_time_and_is_counted_apart():
    rows = [{"source": "fastpath", "ttfa_ms": 5, "asked": False, "error": None, "stages": {"retrieve": 5}},
            {"source": "fastpath", "ttfa_ms": 1, "asked": True, "error": None, "stages": {}},
            {"source": "fastpath", "ttfa_ms": 1, "asked": False, "error": "boom", "stages": {}}]
    out = latency.condition(rows)
    assert out["n_asked"] == 1 and out["n_errors"] == 1
    assert out["by_route"]["fastpath"]["n"] == 1 and out["by_route"]["fastpath"]["p50_ms"] == 5.0
    assert out["by_stage"]["retrieve"]["n"] == 1
    assert out["no_network_share"] == 1.0          # the share still counts every question asked
