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
