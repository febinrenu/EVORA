import json

import pytest
from eval import report as rp


def metric(value, n):
    return {"value": value, "n": n}


def cap_report(n, **metrics):
    return {"split": "x", "n_queries": n, "metrics": {k: metric(*v) for k, v in metrics.items()}}


@pytest.fixture
def reports(tmp_path):
    def write(name, data):
        (tmp_path / name).write_text(json.dumps(data))

    write("eval_dev_capabilities.json", {
        "ours": {"object": cap_report(9, **{"hit@1": (0.3333, 9)}), "negative": cap_report(8, negative_precision=(1.0, 8))},
        "b0": {"object": cap_report(9, **{"hit@1": (1.0, 9)}), "negative": cap_report(8, negative_precision=(0.0, 8))}})
    write("eval_test_capabilities.json", {
        "ours": {"object": cap_report(8, **{"hit@1": (0.625, 8)}), "negative": cap_report(4, negative_precision=(1.0, 4))},
        "b0": {"object": cap_report(8, **{"hit@1": (1.0, 8)}), "negative": cap_report(4, negative_precision=(0.0, 4))}})
    write("eval_dev.json", {"ours": cap_report(28, **{"hit@1": (0.22, 11)})})
    write("ablation.json", {"full system": {"switch": None, "skipped": None, "report": cap_report(19, mrr=(0.44, 11))},
                            "no C5": {"switch": {"id": "C5"}, "skipped": "needs re-index", "report": None}})
    return tmp_path


def test_pooling_is_weighted_by_the_number_of_queries_not_an_average_of_averages(reports):
    rep = rp.build_report(reports, reports / "none.json")
    obj = rep["pooled"]["ours"]["object"]["hit@1"]
    assert obj["n"] == 17 and obj["value"] == pytest.approx((0.3333 * 9 + 0.625 * 8) / 17, abs=1e-3)
    assert obj["splits"] == ["dev", "test"]
    assert rep["pooled"]["b0"]["negative"]["negative_precision"] == {"value": 0.0, "n": 12, "splits": ["dev", "test"]}
    assert rep["pooled"]["ours"]["negative"]["negative_precision"]["value"] == 1.0


def test_pool_ignores_splits_that_lack_the_metric():
    per_split = {"dev": {"ours": {"object": {"metrics": {"hit@1": {"value": 0.5, "n": 4}}}}},
                 "test": {"ours": {"object": {"metrics": {"hit@1": {"value": None, "n": 0}}}}},
                 "judge_sim": {}}
    assert rp.pool(per_split, "ours", "object", "hit@1") == {"value": 0.5, "n": 4, "splits": ["dev"]}
    assert rp.pool(per_split, "ours", "negative", "negative_precision") == {"value": None, "n": 0, "splits": []}


def test_the_report_lists_what_was_not_evaluated_and_what_was(reports):
    rep = rp.build_report(reports, reports / "none.json")
    assert rep["capabilities_evaluated"] == ["negative", "object"]
    assert set(rep["not_evaluated"]) >= {"colour", "carrying", "path", "count"}
    assert "object" not in rep["not_evaluated"]
    assert all(len(reason) > 20 for reason in rep["not_evaluated"].values())


def test_a_capability_that_gets_ground_truth_leaves_the_not_evaluated_list(tmp_path):
    (tmp_path / "eval_dev_capabilities.json").write_text(json.dumps({"ours": {"colour": cap_report(5, **{"hit@1": (0.8, 5)})}}))
    rep = rp.build_report(tmp_path, tmp_path / "none.json")
    assert "colour" not in rep["not_evaluated"] and "colour" in rep["capabilities_evaluated"]


def test_the_activity_diagnostic_is_kept_apart_from_the_capabilities(tmp_path):
    (tmp_path / "eval_dev_capabilities.json").write_text(json.dumps({
        "ours": {"object": cap_report(2), "activity (diagnostic)": cap_report(11)}}))
    rep = rp.build_report(tmp_path, tmp_path / "none.json")
    assert rep["capabilities_evaluated"] == ["object"] and rep["diagnostics"] == ["activity (diagnostic)"]


def test_caveats_travel_with_the_data(reports):
    rep = rp.build_report(reports, reports / "none.json")
    assert any("conservative" in x for x in rep["limits"]) and any("frozen" in x for x in rep["limits"])
    assert any("recall" in x for x in rep["not_shown"]) and rep["supported_claims"]
    assert [a["label"] for a in rep["ablations"]] == ["full system", "no C5"]
    assert rep["ablations"][1]["skipped"] == "needs re-index" and rep["ablations"][1]["metrics"] is None


def test_the_frozen_settings_are_recorded_and_match_the_code_defaults():
    from evora.query.fuse import DEFAULT_WEIGHTS, Calibration
    from evora.query.router import RouterConfig

    frozen = json.loads(rp.FROZEN_FILE.read_text())
    assert frozen["calibration"]["midpoint"] == Calibration().midpoint
    assert frozen["calibration"]["scale"] == Calibration().scale
    assert frozen["accept_threshold"] == RouterConfig().accept
    assert frozen["fusion_weights"] == DEFAULT_WEIGHTS  # if a default changes, the freeze record must be revisited


def test_load_report_gives_the_empty_shape_before_any_evaluation(tmp_path):
    assert rp.load_report(tmp_path / "missing.json") == {"eval": None, "ablations": None}
    saved = tmp_path / "report.json"
    saved.write_text(json.dumps({"pooled": {}}))
    assert rp.load_report(saved) == {"pooled": {}}


def test_cli_writes_the_file_and_reports_when_there_is_nothing(reports, tmp_path, capsys):
    out = tmp_path / "out.json"
    assert rp.main(["--reports", str(reports), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "pooled ours negative: negative_precision 1.0 (n=12)" in printed
    assert json.loads(out.read_text())["pooled"]["ours"]["object"]["hit@1"]["n"] == 17
    empty = tmp_path / "empty"
    empty.mkdir()
    assert rp.main(["--reports", str(empty)]) == 2


def test_the_chance_level_is_computed_from_the_null_system_not_typed(tmp_path):
    (tmp_path / "eval_dev_capabilities.json").write_text(json.dumps({
        "ours": {"object": cap_report(9, **{"hit@1": (0.33, 9)})},
        "null": {"object": cap_report(9, **{"hit@1": (0.44, 9)})}}))
    rep = rp.build_report(tmp_path, tmp_path / "none.json")
    assert any("Chance level" in x and "0.44" in x and "n=9" in x for x in rep["limits"])
    without = tmp_path / "other"
    without.mkdir()
    (without / "eval_dev_capabilities.json").write_text(json.dumps({"ours": {"object": cap_report(9)}}))
    assert not any("Chance level" in x for x in rp.build_report(without, without / "none.json")["limits"])


def test_the_original_frozen_results_are_reported_beside_the_current_ones(tmp_path):
    v1 = tmp_path / "frozen_v1"
    v1.mkdir()
    (tmp_path / "eval_dev_capabilities.json").write_text(json.dumps({"b0": {"object": cap_report(9, **{"hit@1": (1.0, 9)})}}))
    (v1 / "eval_dev_capabilities.json").write_text(json.dumps({"b0": {"object": cap_report(9, **{"hit@1": (0.8, 9)})}}))
    rep = rp.build_report(tmp_path, tmp_path / "none.json")
    assert rep["pooled"]["b0"]["object"]["hit@1"]["value"] == 1.0
    assert rep["pooled_as_first_frozen"]["b0"]["object"]["hit@1"]["value"] == 0.8


def test_the_claims_no_longer_include_what_the_chance_baseline_disproved():
    text = " ".join(rp.SUPPORTED).lower()
    assert "timestamp" not in text and "camera is returned" not in text  # both were overstated against chance
    assert any("nothing there" in c for c in rp.SUPPORTED)
    assert any("chance" in x for x in rp.NOT_SHOWN) and any("track-centric" in x for x in rp.NOT_SHOWN)


def test_the_post_freeze_fix_is_recorded_with_where_the_original_results_are():
    frozen = json.loads(rp.FROZEN_FILE.read_text())
    fix = frozen["post_freeze_fixes"][0]
    assert "no threshold or weight changed" in fix["kind"] and fix["original_results"] == "eval/reports/frozen_v1"


def test_extended_sets_are_pooled_with_their_n_and_reasks_are_summed(tmp_path):
    for folder, files in {
        "sparse": {"eval_dev_capabilities.json": {"ours": {"object": cap_report(
            4, **{"hit@1": (0.5, 4), "timestamp_error_s": (2.0, 3)})}},
                   "eval_test_capabilities.json": {"ours": {"object": cap_report(2, **{"hit@1": (1.0, 2)})}},
                   "eval_dev.json": {"ours": cap_report(4)}, "eval_test.json": {"ours": cap_report(2)}},
        "conversations": {"eval_dev.json": {"ours": cap_report(
            10, ask_precision=(0.75, 4), ask_recall=(1.0, 3), reask_count=(1.0, 10))},
                          "eval_test.json": {"ours": cap_report(
                              5, ask_precision=(1.0, 2), ask_recall=(1.0, 2), reask_count=(0.0, 5))}},
    }.items():
        (tmp_path / folder).mkdir()
        for name, data in files.items():
            (tmp_path / folder / name).write_text(json.dumps(data))
    ext = rp.build_extended(tmp_path)
    low = ext["low_chance_object"]["pooled"]["ours"]
    assert low["object.hit@1"] == {"value": pytest.approx(0.6667, abs=1e-3), "n": 6, "splits": ["dev", "test"]}
    assert ext["low_chance_object"]["timestamp_error_s_by_split"]["dev"]["ours"]["value"] == 2.0  # kept per split
    ask = ext["conversations"]["pooled"]["ours"]
    assert ask["ask_precision"]["value"] == pytest.approx((0.75 * 4 + 1.0 * 2) / 6, abs=1e-3)
    assert ask["reask_count"] == {"value": 1.0, "n": 15, "splits": ["dev", "test"]}   # a sum over 15 queries
    assert rp.build_extended(tmp_path / "nowhere") == {}


def test_the_report_states_lenient_and_strict_hit_side_by_side(tmp_path):
    def caps(lenient, strict):
        return {"object": cap_report(10, **{"hit@1": (lenient, 10), "hit@1_strict": (strict, 10)})}
    (tmp_path / "eval_dev_capabilities.json").write_text(json.dumps(
        {"ours": caps(0.5, 0.5), "b0": caps(1.0, 0.6), "null": caps(0.5, 0.4)}))
    note = [x for x in rp.build_report(tmp_path, tmp_path / "none.json")["limits"] if x.startswith("Hit@1 is reported")][0]
    assert "ours 0.50 to 0.50" in note and "frame baseline 1.00 to 0.60" in note and "random moment 0.50 to 0.40" in note
    assert "n=10" in note and "Compare systems on the strict score" in note
    bare = rp.build_report(tmp_path / "empty", tmp_path / "none.json")["limits"]
    assert any(x.startswith("Hit@1 is reported two ways.") for x in bare)           # no data: the definition still shows
