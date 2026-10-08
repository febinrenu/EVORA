import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest
from eval.queries import load_queries

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "meva_to_queries.py"
spec = importlib.util.spec_from_file_location("meva_to_queries", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules["meva_to_queries"] = m  # dataclasses look their module up here
spec.loader.exec_module(m)

TZ = m.parse_tz("+05:30")


def act(name, start, end, extra=None):
    return (f"- {{'act': {{'act2': {{'{name}': 1.0}}, 'actors': [{{'id1': 1, 'timespan': [{{'tsr0': [{start}, {end}]}}]}}], "
            f"'id2': 1, 'src_status': 'good', 'timespan': [{{'tsr0': [{start}, {end}]}}]}}}}\n")


def write_clip(root, cam, activities, date="2018-03-09", start="10-10-00", end="10-15-00", source="kitware"):
    d = root / m.ANNOTATION_SUBPATH / source / date / start[:2]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{date}.{start}.{end}.school.{cam}.activities.yml").write_text("".join(activities))
    (d / f"{date}.{start}.{end}.school.{cam}.types.yml").write_text("- {}\n")  # other files are ignored


@pytest.fixture
def root(tmp_path):
    write_clip(tmp_path, "G340", [act("person_carries_heavy_object", 300, 450), act("vehicle_stops", 3000, 3090),
                                  act("person_transfers_object", 10, 20)])
    write_clip(tmp_path, "G341", [act("vehicle_stops", 60, 120)])
    write_clip(tmp_path, "G342", [act("person_sits_down", 0, 5)], start="11-00-00", end="11-05-00")
    write_clip(tmp_path, "G343", [act("person_sits_down", 0, 5)], source="kitware-meva-training")
    return tmp_path


def test_clip_names_and_times():
    clip = m.parse_clip_name("2018-03-07.17-35-06.17-40-06.school.G339")
    assert (clip.site, clip.camera, clip.date) == ("school", "G339", "2018-03-07")
    assert m.parse_clip_name("not-a-clip") is None
    assert m.clip_start(clip, TZ) == datetime(2018, 3, 7, 17, 35, 6, tzinfo=TZ)
    over = m.parse_clip_name("2018-03-07.23-57-00.00-02-00.school.G339")
    assert m.clip_end(over, TZ) == datetime(2018, 3, 8, 0, 2, 0, tzinfo=TZ)  # crosses midnight


def test_parse_tz():
    assert m.parse_tz("+05:30").utcoffset(None).total_seconds() == 19800
    assert m.parse_tz("-0800").utcoffset(None).total_seconds() == -28800
    with pytest.raises(ValueError):
        m.parse_tz("IST")


def test_activity_files_only_read_the_requested_sources(root):
    files = m.activity_files(root)
    assert sorted(c.camera for c, _ in files) == ["G340", "G341", "G342"]  # not the training source
    both = m.activity_files(root, ("kitware", "kitware-meva-training"))
    assert len(both) == 4
    assert len(m.activity_files(root / m.ANNOTATION_SUBPATH)) == 3  # either the repo root or the MEVA folder works


def test_read_instances_uses_frame_spans(root):
    clip, path = next((c, p) for c, p in m.activity_files(root) if c.camera == "G340")
    got = {(i.activity, i.start_frame, i.end_frame) for i in m.read_instances(clip, path)}
    assert ("person_carries_heavy_object", 300, 450) in got and len(got) == 3


def test_find_windows_ranks_by_camera_count(root):
    rows = m.find_windows(m.activity_files(root), min_cams=1)
    assert rows[0] == ("2018-03-09", "10-10-00", 2, 4) and len(rows) == 2
    assert m.find_windows(m.activity_files(root), min_cams=3) == []


def test_generated_queries_have_exact_ground_truth_windows(root):
    clips = m.window_clips(m.activity_files(root), "2018-03-09", "10-10-00")
    items, skipped = m.build_queries(clips, TZ, split="test", camera_map={"G340": "cam_01"})
    by_id = {i["id"]: i for i in items}
    heavy = by_id["meva_20180309_1010_person_carries_heavy_object"]
    assert heavy["text"] == "Did a person carry something heavy on 9 March between 10:10 and 10:15?"
    assert heavy["split"] == "test" and heavy["expected"]["verdict"] == "yes"
    # frames 300..450 at 30 fps are 10 s .. 15 s into a clip that starts 10:10:00 (+05:30)
    assert heavy["expected"]["hits"] == [{"camera_id": "cam_01", "start": "2018-03-09T10:10:10.000+05:30",
                                          "end": "2018-03-09T10:10:15.000+05:30"}]
    stops = by_id["meva_20180309_1010_person_carries_heavy_object".replace("person_carries_heavy_object",
                                                                          "vehicle_stops")]
    assert [h["camera_id"] for h in stops["expected"]["hits"]] == ["G341", "cam_01"]  # time ordered, mapped ids
    assert skipped == {"person_transfers_object": 1}


def test_negatives_are_activities_absent_from_the_window(root):
    clips = m.window_clips(m.activity_files(root), "2018-03-09", "10-10-00")
    items, _ = m.build_queries(clips, TZ, max_negatives=3)
    negatives = [i for i in items if not i["expected"]["hits"]]
    assert len(negatives) == 3 and all(i["expected"]["verdict"] == "no" and "negative" in i["tags"] for i in negatives)
    present = {"person_carries_heavy_object", "vehicle_stops"}
    assert not any(i["id"].endswith(a) for i in negatives for a in present)


def test_zero_length_spans_get_a_minimum_window_and_hits_are_capped(tmp_path):
    many = [act("vehicle_starts", 100 + i * 40, 100 + i * 40) for i in range(80)]
    write_clip(tmp_path, "G340", many)
    items, _ = m.build_queries(m.window_clips(m.activity_files(tmp_path), "2018-03-09", "10-10-00"), TZ)
    hits = items[0]["expected"]["hits"]
    assert len(hits) == m.MAX_HITS
    t0, t1 = (datetime.fromisoformat(hits[0][k]) for k in ("start", "end"))
    assert (t1 - t0).total_seconds() == pytest.approx(1.0)


def test_empty_input():
    assert m.build_queries([], TZ) == ([], {})


def test_cli_output_loads_in_the_eval_harness(root, tmp_path, capsys):
    out = tmp_path / "q" / "meva_dev.yaml"
    cmap = tmp_path / "map.json"
    cmap.write_text(json.dumps({"G340": "cam_01", "G341": "cam_02"}))
    code = m.main(["generate", "--annotations", str(root), "--date", "2018-03-09", "--start", "10-10-00",
                   "--split", "judge_sim", "--camera-map", str(cmap), "--out", str(out)])
    assert code == 0 and "queries (" in capsys.readouterr().out
    items = load_queries([out])
    assert items and all(i.split == "judge_sim" and i.workspace == "meva" for i in items)
    heavy = next(i for i in items if i.id.endswith("person_carries_heavy_object"))
    assert heavy.is_retrieval and heavy.expected.hits[0].camera_id == "cam_01"
    assert any(i.is_negative for i in items)


def test_cli_errors_and_find(root, tmp_path, capsys):
    assert m.main(["generate", "--annotations", str(tmp_path / "none"), "--date", "x", "--start", "y",
                   "--out", str(tmp_path / "o.yaml")]) == 2
    assert m.main(["generate", "--annotations", str(root), "--date", "2018-03-09", "--start", "09-00-00",
                   "--out", str(tmp_path / "o.yaml")]) == 2
    capsys.readouterr()
    assert m.main(["find", "--annotations", str(root), "--min-cams", "2"]) == 0
    assert "2018-03-09  10-10-00" in capsys.readouterr().out
