import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest
from eval.queries import load_queries

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("meva_capability_queries", SCRIPTS / "meva_capability_queries.py")
m = importlib.util.module_from_spec(spec)
sys.modules["meva_capability_queries"] = m
spec.loader.exec_module(m)
mq = sys.modules["meva_to_queries"]  # imported by the script above

TZ = m.parse_tz("+05:30")
FPS = 30


def write_clip(root, cam, actors, date="2018-03-09", start="10-10-00", end="10-15-00"):
    """actors: {id: (class, [(first_frame, last_frame), ...])}"""
    d = root / mq.ANNOTATION_SUBPATH / "kitware" / date / start[:2]
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{date}.{start}.{end}.school.{cam}"
    types, geom = [], []
    for aid, (cls, spans) in actors.items():
        types.append(f"- {{'types': {{'cset3': {{'{cls}': 1.0}}, 'id1': {aid}}}}}\n")
        frame = 0
        for a, b in spans:
            for f in range(a, b + 1, 5):
                geom.append(f"- {{'geom': {{'g0': '100 200 150 300', 'id0': {frame}, 'id1': {aid}, 'keyframe': True, "
                            f"'ts0': {f}}}}}\n")
                frame += 1
    (d / f"{stem}.types.yml").write_text("".join(types))
    (d / f"{stem}.geom.yml").write_text("".join(geom))
    (d / f"{stem}.activities.yml").write_text("- {}\n")


def secs(n):
    return n * FPS


@pytest.fixture
def root(tmp_path):
    # G340: a person on screen 5-25 s and another 70-80 s, one car 130-140 s
    write_clip(tmp_path, "G340", {
        1: ("person", [(secs(5), secs(25))]),
        2: ("person", [(secs(70), secs(80))]),
        3: ("vehicle", [(secs(130), secs(140))]),
        4: ("bag", [(secs(5), secs(10))]),
    })
    # G341: people only, so no vehicle is annotated anywhere in its clip
    write_clip(tmp_path, "G341", {1: ("person", [(secs(40), secs(50))])})
    return tmp_path


def build(root, cams=("G340",), **kw):
    cmap = {c: f"cam_{i + 1:02d}" for i, c in enumerate(cams)}
    return m.build_items(root, "2018-03-09", "10-10-00", TZ, cmap, "w", **kw)


def find(items, kind, cls):
    return [i for i in items if i["id"].endswith(f"_{kind}_{cls}")]


def starts(items):
    return sorted(int(i["id"].split("_")[4]) for i in items)


def test_actors_are_read_with_their_classes_and_spans(root):
    d = root / mq.ANNOTATION_SUBPATH / "kitware" / "2018-03-09" / "10"
    actors = m.read_actors(d / "2018-03-09.10-10-00.10-15-00.school.G340.types.yml",
                           d / "2018-03-09.10-10-00.10-15-00.school.G340.geom.yml")
    assert {a.id: a.cls for a in actors.values()} == {1: "person", 2: "person", 3: "vehicle", 4: "bag"}
    assert actors[1].segments() == [(secs(5), secs(25))]


def test_a_gap_of_a_second_or_less_is_one_appearance_but_a_longer_one_is_two():
    a = m.Actor(1, "person", [0, 10, 20, 50, 60, 200])
    assert a.segments() == [(0, 60), (200, 200)]  # 29-frame gaps merge, a 140-frame gap does not


def test_whole_windows_only():
    assert m.windows_of(300.0) == [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300)]
    assert m.windows_of(301.0) == m.windows_of(300.0)  # no one-second sliver at the end


def test_object_queries_carry_exact_hits_clipped_to_the_window(root):
    items = build(root)
    people = {int(i["id"].split("_")[4]): i for i in find(items, "object", "person")}
    first = people[0]
    assert first["text"] == "Was there a person on G340 between 10:10 and 10:11?"
    assert first["expected"]["verdict"] == "yes" and first["intent"] == "exists"
    hit = first["expected"]["hits"][0]
    assert hit["camera_id"] == "cam_01"
    assert datetime.fromisoformat(hit["start"]) == datetime(2018, 3, 9, 10, 10, 5, tzinfo=TZ)
    assert datetime.fromisoformat(hit["end"]) == datetime(2018, 3, 9, 10, 10, 25, tzinfo=TZ)
    assert people[60]["expected"]["hits"][0]["start"].startswith("2018-03-09T10:11:10")


def test_counts_are_not_generated_because_annotated_counts_are_lower_bounds(root):
    items = build(root, cams=("G340", "G341"))
    assert not [i for i in items if i["intent"] == "count" or "cap:count" in i["tags"]]
    assert "count" in m.UNSUPPORTED and "lower bound" in m.UNSUPPORTED["count"]


def test_a_window_merely_empty_of_annotated_actors_is_not_called_empty(root):
    # G340 has people and a car somewhere in its clip: a quiet minute may still hold an unlabelled parked car
    items = build(root)
    assert find(items, "negative", "vehicle") == [] and find(items, "negative", "person") == []


def test_negatives_exist_only_for_a_class_absent_from_the_whole_clip(root):
    items = build(root, cams=("G340", "G341"))
    vehicle_negs = find(items, "negative", "vehicle")
    assert {i["tags"][2] for i in vehicle_negs} == {"G341"}  # only the camera with no annotated vehicle at all
    assert starts(vehicle_negs) == [0, 60, 120, 180]  # the first four whole windows (per_kind caps them)
    assert all(i["expected"] == {"verdict": "no", "hits": []} and i["intent"] == "exists" for i in vehicle_negs)
    assert find(items, "negative", "person") == []


def test_a_brief_flicker_is_neither_presence_nor_a_reason_to_build_a_person_negative(tmp_path):
    write_clip(tmp_path, "G340", {1: ("person", [(secs(61), secs(61) + 10)])})  # a third of a second in 60-120 s
    items = m.build_items(tmp_path, "2018-03-09", "10-10-00", TZ, {"G340": "cam_01"}, "w")
    assert find(items, "object", "person") == []  # too brief to be a positive
    assert find(items, "negative", "person") == []  # the class is annotated in this clip, so no person negatives
    assert len(find(items, "negative", "vehicle")) == 4  # but no vehicle was ever annotated


def test_only_mapped_cameras_and_every_item_is_tagged_with_a_capability(root):
    write_clip(root, "G999", {1: ("person", [(0, secs(20))])})
    items = build(root, cams=("G340", "G341"))
    assert items and all(i["tags"][1].startswith("cap:") and "G999" not in i["tags"] for i in items)
    assert {i["tags"][1] for i in items} == {"cap:object", "cap:negative"}
    assert m.build_items(root, "2018-03-09", "10-10-00", TZ, {"G111": "x"}, "w") == []


def test_the_capabilities_the_data_cannot_ground_are_declared_with_a_reason():
    assert set(m.UNSUPPORTED) == {"count", "carrying", "colour", "path"}
    assert all(len(reason) > 30 for reason in m.UNSUPPORTED.values())


def test_splits_are_by_camera_so_a_test_scene_is_never_a_dev_scene():
    items = []
    for cam, n in [("G1", 10), ("G2", 8), ("G3", 5), ("G4", 3)]:
        items += [{"id": f"{cam}_{i}", "tags": ["meva", "cap:object", cam], "split": "dev"} for i in range(n)]
    counts = m.assign_by_camera(items, {"dev": 0.5, "test": 0.3, "judge_sim": 0.2})
    assert sum(counts.values()) == 26 and all(v > 0 for v in counts.values())
    seen: dict[str, set[str]] = {}
    for it in items:
        seen.setdefault(it["tags"][2], set()).add(it["split"])
    assert all(len(s) == 1 for s in seen.values())  # each camera lives in exactly one split


def test_cameras_already_examined_can_be_pinned_to_dev_so_held_out_splits_stay_unseen():
    items = []
    for cam, n in [("G1", 10), ("G2", 8), ("G3", 5), ("G4", 3), ("G5", 2)]:
        items += [{"id": f"{cam}_{i}", "tags": ["meva", "cap:object", cam], "split": "dev"} for i in range(n)]
    m.assign_by_camera(items, {"dev": 0.5, "test": 0.3, "judge_sim": 0.2}, pinned={"G3": "dev", "G4": "dev"})
    by_cam = {it["tags"][2]: it["split"] for it in items}
    assert by_cam["G3"] == "dev" and by_cam["G4"] == "dev"
    assert {by_cam["G1"], by_cam["G2"], by_cam["G5"]} <= {"test", "judge_sim"}  # nothing else leaks into dev
    assert {"test", "judge_sim"} <= set(by_cam.values())


def test_cli_writes_a_file_the_harness_loads(root, tmp_path, capsys):
    cmap = tmp_path / "map.json"
    cmap.write_text(json.dumps({"G340": "cam_01", "G341": "cam_02"}))
    out = tmp_path / "caps.yaml"
    code = m.main(["--annotations", str(root), "--date", "2018-03-09", "--start", "10-10-00",
                   "--camera-map", str(cmap), "--splits", "dev:1,test:1", "--dev-cameras", "G340", "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "by capability:" in printed and "unsupported - colour" in printed and "unsupported - count" in printed
    items = load_queries([out])
    assert {i.split for i in items} == {"dev", "test"} and any(i.is_negative for i in items)
    assert all(i.split == "dev" for i in items if "G340" in i.tags)  # the pinned camera
    assert m.main(["--annotations", str(root), "--date", "2018-03-09", "--start", "09-00-00",
                   "--camera-map", str(cmap), "--out", str(tmp_path / "none.yaml")]) == 2


def test_coverage_is_the_share_of_the_window_actors_occupy_counting_overlaps_once():
    assert m.coverage([(0.0, 30.0)]) == pytest.approx(0.5)
    assert m.coverage([(0.0, 20.0), (10.0, 30.0)]) == pytest.approx(0.5)      # the overlap is not counted twice
    assert m.coverage([(0.0, 10.0), (40.0, 50.0)]) == pytest.approx(20 / 60)
    assert m.coverage([]) == 0.0


def test_max_coverage_keeps_only_windows_where_chance_is_weak(root):
    persons = find(build(root, max_coverage=0.2), "object", "person")
    assert starts(persons) == [60]                      # 10 s of 60; the 20 s window at 0-60 is 0.33
    both = find(build(root, max_coverage=0.5), "object", "person")
    assert starts(both) == [0, 60]
    assert find(build(root), "object", "person")        # default keeps every window, as before


def test_negatives_can_be_left_out(root):
    assert not [i for i in build(root, cams=("G340", "G341"), with_negatives=False) if "negative" in i["id"]]
    assert [i for i in build(root, cams=("G340", "G341")) if "negative" in i["id"]]
