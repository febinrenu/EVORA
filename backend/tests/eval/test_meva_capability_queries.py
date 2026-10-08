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
    # G340: a person on screen 5-25 s and another 70-80 s; one car 130-140 s; the first minute has one person
    write_clip(tmp_path, "G340", {
        1: ("person", [(secs(5), secs(25))]),
        2: ("person", [(secs(70), secs(80))]),
        3: ("vehicle", [(secs(130), secs(140))]),
        4: ("bag", [(secs(5), secs(10))]),
    })
    return tmp_path


def build(root, **kw):
    return m.build_items(root, "2018-03-09", "10-10-00", TZ, {"G340": "cam_01"}, "w", **kw)


def find(items, kind, cls):
    return [i for i in items if i["id"].endswith(f"_{kind}_{cls}")]


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
    people = {i["id"].split("_")[4]: i for i in find(items, "object", "person")}  # keyed by window start
    first = people["0"]
    assert first["text"] == "Was there a person on G340 between 10:10 and 10:11?"
    assert first["expected"]["verdict"] == "yes" and first["intent"] == "exists"
    hit = first["expected"]["hits"][0]
    assert hit["camera_id"] == "cam_01"
    start = datetime.fromisoformat(hit["start"])
    end = datetime.fromisoformat(hit["end"])
    assert start == datetime(2018, 3, 9, 10, 10, 5, tzinfo=TZ) and end == datetime(2018, 3, 9, 10, 10, 25, tzinfo=TZ)
    assert people["60"]["expected"]["hits"][0]["start"].startswith("2018-03-09T10:11:10")


def test_counts_are_distinct_actors_in_the_window(root):
    items = build(root)
    counts = {i["id"].split("_")[4]: i["expected"]["count"] for i in find(items, "count", "person")}
    assert counts == {"0": 1, "60": 1}  # windows with no person are not count queries (they are negatives)
    vehicles = find(items, "count", "vehicle")
    assert [i["expected"]["count"] for i in vehicles] == [1] and vehicles[0]["intent"] == "count"


def test_negatives_are_windows_with_no_actor_of_that_class_at_all(root):
    items = build(root)
    vehicle_windows = sorted(int(i["id"].split("_")[4]) for i in find(items, "negative", "vehicle"))
    assert vehicle_windows == [0, 60, 180, 240]  # the car is only in 120-180 (frames 130-140 s)
    assert all(i["expected"] == {"verdict": "no", "hits": []} for i in find(items, "negative", "vehicle"))
    person_neg = sorted(int(i["id"].split("_")[4]) for i in find(items, "negative", "person"))
    assert person_neg == [120, 180, 240]


def test_a_brief_flicker_is_neither_presence_nor_proof_of_absence(tmp_path):
    write_clip(tmp_path, "G340", {1: ("person", [(secs(61), secs(61) + 10)])})  # a third of a second in 60-120 s
    items = m.build_items(tmp_path, "2018-03-09", "10-10-00", TZ, {"G340": "cam_01"}, "w")
    assert find(items, "object", "person") == []  # too brief to be a positive...
    empty = sorted(int(i["id"].split("_")[4]) for i in find(items, "negative", "person"))
    assert empty == [0, 120, 180, 240]  # ...and that window is too unclear to call empty, so it is left out


def test_only_mapped_cameras_and_every_item_is_tagged_with_a_capability(root):
    write_clip(root, "G999", {1: ("person", [(0, secs(20))])})
    items = build(root)
    assert items and all(i["tags"][1].startswith("cap:") and "G999" not in i["tags"] for i in items)
    assert {i["tags"][1] for i in items} == {"cap:object", "cap:count", "cap:negative"}
    assert m.build_items(root, "2018-03-09", "10-10-00", TZ, {"G111": "x"}, "w") == []


def test_the_capabilities_the_data_cannot_ground_are_declared():
    assert set(m.UNSUPPORTED) == {"carrying", "colour", "path"}
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


def test_cli_writes_a_file_the_harness_loads(root, tmp_path, capsys):
    cmap = tmp_path / "map.json"
    cmap.write_text(json.dumps({"G340": "cam_01"}))
    out = tmp_path / "caps.yaml"
    code = m.main(["--annotations", str(root), "--date", "2018-03-09", "--start", "10-10-00",
                   "--camera-map", str(cmap), "--splits", "dev:1", "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "by capability:" in printed and "unsupported - colour" in printed and "unsupported - carrying" in printed
    items = load_queries([out])
    assert {i.split for i in items} == {"dev"} and any(i.is_negative for i in items)
    assert any(i.intent == "count" and i.expected.count is not None for i in items)
    assert m.main(["--annotations", str(root), "--date", "2018-03-09", "--start", "09-00-00",
                   "--camera-map", str(cmap), "--out", str(tmp_path / "none.yaml")]) == 2
