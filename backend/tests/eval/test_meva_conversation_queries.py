import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
spec = importlib.util.spec_from_file_location("meva_conversation_queries", SCRIPTS / "meva_conversation_queries.py")
conv = importlib.util.module_from_spec(spec)
sys.modules["meva_conversation_queries"] = conv
spec.loader.exec_module(conv)

HIT = {"camera_id": "cam_03", "start": "2018-03-09T10:11:00.000+05:30", "end": "2018-03-09T10:11:10.000+05:30"}


def source():
    def item(n, verdict, cls, split="dev"):
        return {"id": f"cap_G328_{n}", "text": f"Was there a {cls} on G328 between 10:1{n} and 10:1{n + 1}?",
                "workspace": "w", "intent": "exists", "split": split, "tags": ["meva", "G328"],
                "expected": {"verdict": verdict, "hits": [HIT] if verdict == "yes" else []}}
    return [item(0, "yes", "person"), item(1, "no", "vehicle"), item(2, "yes", "person"),
            {**item(3, "yes", "person"), "tags": ["meva", "G999"]}]  # a camera with no invented place is skipped


def test_the_first_mention_asks_and_binds_the_place_to_the_camera():
    items = conv.build(source(), {"G328": "cam_03"})
    first = items[0]
    assert first["first_time_requires_clarify"] == ["front lot"]
    assert first["clarify_answer"] == {"camera_id": "cam_03"}
    assert "at the front lot between" in first["text"] and "G328" not in first["text"]
    assert all("first_time_requires_clarify" not in i for i in items[1:])    # later mentions must not ask


def test_later_questions_use_fixed_rewordings_and_keep_the_ground_truth():
    items = conv.build(source(), {"G328": "cam_03"})
    assert [i["tags"][3] for i in items] == ["first_mention", "same_wording", "reworded_no_article",
                                            "reworded_area", "reworded_case"]
    texts = [i["text"] for i in items]
    assert "at front lot between" in texts[2] and "at the front lot area between" in texts[3]
    assert "at The Front Lot between" in texts[4]
    # positives come first, so the one negative is reached by the second-to-last rewording slot (index 2)
    assert items[1]["expected"]["verdict"] == "yes"
    assert items[2]["expected"]["verdict"] == "no" and items[2]["expected"]["hits"] == []
    assert items[0]["expected"]["hits"][0]["camera_id"] == "cam_03" and items[0]["split"] == "dev"
    assert not any("G999" in " ".join(i["tags"]) for i in items)
