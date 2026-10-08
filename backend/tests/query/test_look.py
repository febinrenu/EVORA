import cv2
import numpy as np
import pytest
from contracts.models import QueryPlan, Target

from evora.query import look


def plan(intent="exists", with_target=True, noun="person"):
    targets = [Target(noun=noun, cls=[noun], embed_text=f"a photo of a {noun}")] if with_target else []
    return QueryPlan(intent=intent, targets=targets, action="any")


@pytest.mark.parametrize("intent,with_target,text,expected", [
    ("describe", True, "what is in the room", True),
    ("describe", False, "what colour is the carpet", True),
    ("exists", False, "is anyone talking on a phone", True),                     # no object type: a scene question
    ("exists", True, "is anyone sitting down", True),                            # a posture the tracker cannot know
    ("list", True, "show me people holding a cup", True),
    ("count", False, "how many people are sitting", True),
    ("exists", True, "was there a person at the gate", False),                   # the tracker answers this
    ("exists", True, "did a red car pass through the gate", False),
    ("count", True, "how many people are wearing red", False),                  # colour counting has its own path
    ("count", True, "how many chairs are there", False),                        # the detector counts these
    ("first", True, "when did the first person appear", False),
    ("path", True, "where did he go", False),
    ("standing", False, "alert me when someone is sitting", False),
])
def test_only_questions_the_picture_can_answer_are_sent_to_look(intent, with_target, text, expected):
    assert look.wants_look(plan(intent, with_target), text) is expected


def test_yes_no_and_cannot_tell_read_the_start_of_an_answer():
    assert look.yes_no("Yes, two people are sitting.") is True
    assert look.yes_no("No one is sitting down; everyone is standing.") is False
    assert look.yes_no("No, the room is empty.") is False
    assert look.yes_no("Nobody is on the phone.") is False
    assert look.yes_no("There are three people.") is None
    assert look.cannot_tell("I can't tell from these frames, they are too dark.") is True
    assert look.cannot_tell("Two people are sitting.") is False


def frame(value):
    return np.full((360, 640, 3), value, np.uint8)


def test_the_contact_sheet_lays_frames_out_in_order_with_their_labels():
    sheet = look.contact_sheet([frame(40), frame(120), frame(200)], ["A 01:47:00", "B 01:47:20", "C 01:47:40"])
    image = cv2.imdecode(np.frombuffer(sheet, np.uint8), cv2.IMREAD_COLOR)
    w, h = look.CELL
    assert image.shape == (2 * h, 2 * w, 3)                       # three frames in a 2 x 2 grid
    assert abs(int(image[h // 2 + 20, w // 2].mean()) - 40) < 12                      # A top left
    assert abs(int(image[h // 2 + 20, w + w // 2].mean()) - 120) < 12                 # B top right
    assert abs(int(image[h + h // 2 + 20, w // 2].mean()) - 200) < 12                 # C bottom left
    assert look.contact_sheet([], []) is None
    single = cv2.imdecode(np.frombuffer(look.contact_sheet([frame(90)], ["A 10:00:00"]), np.uint8), cv2.IMREAD_COLOR)
    assert single.shape == (look.CELL[1], look.CELL[0], 3)


def test_the_prompt_names_the_camera_the_frames_and_how_to_decline():
    text = look.prompt_for("is anyone sitting down?", "4p-c0", ["A 01:47:00", "B 01:47:20"])
    assert "4p-c0" in text and "A 01:47:00, B 01:47:20" in text and "is anyone sitting down?" in text
    assert look.NOT_VISIBLE in text and "Do not name people" in text


def test_frames_are_spread_over_the_available_ones():
    rows = [(float(t), f"f{t}.jpg") for t in range(20)]
    picked = look.pick_frames(rows, "cam_01")
    assert [f.t for f in picked] == [0.0, 6.0, 13.0, 19.0] and all(f.camera_id == "cam_01" for f in picked)
    assert [f.path for f in look.pick_frames(rows[:2], "cam_01")] == ["f0.jpg", "f1.jpg"]


class FakeGateway:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def vision_text(self, image_jpeg, prompt, *, local_only=True, max_tokens=64, model=None):
        self.calls.append({"bytes": len(image_jpeg), "prompt": prompt, "local_only": local_only, "model": model})
        return self.reply


@pytest.mark.asyncio
async def test_the_answerer_sends_one_labelled_sheet_to_the_local_model_only(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"f{i}.jpg"), frame(60 * (i + 1)))
    gateway = FakeGateway("  No one is sitting.  ")
    answerer = look.LookAnswerer(gateway, tmp_path, model="qwen3.5:4b")
    frames = [look.Frame("cam_01", 1000.0 + 10 * i, f"f{i}.jpg") for i in range(3)] + [look.Frame("cam_01", 9.0, "gone.jpg")]
    out = await answerer.ask("is anyone sitting?", "Gate", frames)
    assert out == "No one is sitting."
    (call,) = gateway.calls
    assert call["local_only"] is True and call["model"] == "qwen3.5:4b"      # the images show people: never the cloud
    assert "A " in call["prompt"] and "B " in call["prompt"] and "C " in call["prompt"] and "D " not in call["prompt"]
    assert await look.LookAnswerer(FakeGateway(None), tmp_path).ask("q", "Gate", frames[:3]) is None
    assert await answerer.ask("q", "Gate", [look.Frame("cam_01", 1.0, "gone.jpg")]) is None    # nothing readable


@pytest.mark.parametrize("reply,expected", [
    ("6", 6), ("  5 ", 5), ("There are 8 chairs.", 8), ("Eight", 8), ("zero", 0), ("None visible", 0),
    ("No chairs", None), ("no idea", None), ("0", 0), ("about twelve", None), ("", None), (None, None),
    ("I can't tell", None), ("999", None), ("the 7th", None),
])
def test_a_count_is_read_from_a_reply_or_is_nothing(reply, expected):
    assert look.parse_count(reply) == expected


@pytest.mark.asyncio
async def test_the_model_counts_frame_by_frame_and_a_missing_number_is_none(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"f{i}.jpg"), frame(80))
    replies = iter(["6", "five", "no idea"])

    class Counting(FakeGateway):
        async def vision_text(self, image_jpeg, prompt, *, local_only=True, max_tokens=64, model=None):
            await super().vision_text(image_jpeg, prompt, local_only=local_only, max_tokens=max_tokens, model=model)
            return next(replies)

    gateway = Counting(None)
    answerer = look.LookAnswerer(gateway, tmp_path, model="m")
    frames = [look.Frame("cam_01", float(i), f"f{i}.jpg") for i in range(3)] + [look.Frame("cam_01", 9.0, "gone.jpg")]
    # "no idea" and an unreadable frame are no number
    assert await answerer.count("red chairs", frames) == [6, 5, None, None]
    assert all("How many red chairs are visible" in c["prompt"] and c["local_only"] for c in gateway.calls)
    assert len(gateway.calls) == 3                                                 # nothing was sent for the missing frame
