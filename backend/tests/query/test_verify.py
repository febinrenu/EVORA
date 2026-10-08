import numpy as np
import pytest
from contracts.models import Evidence, QueryPlan, Target

from evora.llm.schemas import LLMError
from evora.query.verify import (
    SheetUnavailable,
    Verifier,
    VerifyConfig,
    build_contact_sheet,
    describe_target,
    questions_for,
)


def plan(embed="a photo of a red car"):
    return QueryPlan(intent="exists", targets=[Target(noun="car", cls=["car"], embed_text=embed)])


def ev(eid):
    return Evidence(id=eid, camera_id="c", camera_name="C", t_start=0, t_end=1, t_peak=0.5, offset_s=0.5,
                    thumb_url="t", clip_url="c", score=0.5)


class Images:
    def __init__(self, missing=()):
        self.missing, self.asked = set(missing), []

    def crop_for(self, e):
        self.asked.append(e.id)
        return None if e.id in self.missing else f"jpeg-{e.id}".encode()


class Vision:
    def __init__(self, answers=None, error=None):
        self.answers, self.error, self.calls = answers, error, []

    async def vision_yesno(self, image, questions):
        self.calls.append((image, questions))
        if self.error:
            raise self.error
        return self.answers if self.answers is not None else [True] * len(questions)


def fake_sheet(images, cell):
    return b"SHEET:" + b"|".join(images)


async def run(verifier, evidence):
    return [r async for r in verifier.verify(plan(), evidence)]


def test_describe_target_and_questions():
    assert describe_target(plan()) == "a red car"
    assert describe_target(plan("a photo of a person carrying a large bag")) == "a person carrying a large bag"
    assert describe_target(QueryPlan(intent="describe")) == "the object being searched for"
    assert questions_for(plan(), 2) == ["Does image 1 clearly show a red car?", "Does image 2 clearly show a red car?"]


@pytest.mark.asyncio
async def test_one_vision_call_for_the_whole_sheet_and_results_in_rank_order():
    vision = Vision([True, False, None])
    out = await run(Verifier(vision, Images(), sheet_builder=fake_sheet), [ev("a"), ev("b"), ev("c")])
    assert out == [("a", True), ("b", False), ("c", None)]
    assert len(vision.calls) == 1
    sheet, questions = vision.calls[0]
    assert sheet == b"SHEET:jpeg-a|jpeg-b|jpeg-c" and len(questions) == 3


@pytest.mark.asyncio
async def test_evidence_without_an_image_is_unknown_and_not_numbered_on_the_sheet():
    vision = Vision([True, False])
    out = await run(Verifier(vision, Images(missing={"b"}), sheet_builder=fake_sheet), [ev("a"), ev("b"), ev("c")])
    assert out == [("a", True), ("b", None), ("c", False)]
    assert vision.calls[0][0] == b"SHEET:jpeg-a|jpeg-c" and len(vision.calls[0][1]) == 2


@pytest.mark.asyncio
async def test_no_images_means_no_model_call():
    vision = Vision()
    out = await run(Verifier(vision, Images(missing={"a", "b"}), sheet_builder=fake_sheet), [ev("a"), ev("b")])
    assert out == [("a", None), ("b", None)] and vision.calls == []


@pytest.mark.asyncio
async def test_only_the_top_items_are_verified():
    vision = Vision()
    images = Images()
    items = [ev(str(i)) for i in range(15)]
    out = await run(Verifier(vision, images, VerifyConfig(max_items=4), fake_sheet), items)
    assert [i for i, _ in out] == ["0", "1", "2", "3"] and images.asked == ["0", "1", "2", "3"]


@pytest.mark.asyncio
async def test_model_or_sheet_failure_yields_unknown_not_an_exception():
    boom = Verifier(Vision(error=LLMError("down")), Images(), sheet_builder=fake_sheet)
    assert await run(boom, [ev("a")]) == [("a", None)]

    def no_cv2(images, cell):
        raise SheetUnavailable("no opencv")

    assert await run(Verifier(Vision(), Images(), sheet_builder=no_cv2), [ev("a"), ev("b")]) == [("a", None), ("b", None)]


@pytest.mark.asyncio
async def test_a_short_answer_list_leaves_the_rest_unknown():
    out = await run(Verifier(Vision([True]), Images(), sheet_builder=fake_sheet), [ev("a"), ev("b")])
    assert out == [("a", True), ("b", None)]


# ------------------------------------------------------------ the real sheet
cv2 = pytest.importorskip("cv2")


def jpeg(bgr, w=60, h=100):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


def decode(data):
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def test_contact_sheet_layout_numbering_and_aspect():
    cell = 100
    blue, green, red, white = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)
    sheet = decode(build_contact_sheet([jpeg(blue), jpeg(green), jpeg(red), jpeg(white)], cell_px=cell, cols=3))
    assert sheet.shape[:2] == (2 * cell, 3 * cell)  # four images need two rows of three
    centre = lambda r, c: tuple(int(v) for v in sheet[r * cell + cell // 2, c * cell + cell // 2])  # noqa: E731
    assert centre(0, 0)[0] > 200 and centre(0, 0)[1] < 60  # image 1 (blue) top left
    assert centre(0, 1)[1] > 200 and centre(0, 1)[2] < 60  # image 2 (green) next to it
    assert centre(0, 2)[2] > 200 and centre(0, 2)[0] < 60  # image 3 (red)
    assert min(centre(1, 0)) > 200                          # image 4 (white) on the second row
    assert centre(1, 1) == (255, 255, 255)                  # unused cells stay blank
    badge = sheet[2:30, 2:30]
    assert badge.mean() < 200 and (badge > 200).any()       # a dark badge with a light number in it


def test_sheet_keeps_tall_images_inside_their_cell():
    sheet = decode(build_contact_sheet([jpeg((0, 0, 255), w=20, h=200)], cell_px=100))
    assert sheet.shape[:2] == (100, 300)
    red_cols = np.where((sheet[:, :, 2] > 200) & (sheet[:, :, 0] < 60))[1]
    assert red_cols.min() > 30 and red_cols.max() < 70  # narrow strip centred in its cell, not stretched


def test_sheet_errors_are_reported_not_raised_as_cv_errors():
    with pytest.raises(SheetUnavailable):
        build_contact_sheet([])
    with pytest.raises(SheetUnavailable, match="could not be decoded"):
        build_contact_sheet([b"not an image"])
