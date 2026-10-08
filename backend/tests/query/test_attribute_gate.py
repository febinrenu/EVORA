from contracts.models import QueryPlan, Target

from evora.query.fuse import BM25
from evora.query.logic import Match
from evora.query.retrieve import Retriever
from evora.query.router import _backed_by_attributes


def plan(intent="count", attrs=("brown",)):
    target = Target(noun="person", cls=["person"], attributes=list(attrs), embed_text="a person wearing a brown shirt")
    return QueryPlan(intent=intent, targets=[target])


def match(tid, *why):
    return Match(tid, "cam_01", 0.0, 0.0, 1.0, 0.8, None, None, tuple(why))


def test_a_count_only_includes_people_something_backs():
    notes: list[str] = []
    accepted = [match("a", "caption match"), match("b", "siglip 0.09"), match("c", "colour brown 0.80")]
    kept = _backed_by_attributes(plan("count"), accepted, notes)
    assert [m.track_id for m in kept] == ["a", "c"]
    assert notes == ["1 more could not be checked for brown."]


def test_first_uses_backed_tracks_and_falls_back_when_none_is_backed():
    accepted = [match("early", "siglip 0.09"), match("later", "caption match")]
    assert [m.track_id for m in _backed_by_attributes(plan("first"), accepted, [])] == ["later"]
    unbacked = [match("x", "siglip 0.09")]
    assert _backed_by_attributes(plan("first"), unbacked, []) == unbacked  # reported as "can't confirm", not as found


def test_questions_without_attributes_are_untouched():
    accepted = [match("a", "siglip 0.09")]
    assert _backed_by_attributes(plan("count", attrs=()), accepted, []) == accepted


def verdict(caption, attrs=None):
    bm25 = BM25({"t": caption}) if caption else None
    target = plan().targets[0]
    return Retriever._colour_verdict(bm25, "t", target, attrs or {})


def test_caption_colour_verdicts():
    assert verdict("a person in a brown top and beige trousers") is True
    assert verdict("a person in a black top and brown trousers") is False   # brown, but on the trousers
    assert verdict("a person in a dark top and dark trousers") is None      # dark is not a colour that contradicts brown
    assert verdict("a person in a gray top") is False
    assert verdict(None) is None


def test_a_confident_stored_upper_colour_decides_when_there_is_no_caption():
    assert verdict(None, {"upper_color": "brown"}) is True
    assert verdict(None, {"upper_color": "black"}) is False
    assert verdict(None, {"upper_color": "black", "is_ir": True}) is None
