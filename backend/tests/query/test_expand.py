import pytest
from contracts.models import Target

from evora.llm.schemas import LLMError
from evora.query.expand import MAX_VARIANTS, Expansion, expand, keeps_attributes, lexicon_variants


def target(noun="car", embed="a photo of a red car", attrs=("red",)):
    return Target(noun=noun, cls=[noun], attributes=list(attrs), embed_text=embed)


class FakeGateway:
    def __init__(self, texts=None, error=None):
        self.texts, self.error, self.calls = texts or [], error, []

    async def chat_json(self, task, messages, schema):
        self.calls.append((task, messages))
        if self.error:
            raise self.error
        return Expansion(texts=self.texts)


@pytest.mark.asyncio
async def test_off_returns_only_the_original():
    assert await expand(target(), "off") == ["a photo of a red car"]
    assert await expand(Target(noun="x", embed_text=""), "lexicon") == []


@pytest.mark.asyncio
async def test_lexicon_widens_the_noun_and_keeps_the_colour():
    out = await expand(target(), "lexicon")
    assert out == ["a photo of a red car", "a photo of a red sedan", "a photo of a red hatchback",
                   "a photo of a red suv"]
    assert len(out) <= MAX_VARIANTS


@pytest.mark.asyncio
async def test_lexicon_without_an_entry_returns_the_original():
    t = target("umbrella", "a photo of a person with an umbrella", ())
    assert await expand(t, "lexicon") == ["a photo of a person with an umbrella"]


def test_lexicon_replaces_only_the_noun_word():
    t = target("car", "a photo of a red car near a carpet", ("red",))
    assert lexicon_variants(t)[0] == "a photo of a red sedan near a carpet"  # "carpet" untouched


@pytest.mark.asyncio
async def test_llm_variants_are_cleaned_and_must_keep_attributes():
    gw = FakeGateway(["a red sedan", "A PHOTO OF A BLUE CAR", "a photo of a red hatchback.", "",
                      "a photo of a red car", "x" * 200, "a photo of a red taxi"])
    out = await expand(target(), "llm", gw)
    assert out == ["a photo of a red car", "a photo of a red sedan", "a photo of a red hatchback",
                   "a photo of a red taxi"]
    assert gw.calls[0][0] == "expand" and "a photo of a red car" in gw.calls[0][1][1]["content"]


@pytest.mark.asyncio
async def test_llm_failure_or_useless_answer_falls_back_to_the_lexicon():
    lexicon = await expand(target(), "lexicon")
    assert await expand(target(), "llm", FakeGateway(error=LLMError("down"))) == lexicon
    assert await expand(target(), "llm", FakeGateway(["a photo of a blue car"])) == lexicon  # drops "red"
    assert await expand(target(), "llm", None) == lexicon


def test_attribute_check_handles_underscores_and_word_boundaries():
    t = target("person", "a photo of a person carrying a large bag", ("large_bag",))
    assert keeps_attributes("a photo of a pedestrian carrying a large bag", t)
    assert not keeps_attributes("a photo of a pedestrian", t)
    assert not keeps_attributes("a photo of a scarred car", target(attrs=("red",)))
