import pytest
from eval import calibrate_alias as cal

from evora.memory.embedder import HashingEmbedder


def test_recommend_puts_tau_hi_above_every_different_pair_and_tau_lo_below_the_paraphrases():
    same = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.88, 0.92]
    different = [0.30, 0.50, 0.70, 0.78, 0.40]
    rec = cal.recommend(same, different, min_paraphrase_recall=1.0)
    assert rec.false_accepts == 0 and rec.tau_hi > max(different)
    assert rec.paraphrases_below_lo == 0 and rec.tau_lo < min(same)
    assert rec.tau_lo < rec.tau_hi and 0 < rec.grey_share <= 1


def test_recommend_can_trade_a_few_misses_for_a_tighter_grey_band():
    same = [0.9] * 19 + [0.1]  # one hopeless paraphrase
    rec = cal.recommend(same, [0.2, 0.3], min_paraphrase_recall=0.95)
    assert rec.paraphrases_below_lo == 1 and rec.tau_lo < rec.tau_hi  # the one hopeless paraphrase is given up on


def test_recommend_keeps_the_band_ordered_when_the_distributions_overlap():
    rec = cal.recommend([0.5, 0.6, 0.7], [0.9, 0.95])  # different pairs score higher than the paraphrases
    assert rec.tau_lo < rec.tau_hi
    with pytest.raises(ValueError):
        cal.recommend([], [0.1])


def test_pair_set_is_balanced_unique_and_covers_every_kind():
    keys = [(a, b) for a, b, _, _ in cal.PAIRS]
    assert len(keys) == len(set(keys))
    for kind in ("place", "object", "time"):
        flags = [same for *_, k, same in [(a, b, k, s) for a, b, k, s in cal.PAIRS] if k == kind]
        assert flags.count(True) >= 5 and flags.count(False) >= 5, kind
    assert sum(1 for *_, s in cal.PAIRS if s) / len(cal.PAIRS) > 0.4


def test_scoring_and_report_with_the_offline_embedder():
    scored = cal.score_pairs(HashingEmbedder().embed)
    assert len(scored) == len(cal.PAIRS) and all(-1.0 <= s <= 1.0001 for *_, s in scored)
    identical = next(s for a, b, _, _, s in scored if a == b)
    assert identical == pytest.approx(1.0, abs=1e-4)
    text = cal.report(scored, 0.85, 0.6)
    assert "recommended: tau_hi" in text and "current config: tau_hi 0.85" in text
    assert "hardest different pairs" in text
