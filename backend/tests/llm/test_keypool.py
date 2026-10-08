import pytest

from evora.llm.keypool import KeyPool, mask_key, parse_duration


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.mark.parametrize(
    "text,expected",
    [
        ("7.66s", 7.66),
        ("2m59.56s", 179.56),
        ("120ms", 0.12),
        ("1h2m3s", 3723.0),
        ("30", 30.0),
        (None, None),
        ("", None),
        ("soon", None),
    ],
)
def test_parse_duration(text, expected):
    got = parse_duration(text)
    assert got == pytest.approx(expected) if expected is not None else got is None


def test_mask_key_never_shows_prefix():
    assert mask_key("gsk_abcdef1234") == "...1234"
    assert "gsk_" not in mask_key("gsk_abcdef1234")


def test_picks_key_with_most_headroom():
    clock = Clock()
    pool = KeyPool(["a", "b", "c"], clock)
    pool.update_from_headers(0, "m", {"x-ratelimit-remaining-tokens": "500", "x-ratelimit-reset-tokens": "30s"})
    pool.update_from_headers(1, "m", {"X-RateLimit-Remaining-Tokens": "6000", "x-ratelimit-reset-tokens": "30s"})
    pool.update_from_headers(2, "m", {"x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "30s"})
    assert pool.pick("m") == 1


def test_need_tokens_skips_depleted_keys():
    clock = Clock()
    pool = KeyPool(["a", "b"], clock)
    pool.update_from_headers(0, "m", {"x-ratelimit-remaining-tokens": "300", "x-ratelimit-reset-tokens": "30s"})
    pool.update_from_headers(1, "m", {"x-ratelimit-remaining-tokens": "200", "x-ratelimit-reset-tokens": "30s"})
    assert pool.pick("m", need_tokens=1000) is None
    clock.t += 31  # window refilled
    assert pool.pick("m", need_tokens=1000) is not None


def test_429_backs_off_only_that_key():
    clock = Clock()
    pool = KeyPool(["a", "b"], clock)
    pool.record_rate_limited(0, "m", retry_after=10)
    assert pool.pick("m") == 1
    clock.t += 11
    assert pool.available(0, "m")


def test_limits_are_per_model():
    clock = Clock()
    pool = KeyPool(["a"], clock)
    pool.record_rate_limited(0, "m1", retry_after=10)
    assert pool.pick("m1") is None
    assert pool.pick("m2") == 0


def test_circuit_opens_after_three_failures_and_closes_after_60s():
    clock = Clock()
    pool = KeyPool(["a"], clock)
    for _ in range(2):
        pool.record_failure(0, "m")
    assert pool.available(0, "m")
    pool.record_failure(0, "m")
    assert not pool.available(0, "m")
    assert pool.all_unavailable("m")
    clock.t += 61
    assert pool.available(0, "m")


def test_success_resets_failure_count():
    pool = KeyPool(["a"], Clock())
    pool.record_failure(0, "m")
    pool.record_failure(0, "m")
    pool.record_success(0, "m")
    pool.record_failure(0, "m")
    assert pool.available(0, "m")


def test_from_env_value_ignores_blanks():
    pool = KeyPool.from_env_value(" k1 , ,k2,")
    assert len(pool) == 2
    assert KeyPool.from_env_value(None).pick("m") is None


def test_from_env_accepts_comma_list_and_numbered_variables():
    env = {"GROQ_KEYS": "k1,k2", "GROQ_KEY-2": "k4", "GROQ_KEY-1": "k3", "GROQ_KEY_10": "k5", "OTHER": "x",
           "GROQ_KEY-3": " ", "GROQ_KEY-9": "k1"}
    pool = KeyPool.from_env(env)
    assert [pool.key(i) for i in range(len(pool))] == ["k1", "k2", "k3", "k4", "k5"]
    assert len(KeyPool.from_env({})) == 0


def test_pick_sticks_to_the_last_good_key_until_it_runs_low_or_is_limited():
    clock = Clock()
    pool = KeyPool(["a", "b"], clock)
    pool.record_success(1, "m")
    assert pool.pick("m") == 1  # not the first key: the last one that worked
    pool.update_from_headers(1, "m", {"x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "30s"})
    assert pool.pick("m", need_tokens=2000) == 0  # no longer has room
    pool.record_success(0, "m")
    pool.record_rate_limited(0, "m", retry_after=10)
    assert pool.pick("m") == 1  # limited keys are skipped
    assert pool.pick("other-model") == 0  # stickiness is per model
