import importlib.util
import sys
from pathlib import Path

import httpx

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
spec = importlib.util.spec_from_file_location("check_local_models", SCRIPTS / "check_local_models.py")
chk = importlib.util.module_from_spec(spec)
sys.modules["check_local_models"] = chk
spec.loader.exec_module(chk)


def test_the_wanted_model_wins_then_an_installed_fallback_then_nothing():
    assert chk.pick_model(["qwen3.5:4b", "qwen3-vl:4b-instruct", "qwen3-vl:4b"]) == "qwen3-vl:4b-instruct"
    assert chk.pick_model(["qwen2.5:0.5b", "qwen3.5:4b"]) == "qwen3.5:4b"
    assert chk.pick_model(["qwen3-vl:4b"]) == "qwen3-vl:4b"
    assert chk.pick_model(["qwen2.5:0.5b"]) is None and chk.pick_model([]) is None


def test_gpu_share_is_how_much_of_the_model_sits_in_gpu_memory():
    assert chk.gpu_share({"size": 100, "size_vram": 100}) == 1.0
    assert chk.gpu_share({"size": 100, "size_vram": 40}) == 0.4
    assert chk.gpu_share({"size": 0, "size_vram": 5}) == 0.0 and chk.gpu_share({}) == 0.0


def test_numbers_are_read_from_replies():
    assert chk.parse_number("3") == 3 and chk.parse_number("There are 3 red squares.") == 3
    assert chk.parse_number("three") is None and chk.parse_number("") is None


def test_the_test_picture_really_has_three_red_squares_and_two_blue_ones():
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(chk.test_picture(), np.uint8), cv2.IMREAD_COLOR)
    red = ((image[:, :, 2] > 180) & (image[:, :, 0] < 60)).astype(np.uint8)
    blue = ((image[:, :, 0] > 180) & (image[:, :, 2] < 60)).astype(np.uint8)
    assert cv2.connectedComponents(red)[0] - 1 == 3 and cv2.connectedComponents(blue)[0] - 1 == 2


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_missing_ollama_or_model_fails_with_the_command_to_fix_it():
    down = client(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    (check,), model = chk.check_ollama("http://x", down)
    assert check.status == "fail" and "ollama serve" in check.fix and model is None
    empty = client(lambda r: httpx.Response(200, json={"models": [{"name": "qwen2.5:0.5b"}]}))
    (check,), model = chk.check_ollama("http://x", empty)
    assert check.status == "fail" and check.fix == f"ollama pull {chk.WANTED}" and model is None
    other = client(lambda r: httpx.Response(200, json={"models": [{"name": "qwen3.5:4b"}]}))
    (check,), model = chk.check_ollama("http://x", other)
    assert check.status == "warn" and model == "qwen3.5:4b"
    ready = client(lambda r: httpx.Response(200, json={"models": [{"name": chk.WANTED}]}))
    (check,), model = chk.check_ollama("http://x", ready)
    assert check.status == "ok" and model == chk.WANTED


def test_the_picture_checks_pass_on_right_answers_and_fail_on_wrong_ones():
    def model_saying(count, blue):
        def handler(request):
            if request.url.path == "/api/ps":
                return httpx.Response(200, json={"models": [{"name": "m", "size": 10, "size_vram": 4}]})
            question = request.content.decode()
            return httpx.Response(200, json={"message": {"content": count if "red squares" in question else blue}})
        return client(handler)
    good = {c.name: c for c in chk.check_picture("http://x", "m", model_saying("3", "Yes."))}
    assert good["Counts what it sees"].status == "ok" and good["Reads colours"].status == "ok"
    assert good["Model runs on the GPU"].status == "warn" and "40%" in good["Model runs on the GPU"].detail
    bad = {c.name: c for c in chk.check_picture("http://x", "m", model_saying("5", "No."))}
    assert bad["Counts what it sees"].status == "fail" and bad["Reads colours"].status == "fail"


def test_a_line_shows_the_fix_only_when_something_is_wrong():
    assert "->" not in chk.Check("a", "ok", "fine", "never shown").line()
    assert "-> do this" in chk.Check("a", "warn", "slow", "do this").line()
