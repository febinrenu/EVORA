import numpy as np
import pytest

pytest.importorskip("ultralytics")

from evora.perception import openvocab  # noqa: E402


def test_missing_assets_raise_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("EVORA_MODELS_DIR", str(tmp_path))
    with pytest.raises(openvocab.OpenVocabUnavailable, match="models_download"):
        openvocab.OpenVocabDetector()


def _ready() -> bool:
    models = openvocab._models_dir()
    try:
        import clip  # noqa: F401
    except ImportError:
        return False
    return (models / "yoloe-26n-seg.pt").is_file() and (models / openvocab.TEXT_ENCODER).is_file()


@pytest.mark.skipif(not _ready(), reason="YOLOE checkpoint, text encoder or clip package not installed")
def test_text_prompts_produce_normalised_labelled_boxes():
    det = openvocab.OpenVocabDetector(device="cpu")
    out = det.detect(np.zeros((240, 320, 3), dtype=np.uint8), ["red car", "person"], conf=0.01)
    assert isinstance(out, list)
    assert all(b.label in ("red car", "person") and all(0.0 <= v <= 1.0 for v in b.xyxy) for b in out)
    assert det.detect(np.zeros((8, 8, 3), dtype=np.uint8), []) == []
