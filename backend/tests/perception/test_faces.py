import cv2
import numpy as np
import pytest

pytest.importorskip("cv2")

from evora.perception import faces  # noqa: E402

needs_model = pytest.mark.skipif(not faces.model_path().is_file(), reason="YuNet model not downloaded")


def _sharp(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


@needs_model
def test_a_face_is_blurred_and_the_rest_is_untouched():
    data = pytest.importorskip("skimage.data")
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    boxes = faces.detect_faces(img)
    assert boxes, "the sample portrait should contain a face"
    out = faces.blur_frame(img)
    x, y, w, h = boxes[0]
    assert _sharp(out[y : y + h, x : x + w]) < 0.1 * _sharp(img[y : y + h, x : x + w])
    assert np.array_equal(out[-40:, -40:], img[-40:, -40:])


@needs_model
def test_bytes_in_jpeg_bytes_out():
    data = pytest.importorskip("skimage.data")
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img)
    out = faces.blur_faces(buf.tobytes())
    assert out[:2] == b"\xff\xd8"
    assert cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR).shape == img.shape


@needs_model
def test_image_without_faces_keeps_its_pixels():
    img = np.full((120, 160, 3), 90, dtype=np.uint8)
    assert faces.detect_faces(img) == []
    assert np.array_equal(faces.blur_frame(img), img)


def test_missing_model_raises_instead_of_returning_unblurred(tmp_path, monkeypatch):
    monkeypatch.setenv("EVORA_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(faces, "_local", __import__("threading").local())
    with pytest.raises(faces.FaceBlurUnavailable):
        faces.blur_frame(np.zeros((32, 32, 3), dtype=np.uint8))


def test_garbage_bytes_are_rejected():
    with pytest.raises(ValueError):
        faces.blur_faces(b"not an image")


def test_threads_blur_at_the_same_time_with_their_own_detector():
    import threading

    import numpy as np

    from evora.perception import faces

    if not faces.model_path().is_file():
        import pytest

        pytest.skip("YuNet model not downloaded")
    seen, errors = [], []

    def work(size):
        try:
            frame = np.zeros((size, size * 2, 3), dtype=np.uint8)
            for _ in range(3):
                assert faces.blur_frame(frame).shape == frame.shape
            seen.append(id(faces._local.detector))
        except Exception as exc:  # noqa: BLE001 - the test reports any failure below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(s,)) for s in (240, 360, 480, 720)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(set(seen)) == 4  # four threads, four detectors, different frame sizes at once
