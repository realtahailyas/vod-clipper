import shutil
import subprocess

import numpy as np
import pytest

from clipper.analysis import audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg not available")


def _make_video(path, with_audio=True):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=20"]
    if with_audio:
        fc = ("sine=frequency=440:duration=10,volume=0.02[a];"
              "sine=frequency=440:duration=4,volume=0.8[b];"
              "sine=frequency=440:duration=6,volume=0.02[c];"
              "[a][b][c]concat=n=3:v=0:a=1[out]")
        cmd += ["-filter_complex", fc, "-map", "0:v", "-map", "[out]", "-c:a", "aac"]
    cmd += ["-c:v", "mpeg4", "-shortest", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


def test_features_burst_and_cache(tmp_path, monkeypatch):
    v = tmp_path / "t.mp4"
    _make_video(v)
    cache = tmp_path / "cache"
    seen = []
    f = audio.compute_features(v, hop=0.5, cache_dir=cache, progress=seen.append)
    assert 18.5 <= f.duration <= 20.5
    assert f.loudness_db.dtype == np.float32 and len(f.loudness_db) == len(f.vocal_db)
    assert seen and seen[-1] == 1.0
    t = f.times()
    burst = (t > 10.7) & (t < 13.3)
    quiet = (t < 8.5) | (t > 15.5)
    assert f.loudness_db[burst].mean() > f.loudness_db[quiet].mean() + 15
    assert f.vocal_db[burst].mean() > f.vocal_db[quiet].mean() + 15
    assert f.loudness_db.min() >= -90 and f.index(-5) == 0 and f.index(1e9) == len(t) - 1

    files = list(cache.glob("*.npz"))
    assert len(files) == 1
    mtime = files[0].stat().st_mtime_ns

    def boom(*a, **k):
        raise AssertionError("ffmpeg must not run when cache exists")
    monkeypatch.setattr(audio.subprocess, "Popen", boom)
    g = audio.compute_features(v, hop=0.5, cache_dir=cache)
    np.testing.assert_array_equal(g.loudness_db, f.loudness_db)
    assert files[0].stat().st_mtime_ns == mtime


def test_no_audio_stream_raises(tmp_path):
    v = tmp_path / "silent.mp4"
    _make_video(v, with_audio=False)
    with pytest.raises(ValueError, match="no audio"):
        audio.compute_features(v, cache_dir=tmp_path / "c")
