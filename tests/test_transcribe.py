import sys
import types

import pytest

from clipper import transcribe
from clipper.config import TranscriptionConfig
from clipper.transcribe import Transcriber, resolve_cpu_threads


def test_explicit_threads():
    assert resolve_cpu_threads(3) == 3
    assert resolve_cpu_threads(32) == 32


@pytest.mark.parametrize("cores,expected", [(1, 1), (2, 1), (4, 2), (8, 4), (12, 6), (64, 6), (None, 2)])
def test_auto_threads(monkeypatch, cores, expected):
    monkeypatch.setattr(transcribe.os, "cpu_count", lambda: cores)
    assert resolve_cpu_threads(0) == expected


def test_whisper_model_gets_threads(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, *a, **kw):
            calls.append(kw)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel))
    t = Transcriber(TranscriptionConfig(device="cpu", cpu_threads=3))
    t._load()
    assert calls[0]["cpu_threads"] == 3 and calls[0]["num_workers"] == 1
