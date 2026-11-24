import numpy as np

from clipper.analysis.audio import AudioFeatures
from clipper.analysis.highlights import refine_with_transcript, score_curve, select_highlights
from clipper.config import HighlightsConfig
from clipper.models import Highlight, Segment, Transcript

HOP = 0.5


def make_features(duration=600.0, bursts=(), seed=0):
    n = int(duration / HOP)
    rng = np.random.default_rng(seed)
    loud = -30 + rng.normal(0, 1.0, n)
    vocal = -35 + rng.normal(0, 1.0, n)
    for centre, amp in bursts:
        a, b = int((centre - 3) / HOP), int((centre + 3) / HOP)
        loud[a:b] += amp
        vocal[a:b] += amp
    return AudioFeatures(HOP, duration, loud.astype(np.float32), vocal.astype(np.float32))


def run(feats, cfg, duration=None, chat=None):
    score, comps = score_curve(feats, cfg, chat)
    return select_highlights(feats, score, comps, cfg, duration or feats.duration), score, comps


def check_invariants(hl, cfg, duration):
    for h in hl:
        assert 0 <= h.start < h.peak < h.end <= duration + 1e-6
        assert cfg.min_duration - 1e-6 <= h.duration <= cfg.max_duration + 1e-6
    s = sorted(hl, key=lambda h: h.start)
    for a, b in zip(s, s[1:]):
        assert a.end <= b.start + 1e-9
    assert [h.rank for h in hl] == list(range(1, len(hl) + 1))
    assert [h.score for h in hl] == sorted((h.score for h in hl), reverse=True)


def test_picks_bursts_in_rank_order():
    feats = make_features(bursts=[(120, 20), (300, 15), (480, 11)])
    cfg = HighlightsConfig(num_clips=3, skip_start=60, skip_end=30, min_gap=60)
    hl, _, comps = run(feats, cfg)
    assert len(hl) == 3
    check_invariants(hl, cfg, 600)
    assert [round(h.peak / 60) for h in hl] == [2, 5, 8] or sorted(round(h.peak / 60) for h in hl) == [2, 5, 8]
    by_rank = [min((120, 300, 480), key=lambda c: abs(c - h.peak)) for h in hl]
    assert by_rank == [120, 300, 480]
    for h, c in zip(hl, by_rank):
        assert abs(h.peak - c) <= 8
        assert set(h.reasons) == {"loudness", "vocal", "excitement", "chat"}
    assert set(comps) == {"loudness", "vocal", "excitement", "chat"}


def test_many_clips_never_overlap_and_durations_hold():
    feats = make_features(duration=900.0, bursts=[(100 + 80 * i, 8 + i) for i in range(9)], seed=3)
    cfg = HighlightsConfig(num_clips=12, skip_start=30, skip_end=30)
    hl, _, _ = run(feats, cfg)
    assert len(hl) >= 5
    check_invariants(hl, cfg, 900)


def test_deterministic():
    feats = make_features(bursts=[(150, 12), (400, 14)])
    cfg = HighlightsConfig(num_clips=4)
    a, _, _ = run(feats, cfg)
    b, _, _ = run(feats, cfg)
    assert a == b


def test_skip_start_and_skip_end_respected():
    feats = make_features(bursts=[(30, 40), (300, 10), (580, 40)])
    cfg = HighlightsConfig(num_clips=5, skip_start=60, skip_end=30)
    hl, _, _ = run(feats, cfg)
    assert hl
    assert all(60 <= h.peak <= 570 for h in hl)
    assert abs(hl[0].peak - 300) <= 8


def test_silent_audio_returns_nothing():
    n = 1200
    feats = AudioFeatures(HOP, 600.0, np.full(n, -90, np.float32), np.full(n, -90, np.float32))
    hl, score, _ = run(feats, HighlightsConfig())
    assert hl == []
    assert np.all(np.isfinite(score))


def test_flat_audio_does_not_crash():
    feats = AudioFeatures(HOP, 600.0, np.full(1200, -30, np.float32), np.full(1200, -35, np.float32))
    hl, _, _ = run(feats, HighlightsConfig())
    check_invariants(hl, HighlightsConfig(), 600)


def test_short_videos():
    cfg = HighlightsConfig()
    tiny = make_features(duration=3.0)
    assert run(tiny, cfg)[0] == []
    short = make_features(duration=12.0, bursts=[(6, 15)])
    hl, _, _ = run(short, cfg)
    assert len(hl) == 1 and hl[0].start == 0.0 and hl[0].end == 12.0 and hl[0].rank == 1


def test_video_just_over_min_duration():
    cfg = HighlightsConfig(skip_start=60, skip_end=30)
    feats = make_features(duration=45.0, bursts=[(20, 15)])
    hl, _, _ = run(feats, cfg)
    check_invariants(hl, cfg, 45)
    assert len(hl) <= 1


def test_chat_pulls_peak():
    feats = make_features(bursts=[])
    cfg = HighlightsConfig(num_clips=1)
    chat = np.zeros(len(feats.loudness_db), np.float32)
    chat[int(350 / HOP): int(360 / HOP)] = 10.0
    hl, _, comps = run(feats, cfg, chat=chat)
    assert hl and 340 <= hl[0].peak <= 362
    assert hl[0].reasons["chat"] > 3


def seg(a, b, text="x"):
    return Segment(text, a, b)


def test_refine_snaps_to_sentence_edges():
    cfg = HighlightsConfig()
    h = Highlight(100.0, 140.0, 120.0, 5.0, 1)
    t = Transcript([seg(98.5, 102.0), seg(104, 120), seg(136.0, 141.0)])
    r = refine_with_transcript(h, t, cfg, 1000.0)
    assert abs(r.start - (98.5 - 0.15)) < 1e-9
    assert abs(r.end - (141.0 + 0.35)) < 1e-9
    assert r.peak == h.peak and r.rank == 1


def test_refine_ignores_far_boundaries_and_empty():
    cfg = HighlightsConfig()
    h = Highlight(100.0, 140.0, 120.0, 5.0, 1)
    far = Transcript([seg(90.0, 105.0), seg(120, 150.0)])
    r = refine_with_transcript(h, far, cfg, 1000.0)
    assert (r.start, r.end) == (100.0, 140.0)
    assert refine_with_transcript(h, Transcript(), cfg, 1000.0) is h
    off = HighlightsConfig(snap_to_sentences=False)
    assert refine_with_transcript(h, Transcript([seg(99, 101)]), off, 1000.0) is h


def test_refine_clamps_and_respects_max_duration():
    cfg = HighlightsConfig()
    h = Highlight(1.0, 40.0, 20.0, 5.0, 1)
    r = refine_with_transcript(h, Transcript([seg(0.0, 2.0), seg(30, 39.9)]), cfg, 40.0)
    assert r.start == 0.0 and r.end <= 40.0
    h2 = Highlight(100.0, 158.0, 120.0, 5.0, 1)
    r2 = refine_with_transcript(h2, Transcript([seg(96.0, 101.0), seg(150, 158.5)]), cfg, 1000.0)
    assert r2.duration <= cfg.max_duration + 1e-9
