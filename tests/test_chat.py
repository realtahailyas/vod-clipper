import json

import numpy as np
import pytest

from clipper.analysis.chat import chat_activity, find_chat_file, load_chat_timestamps


def test_find_chat_file_priority(tmp_path):
    v = tmp_path / "vod.mp4"
    v.write_bytes(b"")
    assert find_chat_file(v) is None
    (tmp_path / "vod.csv").write_text("time\n1\n")
    assert find_chat_file(v).name == "vod.csv"
    (tmp_path / "vod.chat.json").write_text("[]")
    assert find_chat_file(v).name == "vod.chat.json"


def test_twitchdownloader_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"comments": [{"content_offset_seconds": 12.5}, {"content_offset_seconds": 3},
                                          {"content_offset_seconds": 99.25}]}))
    np.testing.assert_allclose(load_chat_timestamps(p), [3, 12.5, 99.25])


def test_generic_list_with_hms_strings(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"time": "0:01:05", "msg": "a"}, {"time": "00:10", "msg": "b"},
                             {"time": "garbage"}, {"offset": 7.5}]))
    np.testing.assert_allclose(load_chat_timestamps(p), [7.5, 10, 65])


def test_chat_downloader_time_in_seconds_and_absolute_timestamp(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"time_in_seconds": 5.0, "timestamp": 1_700_000_000_000_000},
                             {"time_in_seconds": 9.0, "timestamp": 1_700_000_004_000_000}]))
    np.testing.assert_allclose(load_chat_timestamps(p), [5, 9])
    p2 = tmp_path / "d.json"
    p2.write_text(json.dumps([{"timestamp": 1_700_000_100_000_000}, {"timestamp": 1_700_000_104_000_000}]))
    np.testing.assert_allclose(load_chat_timestamps(p2), [0, 4])


def test_json_lines(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"time_in_seconds": 2}\n\n{"time_in_seconds": 1}\nnot json\n{"time_text": "1:00"}\n')
    np.testing.assert_allclose(load_chat_timestamps(p), [1, 2, 60])


def test_csv(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("user,message,timestamp\nbob,hi,0:00:30\nann,yo,10\nx,bad,zzz\n", encoding="utf-8")
    np.testing.assert_allclose(load_chat_timestamps(p), [10, 30])


def test_zero_rows_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"comments": []}))
    with pytest.raises(ValueError):
        load_chat_timestamps(p)
    p2 = tmp_path / "c.csv"
    p2.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_chat_timestamps(p2)
    p3 = tmp_path / "x.json"
    p3.write_text("%%% nothing")
    with pytest.raises(ValueError):
        load_chat_timestamps(p3)


def test_chat_activity_length_and_delay_shift():
    ts = np.array([100.0 + 0.1 * i for i in range(50)])  # burst 100-105 s
    a0 = chat_activity(ts, duration=200.0, hop=0.5, window=2.0, delay=0.0)
    a4 = chat_activity(ts, duration=200.0, hop=0.5, window=2.0, delay=4.0)
    assert len(a0) == 400 and a0.dtype == np.float32
    c0 = a0.argmax() * 0.5
    c4 = a4.argmax() * 0.5
    assert 100 <= c0 <= 106
    assert abs((c0 - c4) - 4.0) <= 1.0
    assert abs(a0.sum() - a4.sum()) < 1e-3


def test_chat_activity_short_duration():
    a = chat_activity(np.array([0.2, 0.4]), duration=1.2, hop=0.5, window=10.0, delay=0.0)
    assert len(a) == 3
