from sandevistan_read.audio_quality import assess_transcription


def _turns() -> list[dict]:
    return [
        {"speaker": "HOST_A", "text": "The loop changes the program.", "start_seconds": 0.0, "end_seconds": 2.0},
        {"speaker": "HOST_B", "text": "Why does that matter?", "start_seconds": 2.2, "end_seconds": 4.0},
    ]


def test_audio_quality_accepts_accurate_two_speaker_transcript() -> None:
    result = assess_transcription(
        _turns(),
        {"model": "qwen3-asr-0.6b", "compute_device": "gpu", "segments": [
            {"start": 0.0, "end": 1.9, "speaker": "SPEAKER_00", "text": "The loop changes the program."},
            {"start": 2.2, "end": 3.9, "speaker": "SPEAKER_01", "text": "Why does that matter?"},
        ]},
        "en",
    )
    assert result["passed"] is True
    assert result["error_rate"] == 0
    assert result["speaker_alignment"] == 1


def test_audio_quality_identifies_bad_turn_and_speaker_collapse() -> None:
    result = assess_transcription(
        _turns(),
        {"segments": [
            {"start": 0.0, "end": 1.9, "speaker": "SPEAKER_00", "text": "unrelated words entirely"},
            {"start": 2.2, "end": 3.9, "speaker": "SPEAKER_00", "text": "Why does that matter?"},
        ]},
        "en",
    )
    assert result["passed"] is False
    assert result["turn_errors"] == [0]
    assert result["speaker_alignment"] < 0.95


def test_audio_quality_uses_word_timestamps_when_segment_crosses_turn_boundary() -> None:
    turns = [
        {"speaker": "HOST_A", "text": "First answer", "start_seconds": 0.0, "end_seconds": 1.0},
        {"speaker": "HOST_B", "text": "Second answer", "start_seconds": 1.0, "end_seconds": 2.0},
    ]
    result = assess_transcription(
        turns,
        {"segments": [{
            "start": 0.0,
            "end": 2.0,
            "speaker": "SPEAKER_00",
            "text": "First answer Second answer",
            "words": [
                {"text": "First", "start": 0.1, "end": 0.3},
                {"text": "answer", "start": 0.4, "end": 0.8},
                {"text": "Second", "start": 1.1, "end": 1.4},
                {"text": "answer", "start": 1.5, "end": 1.8},
            ],
        }]},
        "en",
    )
    assert result["error_rate"] == 0
    assert result["turn_errors"] == []


def test_audio_quality_treats_chinese_spoken_numbers_as_equivalent() -> None:
    turns = [{
        "speaker": "HOST_A",
        "text": "q 从 0.10 到 0.45，z 从 5、11、24 到 340。",
        "start_seconds": 0.0,
        "end_seconds": 5.0,
    }]
    result = assess_transcription(
        turns,
        {"segments": [{
            "start": 0.0,
            "end": 5.0,
            "speaker": "SPEAKER_00",
            "text": "Q从零点一零到零点四五，Z从五、十一、二十四到三百四十。",
        }]},
        "zh-CN",
    )
    assert result["error_rate"] == 0
    assert result["turn_errors"] == []
