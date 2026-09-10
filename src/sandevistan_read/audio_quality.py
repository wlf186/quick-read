from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


_ZH_DIGITS = "零一二三四五六七八九"


def _integer_to_spoken_zh(value: str) -> str:
    number = int(value or "0")
    if number == 0:
        return _ZH_DIGITS[0]
    if number >= 10_000:
        return "".join(_ZH_DIGITS[int(digit)] for digit in str(number))
    units = ((1000, "千"), (100, "百"), (10, "十"), (1, ""))
    output: list[str] = []
    pending_zero = False
    for divisor, suffix in units:
        digit, number = divmod(number, divisor)
        if digit:
            if pending_zero and output:
                output.append("零")
            if not (divisor == 10 and digit == 1 and not output):
                output.append(_ZH_DIGITS[digit])
            output.append(suffix)
            pending_zero = False
        elif output and number:
            pending_zero = True
    return "".join(output)


def _normalize_spoken_numbers_zh(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        integer, dot, fraction = match.group(0).partition(".")
        spoken = _integer_to_spoken_zh(integer)
        if dot:
            spoken += "点" + "".join(_ZH_DIGITS[int(digit)] for digit in fraction)
        return spoken

    return re.sub(r"\d+(?:\.\d+)?", replace, text)


def _tokens(text: str, language: str) -> list[str]:
    if language == "en":
        return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())
    text = _normalize_spoken_numbers_zh(text)
    return re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", text.lower())


def _edit_distance(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row_index, left_value in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


AUDIO_ASSESSMENT_VERSION = 2


def _speaker_assessment(turns: list[dict[str, Any]], segments: list[dict[str, Any]]) -> dict[str, Any]:
    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    observations: list[dict[str, Any]] = []
    total = evaluated = 0.0
    methods: set[str] = set()
    for segment_index, segment in enumerate(segments):
        words = [word for word in segment.get("words") or [] if isinstance(word, dict)
                 and word.get("start") is not None and word.get("end") is not None]
        use_words = bool(words) and all(word.get("speaker") or word.get("speaker_label") for word in words)
        methods.add("word_speaker" if use_words else "segment_overlap")
        for value in words if use_words else [segment]:
            start, end = float(value.get("start") or 0), float(value.get("end") or 0)
            duration = max(0.0, end - start)
            total += duration
            label = str(value.get("speaker") or value.get("speaker_label") or "")
            if not label or not duration:
                continue
            for index, turn in enumerate(turns):
                overlap = max(0.0, min(end, float(turn.get("end_seconds") or 0))
                              - max(start, float(turn.get("start_seconds") or 0)))
                if overlap:
                    host = str(turn["speaker"])
                    votes[label][host] += overlap
                    evaluated += overlap
                    observations.append({"segment": segment_index, "turn": index, "start": start,
                                         "end": end, "speaker": label, "expected": host, "weight": overlap})
    labels = sorted(votes, key=lambda label: sum(votes[label].values()), reverse=True)[:2]
    mapping: dict[str, str] = {}
    matched = 0.0
    if len(labels) == 1:
        host = max(votes[labels[0]], key=votes[labels[0]].get)
        mapping, matched = {labels[0]: host}, votes[labels[0]][host]
    elif len(labels) == 2:
        mapping, matched = max([
            ({labels[0]: "HOST_A", labels[1]: "HOST_B"}, votes[labels[0]]["HOST_A"] + votes[labels[1]]["HOST_B"]),
            ({labels[0]: "HOST_B", labels[1]: "HOST_A"}, votes[labels[0]]["HOST_B"] + votes[labels[1]]["HOST_A"]),
        ], key=lambda item: item[1])
    issues: dict[int, dict[str, Any]] = {}
    for item in observations:
        if mapping.get(item["speaker"]) != item["expected"]:
            issue = issues.setdefault(item["turn"], {"turn": item["turn"], "start_seconds": item["start"],
                                                    "end_seconds": item["end"], "mismatched_seconds": 0.0})
            issue["end_seconds"] = max(issue["end_seconds"], item["end"])
            issue["mismatched_seconds"] += item["weight"]
    for issue in issues.values():
        issue["mismatched_seconds"] = round(issue["mismatched_seconds"], 3)
    return {"speaker_alignment": round(matched / evaluated, 4) if evaluated else 0.0,
            "speaker_mapping": mapping, "speaker_method": next(iter(methods)) if len(methods) == 1 else "mixed" if methods else "unavailable",
            "speaker_coverage": round(min(1.0, evaluated / total), 4) if total else 0.0,
            "speaker_evaluated_seconds": round(evaluated, 3), "speaker_issues": list(issues.values())}


def _segment_host_alignment(turns: list[dict[str, Any]], segments: list[dict[str, Any]]) -> tuple[float, dict[str, str]]:
    report = _speaker_assessment(turns, segments)
    return report["speaker_alignment"], report["speaker_mapping"]


def assess_transcription(turns: list[dict[str, Any]], result: dict[str, Any], language: str) -> dict[str, Any]:
    segments = [item for item in result.get("segments") or [] if isinstance(item, dict)]
    by_turn: dict[int, list[str]] = defaultdict(list)
    for segment in segments:
        timed_words = [
            word for word in segment.get("words") or []
            if isinstance(word, dict) and word.get("start") is not None and word.get("end") is not None
        ]
        values = timed_words or [segment]
        for value in values:
            start, end = float(value.get("start") or 0), float(value.get("end") or 0)
            midpoint = (start + end) / 2
            index = next(
                (
                    position for position, turn in enumerate(turns)
                    if float(turn.get("start_seconds") or 0) <= midpoint <= float(turn.get("end_seconds") or 0)
                ),
                None,
            )
            if index is not None:
                by_turn[index].append(str(value.get("text") or ""))
    total_edits = total_reference = 0
    turn_errors: list[int] = []
    threshold = 0.12 if language == "en" else 0.08
    for index, turn in enumerate(turns):
        expected = _tokens(str(turn.get("text") or ""), language)
        observed = _tokens(" ".join(by_turn.get(index, [])), language)
        if not expected:
            continue
        edits = _edit_distance(expected, observed)
        rate = edits / len(expected)
        total_edits += edits
        total_reference += len(expected)
        if len(expected) >= 4 and rate > threshold * 3:
            turn_errors.append(index)
    error_rate = total_edits / max(1, total_reference)
    speaker_report = _speaker_assessment(turns, segments)
    speaker_alignment = speaker_report["speaker_alignment"]
    gaps: list[tuple[float, float]] = []
    ordered = sorted(segments, key=lambda item: float(item.get("start") or 0))
    for left, right in zip(ordered, ordered[1:]):
        gap = float(right.get("start") or 0) - float(left.get("end") or 0)
        if gap > 0:
            gaps.append(((float(left.get("end") or 0) + float(right.get("start") or 0)) / 2, gap))
    silence_outliers = sum(gap > 1.5 for _, gap in gaps)
    # 把超长静音按中点归属到所在轮次（轮首静音属于后一轮），让渲染层只重合成这些轮
    silence_outlier_turns = sorted({
        index
        for midpoint, gap in gaps
        if gap > 1.5
        for index, turn in enumerate(turns)
        if float(turn.get("start_seconds") or 0) <= midpoint <= float(turn.get("end_seconds") or 0)
    })
    passed = bool(
        segments
        and error_rate <= threshold
        and not turn_errors
        and speaker_alignment >= 0.95
        and speaker_report["speaker_coverage"] >= 0.95
        and silence_outliers == 0
    )
    return {
        "passed": passed,
        "metric": "wer" if language == "en" else "cer",
        "error_rate": round(error_rate, 4),
        "threshold": threshold,
        "assessment_version": AUDIO_ASSESSMENT_VERSION,
        **speaker_report,
        "turn_errors": turn_errors,
        "segment_count": len(segments),
        "silence_outliers": silence_outliers,
        "silence_outlier_turns": silence_outlier_turns,
        "max_internal_silence": round(max((gap for _, gap in gaps), default=0.0), 3),
        "asr_model": result.get("model"),
        "compute_device": result.get("compute_device"),
        "device_fallback": bool(result.get("fallback_used")),
        "fallback_reason": result.get("fallback_reason"),
    }


def repair_turn_indexes(report: dict[str, Any]) -> list[int]:
    """需要重合成的轮次：转写错轮与超长静音归属轮的并集，生产与评测路径共用。"""
    return sorted(set(report.get("turn_errors") or []) | set(report.get("silence_outlier_turns") or []))
