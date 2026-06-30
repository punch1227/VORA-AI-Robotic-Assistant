"""
Validation tests for the Thai command router.

Run with:
    cd VORA && python -m pytest Gateway/tests/test_command_router.py -v

Covers:
  - the six priority control intents (exact form)
  - noisy / typo-like variants (repeated letters, spaces, wake-word bleed)
  - existing motion / find intents still work
  - unknown command shape → parse_control_intent returns None
"""
import re
import pytest

from gateway.intent_parser import (
    normalize_thai_command,
    parse_control_intent,
    parse_intent,
    parse_find_intent,
    looks_like_command,
)


# ------- (1) Exact control commands -------
@pytest.mark.parametrize("text,expected", [
    ("ยกเลิกคำสั่ง",      "cancel"),
    ("พูดอีกครั้ง",        "repeat_last_response"),
    ("รายงานสถานะ",        "report_status"),
    ("ตอนนี้อยู่ที่ไหน",   "where_am_i"),
    ("กลับจุดเริ่มต้น",   "return_home"),
    ("หยุด",              "stop"),
])
def test_control_intents_exact(text, expected):
    norm = normalize_thai_command(text)
    res = parse_control_intent(norm)
    assert res is not None, f"expected {expected} for '{text}' (norm='{norm}')"
    assert res["intent"] == expected


# ------- (2) Noisy / typo-like variants -------
@pytest.mark.parametrize("text,expected", [
    # extra spaces
    ("ยก เลิก คำสั่ง ครับ",      "cancel"),
    # duplicated letters (STT stutter)
    ("หยุุุด",                    "stop"),
    ("พููดอีกครั้งงง",            "repeat_last_response"),
    # wake-word bleed
    ("โวร่า รายงานสถานะ",          "report_status"),
    ("VORA ตอนนี้อยู่ที่ไหน",     "where_am_i"),
    # near-miss phrasings
    ("กลับบ้าน",                   "return_home"),
    ("กลับแท่นชาร์จ",              "return_home"),
    ("ทวนคำพูด",                   "repeat_last_response"),
    ("เช็คสถานะ",                  "report_status"),
    ("อยู่ตรงไหน",                 "where_am_i"),
    ("ยกเลิกครับ",                 "cancel"),
    # trailing particles
    ("หยุดนะครับ",                 "stop"),
])
def test_control_intents_noisy(text, expected):
    norm = normalize_thai_command(text)
    res = parse_control_intent(norm)
    assert res is not None, f"expected {expected} for '{text}' (norm='{norm}')"
    assert res["intent"] == expected


# ------- (3) Existing motion/find still works on normalized text -------
def test_existing_motion_still_works():
    for t in ["เดินหน้า", "ถอยหลัง 1 เมตร", "เลี้ยวซ้าย"]:
        assert parse_intent(t) is not None, f"motion parser broke on {t!r}"


def test_existing_find_still_works():
    assert parse_find_intent("หากุญแจ") == "กุญแจ"
    assert parse_find_intent("ช่วยหาปากกาให้หน่อย") is not None


# ------- (4) Unknown command shapes → no control intent -------
@pytest.mark.parametrize("text", [
    "2 + 2 เท่ากับเท่าไหร่",
    "Gemma คืออะไร",
    "คุณคือใคร",
    "เล่าเรื่องตลกหน่อย",
    "อธิบายระบบของคุณ",
])
def test_not_a_control_intent(text):
    norm = normalize_thai_command(text)
    assert parse_control_intent(norm) is None, \
        f"'{text}' wrongly matched control intent (norm='{norm}')"


# ------- (5) command-like heuristic -------
def test_looks_like_command():
    assert looks_like_command("เดินหน้า")
    assert looks_like_command("หากุญแจ")
    assert not looks_like_command("2 + 2 เท่ากับเท่าไหร่")


# ------- (6) Task-oriented mode default & out-of-scope constant -------
def test_task_oriented_mode_defaults_on():
    from app.core import vora_memory
    assert vora_memory.TASK_ORIENTED_MODE is True
    assert "ขออภัย" in vora_memory.OUT_OF_SCOPE_RESPONSE


# ------- (7) Identity prompt no longer leaks architecture -------
def test_identity_prompt_no_leak():
    from app.core import vora_memory
    blob = vora_memory.VORA_IDENTITY + vora_memory.CHAT_SYSTEM_PROMPT
    for forbidden in ["Gemma", "Qwen", "A6000", "KMUTNB", "Ollama", "system prompt"]:
        assert forbidden.lower() not in blob.lower(), \
            f"identity leaks '{forbidden}'"
