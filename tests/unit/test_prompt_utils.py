"""Unit tests for backend/prompt_utils.py"""
from backend.prompt_utils import expand_prompt


def test_single_vector_expansion():
    assert expand_prompt("a photo of a <sks>", {"sks": 1}) == "a photo of a sks.0"


def test_multi_vector_expansion():
    assert expand_prompt("a photo of a <sks>", {"sks": 3}) == "a photo of a sks.0 sks.1 sks.2"


def test_unresolved_token_left_as_is():
    assert expand_prompt("a photo of a <unknown>", {"sks": 1}) == "a photo of a <unknown>"


def test_multiple_tokens_in_one_string():
    result = expand_prompt("a <sks> and a <foo>", {"sks": 2, "foo": 1})
    assert result == "a sks.0 sks.1 and a foo.0"


def test_class_prompt_suffix_preserved():
    result = expand_prompt("a photo of a <sks> dog", {"sks": 2})
    assert result == "a photo of a sks.0 sks.1 dog"
