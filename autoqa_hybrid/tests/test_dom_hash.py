"""Tests for the Rust DOM hashing + mutation delta engine via Python."""
import pytest

from autoqa import native
from autoqa.driver import flatten_dom_tokens, _fnv1a


class TestHashTokens:
    def test_stable_for_equal_streams(self):
        a = ["div", "#btn", "submit"]
        assert native.hash_tokens(a) == native.hash_tokens(list(a))

    def test_changes_on_any_mutation(self):
        base = ["div", "#btn", "submit"]
        assert native.hash_tokens(base) != native.hash_tokens(base + ["input"])
        assert native.hash_tokens(base) != native.hash_tokens(list(reversed(base)))
        # token boundary sensitivity
        assert native.hash_tokens(["ab", "c"]) != native.hash_tokens(["a", "bc"])

    def test_matches_pure_python_fallback(self):
        tokens = ["tag:a|id:1", "tag:BUTTON|id:2", "text:hello world"]
        assert native.hash_tokens(tokens) == _fnv1a(tokens)

    def test_empty_stream(self):
        assert native.hash_tokens([]) == 0xCBF29CE484222325


class TestDomDelta:
    def test_replace_in_middle(self):
        delta = native.dom_delta(["a", "b", "c"], ["a", "x", "c"])
        ops = [(d.op, d.before_index, d.after_index) for d in delta]
        assert ops == [(0, 0, 0), (1, 1, 1), (0, 2, 2)]
        assert delta[1].token == "x"

    def test_insert_and_remove(self):
        delta = native.dom_delta(["a", "b"], ["a", "new", "b"])
        assert any(d.op == 3 and d.after_index == 1 for d in delta)

        delta = native.dom_delta(["a", "gone", "b"], ["a", "b"])
        assert any(d.op == 2 and d.before_index == 1 for d in delta)

    def test_identical_streams_are_all_keeps(self):
        stream = [f"tok{i}" for i in range(100)]
        delta = native.dom_delta(stream, list(stream))
        assert all(d.op == 0 for d in delta)
        assert len(delta) == 100

    def test_large_stream_trims_prefix_and_suffix(self):
        before = [f"tok{i}" for i in range(20_000)]
        after = list(before)
        after[10] = "CHANGED"
        delta = native.dom_delta(before, after)
        # one widget change → one coalesced replace-style mutation
        assert sum(1 for d in delta if d.op != 0) == 1
        assert delta[-1].after_index == 19_999

    def test_op_values_are_documented_ints(self):
        delta = native.dom_delta(["a"], ["b"])
        assert delta[0].op in (1, 2, 3)


class TestDriverIntegration:
    def test_flatten_tokens_are_hashable_and_stable(self):
        elements = [
            {
                "id": 0,
                "tag": "button",
                "text": "Submit",
                "selector": "#submit-btn",
                "name": "",
                "role": "",
                "disabled": False,
                "visible": True,
            }
        ]
        toks = flatten_dom_tokens(elements)
        assert len(toks) == 1
        assert native.hash_tokens(toks) == native.hash_tokens(flatten_dom_tokens(elements))

    def test_mutation_changes_digest(self):
        base = [
            {"id": 0, "tag": "button", "text": "Submit", "selector": "#s", "visible": True},
        ]
        changed = [
            {"id": 0, "tag": "button", "text": "Disabled!", "selector": "#s", "visible": True},
        ]
        assert native.hash_tokens(flatten_dom_tokens(base)) != native.hash_tokens(
            flatten_dom_tokens(changed)
        )
