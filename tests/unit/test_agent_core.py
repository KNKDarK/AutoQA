"""Unit tests for src.agent.core.AutoQAAgent — decisions, retries, parsing.

No real Gemini calls are made: the genai client is stubbed, and the browser is
never started.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

import src.agent.core as core_mod
from tests.integration.helpers import FakeGeminiClient


DECISION = {
    "thought": "Click the submit button",
    "action": "click",
    "selector": "#submit-btn",
    "value": None,
    "bug_description": None,
}


async def _no_sleep(_: float = 0) -> None:
    return None


@pytest.fixture()
def agent(monkeypatch):
    """AutoQAAgent with the genai client stubbed before construction."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-unit-tests")
    monkeypatch.setattr(core_mod.genai, "Client", FakeGeminiClient)
    monkeypatch.setattr(core_mod.asyncio, "sleep", _no_sleep)
    return core_mod.AutoQAAgent(goal="Submit the checkout form")


def make_state(**overrides):
    state = {
        "url": "https://example.com/checkout",
        "title": "Checkout",
        "screenshot": b"\x89PNG-fake-bytes",
        "interactive_elements": [
            {"id": 0, "tag": "button", "text": "Submit", "selector": "#submit-btn"}
        ],
        "dom_elements_truncated": False,
    }
    state.update(overrides)
    return state


class TestDecisionParsing:
    def test_plain_json_decision(self, agent):
        agent.client = FakeGeminiClient([DECISION])
        decision = asyncio.run(agent._decide_next_action(make_state()))
        assert decision["action"] == "click"
        assert decision["selector"] == "#submit-btn"

    def test_markdown_fenced_json_is_tolerated(self, agent):
        fenced = dict(DECISION)
        agent.client = FakeGeminiClient(
            [{"text": "```json\n" + json.dumps(fenced) + "\n```"}]
        )
        decision = asyncio.run(agent._decide_next_action(make_state()))
        assert decision["action"] == "click"

    def test_complete_and_bug_found_actions_pass_through(self, agent):
        for action in ("COMPLETE", "BUG_FOUND"):
            payload = dict(DECISION, action=action)
            agent.client = FakeGeminiClient([payload])
            decision = asyncio.run(agent._decide_next_action(make_state()))
            assert decision["action"] == action

    def test_non_object_json_falls_back_to_next_model(self, agent):
        # First model returns a JSON array → ValueError (non-503) → fallback.
        agent.client = FakeGeminiClient(
            [{"text": "[1, 2, 3]"}, {"text": json.dumps(DECISION)}]
        )
        decision = asyncio.run(agent._decide_next_action(make_state()))
        assert decision["action"] == "click"
        used_models = [c["model"] for c in agent.client.calls]
        assert used_models == ["gemini-2.5-flash", "gemini-3.8-flash"]

    def test_first_model_is_gemini_2_5_flash(self, agent):
        agent.client = FakeGeminiClient([DECISION])
        asyncio.run(agent._decide_next_action(make_state()))
        [call] = agent.client.calls
        assert call["model"] == "gemini-2.5-flash"
        assert call["n_contents"] == 2  # screenshot + prompt


class TestModelSelectionAndRetry:
    def test_prompt_carries_recent_action_memory(self, agent):
        """The model must see prior actions to stop re-filling fields."""
        agent.history.extend(
            [
                {"step": 1, "state": {}, "decision": {"action": "fill", "selector": "#user", "value": "pwd"}},
                {"step": 2, "state": {}, "decision": {"action": "click", "selector": "#login"}},
            ]
        )
        agent.client = FakeGeminiClient([DECISION])
        asyncio.run(agent._decide_next_action(make_state()))

        prompt = agent.client.calls[0]["contents"][1]
        assert "Actions already executed" in prompt
        assert "step 1: fill on '#user' (value already set: 'pwd')" in prompt
        assert "step 2: click on '#login'" in prompt

    def test_empty_history_mentioned_in_prompt(self, agent):
        agent.client = FakeGeminiClient([DECISION])
        asyncio.run(agent._decide_next_action(make_state()))
        prompt = agent.client.calls[0]["contents"][1]
        assert "(none yet)" in prompt
    def test_503_retries_primary_then_succeeds_same_model(self, agent):
        """Two 503s on the primary model, then success on its 3rd attempt."""
        calls = []
        script = ["503", "503", "ok"]

        def generate_content(model=None, contents=None, config=None):
            calls.append(model)
            result = script[len(calls) - 1]
            if result == "503":
                raise Exception(f"Google Gemini {model} error 503 busy")
            return SimpleNamespace(text=json.dumps(DECISION))

        agent.client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )

        decision = asyncio.run(agent._decide_next_action(make_state()))
        assert decision["action"] == "click"

        # 3 attempts stay on the primary model before succeeding
        assert calls == ["gemini-2.5-flash"] * 3

    def test_503_exhausts_primary_then_falls_back(self, agent):
        """All 3 primary attempts 503 → the fallback model is tried."""
        calls = []
        script = ["503", "503", "503", "ok"]

        def generate_content(model=None, contents=None, config=None):
            calls.append(model)
            result = script[len(calls) - 1]
            if result == "503":
                raise Exception(f"Google Gemini {model} error 503 busy")
            return SimpleNamespace(text=json.dumps(DECISION))

        agent.client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )

        decision = asyncio.run(agent._decide_next_action(make_state()))
        assert decision["action"] == "click"

        # primary retried 3×, then fallback succeeds on its 1st attempt
        assert calls == ["gemini-2.5-flash"] * 3 + ["gemini-3.8-flash"]

    def test_429_is_retryable_too(self, agent):
        """Rate limits (429) must back off instead of killing the run."""
        calls = []
        script = ["429", "429", "ok"]

        def generate_content(model=None, contents=None, config=None):
            calls.append(model)
            result = script[len(calls) - 1]
            if result == "429":
                raise Exception(f"429 RESOURCE_EXHAUSTED on {model}")
            return SimpleNamespace(text=json.dumps(DECISION))

        agent.client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )
        decision = asyncio.run(agent._decide_next_action(make_state()))
        assert decision["action"] == "click"
        assert calls == ["gemini-2.5-flash"] * 3

    def test_exhausted_retries_raise_last_exception(self, agent):
        calls = []

        def generate_content(model=None, contents=None, config=None):
            calls.append(model)
            raise Exception(f"Google Gemini {model} error 503 busy")

        agent.client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )
        with pytest.raises(Exception, match="503"):
            asyncio.run(agent._decide_next_action(make_state()))
        # 3 attempts per candidate model × 2 candidate models
        assert len(calls) == 6


class TestConstructionGuards:
    def test_missing_api_key_raises_with_clear_message(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(core_mod.genai, "Client", FakeGeminiClient)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            core_mod.AutoQAAgent(goal="anything")

    def test_headless_env_is_respected(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-unit-tests")
        monkeypatch.setattr(core_mod.genai, "Client", FakeGeminiClient)

        for value, expected in [
            ("false", False),
            ("0", False),
            ("1", True),
            ("true", True),
            ("TRUE", True),
            ("yes", True),
            ("on", True),
        ]:
            monkeypatch.setenv("HEADLESS", value)
            agent = core_mod.AutoQAAgent(goal="x")
            assert agent.driver.headless is expected, value
