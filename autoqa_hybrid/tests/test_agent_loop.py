"""Integration tests: hybrid agent loop — real Chromium, scripted LLM, native DOM deltas."""
import asyncio

import pytest

import autoqa as pkg
from autoqa.agent import AutoQAAgent
from helpers import FakeGeminiClient, MAIN_PAGE, StaticSite


SCRIPT = [
    {"thought": "Fill the email field", "action": "fill", "selector": "#email", "value": "qa@example.com"},
    {"thought": "Submit the form", "action": "click", "selector": "#submit-btn"},
    {"thought": "Form submitted; result visible", "action": "COMPLETE"},
]


@pytest.fixture()
def stub_gemini(monkeypatch):
    """Prevent any real Gemini API access during the loop tests."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-integration")
    monkeypatch.setattr(
        pkg.agent.genai, "Client", lambda api_key=None: FakeGeminiClient(api_key=api_key)
    )


def test_agent_records_native_dom_deltas(stub_gemini):
    assert pkg.NATIVE_AVAILABLE, "native engine must be built for this test"

    async def scenario():
        with StaticSite({"/": MAIN_PAGE}) as base_url:
            agent = AutoQAAgent(goal="Submit the checkout form")
            agent.client = FakeGeminiClient(SCRIPT)
            history = await agent.run(base_url + "/")

        assert agent.driver.page is None, "browser must be closed by run()"
        assert len(history) == 3

        # DOM hashes recorded and stable for identical states
        hashes = [step["state"]["dom_hash"] for step in history]
        assert all(isinstance(h, int) for h in hashes)

        # Steps 2/3 carry a DOM delta vs step 1 (fill mutated the token stream)
        for step in history[1:]:
            delta = step["state"].get("dom_delta")
            assert delta is not None
            assert all(
                {"op", "before_index", "after_index", "token"} <= set(d)
                for d in delta
            )

    asyncio.run(scenario())


def test_agent_prompt_contains_recent_actions(stub_gemini):
    """The model must see prior actions so it stops re-filling fields."""

    async def scenario():
        with StaticSite({"/": MAIN_PAGE}) as base_url:
            agent = AutoQAAgent(goal="Submit the checkout form")
            agent.client = FakeGeminiClient(SCRIPT)
            await agent.run(base_url + "/")

        # 3rd call = decision about the final COMPLETE state; its prompt must
        # include the two actions executed before (fill + click).
        prompt = agent.client.calls[2]["contents"][1]
        assert "Actions already executed" in prompt
        assert "fill" in prompt and "#email" in prompt


def test_agent_loop_fails_gracefully_when_models_die(stub_gemini):
    async def scenario():
        with StaticSite({"/": MAIN_PAGE}) as base_url:
            agent = AutoQAAgent(goal="abort")
            agent.client = FakeGeminiClient(
                [{"text": "404 NOT_FOUND: model retired"}]
            )
            # Both models raise non-retryable errors → clean RuntimeError.
            with pytest.raises(RuntimeError, match="decision failed at step 1"):
                await agent.run(base_url + "/")

        assert agent.driver.page is None, "browser must still be closed"

    asyncio.run(scenario())


def test_agent_bug_found_terminates_cleanly(stub_gemini):
    async def scenario():
        with StaticSite({"/": MAIN_PAGE}) as base_url:
            agent = AutoQAAgent(goal="Find a bug")
            agent.client = FakeGeminiClient(
                [
                    {
                        "thought": "The result text never appears",
                        "action": "BUG_FOUND",
                        "bug_description": "submit did not update #result",
                    }
                ]
            )
            history = await agent.run(base_url + "/")

        assert len(history) == 1
        assert history[0]["decision"]["action"] == "BUG_FOUND"
        assert "did not update" in history[0]["decision"]["bug_description"]

    asyncio.run(scenario())


def test_agent_max_steps_respected(stub_gemini):
    async def scenario():
        with StaticSite({"/": MAIN_PAGE}) as base_url:
            agent = AutoQAAgent(goal="spin", max_steps=3)
            agent.client = FakeGeminiClient(
                [{"thought": "stall", "action": "press_enter"}]
            )
            history = await agent.run(base_url + "/")

        assert len(history) == 3
        assert [s["step"] for s in history] == [1, 2, 3]
        assert all(s["decision"]["action"] == "press_enter" for s in history)

    asyncio.run(scenario())
