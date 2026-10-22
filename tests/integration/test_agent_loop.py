"""Integration test: the full AutoQAAgent loop — real Chromium, scripted LLM.

Uses a local static server (no external network) and a stubbed genai client,
so the only real component exercised is the browser + agent control flow.
"""
import asyncio

import pytest

import src.agent.core as core_mod
from tests.integration.helpers import FakeGeminiClient, MAIN_PAGE, StaticSite


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
        core_mod.genai, "Client", lambda api_key=None: FakeGeminiClient(api_key=api_key)
    )


def test_agent_loop_reaches_complete_and_closes_browser(stub_gemini):
    async def scenario():
        with StaticSite({"/": MAIN_PAGE}) as base_url:
            agent = core_mod.AutoQAAgent(goal="Submit the checkout form")
            # Inject the scripted responses before the run starts.
            agent.client = FakeGeminiClient(SCRIPT)
            history = await agent.run(base_url + "/")

        assert agent.driver.page is None, "browser must be closed by run()"

        # 1 fill + 1 click + terminal COMPLETE = 3 recorded steps
        assert len(history) == 3
        used_models = [c["model"] for c in agent.client.calls]
        # gemini-1.5-flash (retired) must never be requested
        assert all(m == "gemini-2.5-flash" for m in used_models)
        assert len(agent.client.calls) == 3

        fill, click, complete = (step["decision"] for step in history)
        assert fill["action"] == "fill" and fill["value"] == "qa@example.com"
        assert click["action"] == "click"
        assert complete["action"] == "COMPLETE"

        # Every recorded step carries a real screenshot
        for step in history:
            assert step["state"]["screenshot"][:4] == b"\x89PNG"

        # The pre-COMPLETE state has the interactive elements we scripted
        for step in history:
            selectors = [el["selector"] for el in step["state"]["interactive_elements"]]
            assert "#email" in selectors and "#submit-btn" in selectors

    asyncio.run(scenario())


def test_agent_loop_bug_found_terminates_successfully(stub_gemini):
    async def scenario():
        with StaticSite({"/": MAIN_PAGE}) as base_url:
            agent = core_mod.AutoQAAgent(goal="Find a bug")
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


def test_agent_loop_records_all_steps_up_to_max_steps(stub_gemini):
    # press_enter needs no selector and never changes the page → repeatable.
    script = [{"thought": "stall", "action": "press_enter"}]

    async def scenario():
        with StaticSite({"/": MAIN_PAGE}) as base_url:
            agent = core_mod.AutoQAAgent(goal="spin", max_steps=3)
            agent.client = FakeGeminiClient(script)
            history = await agent.run(base_url + "/")

        assert len(history) == 3
        assert [s["step"] for s in history] == [1, 2, 3]
        assert all(s["decision"]["action"] == "press_enter" for s in history)

    asyncio.run(scenario())

