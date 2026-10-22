import os
import json
import asyncio
import re
from typing import Any, Dict, List
from google import genai
from google.genai import types
from src.browser.driver import BrowserDriver

_JSON_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*)\n?\s*```\s*$", re.DOTALL)

TRUTHY = {"1", "true", "yes", "on"}


class AutoQAAgent:
    """Multimodal Vision Agent loop for web interaction."""

    def __init__(self, goal: str, max_steps: int = 10):
        self.goal = goal
        self.max_steps = max_steps

        headless = os.getenv("HEADLESS", "true").strip().lower() in TRUTHY
        self.driver = BrowserDriver(headless=headless)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file, "
                "e.g. `GEMINI_API_KEY=AIza...`, before running the agent."
            )
        self.client = genai.Client(api_key=api_key)
        self.history: List[Dict[str, Any]] = []

    async def run(self, start_url: str) -> List[Dict[str, Any]]:
        """Main agent execution loop."""
        await self.driver.start()
        # Everything after start() must stay inside try so the browser is
        # always closed, even when navigation fails.
        try:
            await self.driver.navigate(start_url)

            for step in range(1, self.max_steps + 1):
                print(f"\n[Step {step}/{self.max_steps}] Capturing UI state...")
                state = await self.driver.capture_state()

                try:
                    decision = await self._decide_next_action(state)
                except Exception as exc:
                    # Fail with clear context instead of a raw API traceback
                    # when every candidate model is unavailable at step N.
                    raise RuntimeError(
                        f"[AutoQA Core] ❌ LLM decision failed at step "
                        f"{step}/{self.max_steps} after exhausting all retry "
                        f"models; run aborted. History recorded: "
                        f"{len(self.history)} step(s). Original error: {exc}"
                    ) from exc
                print(f"[Agent Decision] Reason: {decision.get('thought')}")
                print(
                    f"[Agent Action] Executing: {decision.get('action')}"
                    f" on '{decision.get('selector')}'"
                )

                # Record every step, including terminal COMPLETE/BUG_FOUND
                # decisions, so the returned trajectory is complete.
                self.history.append(
                    {"step": step, "state": state, "decision": decision}
                )

                action = str(decision.get("action", "")).strip().upper()
                if action == "COMPLETE":
                    print("✅ Goal accomplished!")
                    break
                elif action == "BUG_FOUND":
                    print(f"🐛 Bug Detected: {decision.get('bug_description')}")
                    break

                await self.driver.execute_action(
                    action_type=decision.get("action"),
                    selector=decision.get("selector"),
                    text=decision.get("value"),
                )

        finally:
            await self.driver.close()

        return self.history

    async def _decide_next_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calls Gemini Vision API for UI decision with automatic retry logic."""
        system_prompt = f"""You are an Autonomous Web QA Engineer.
Target Goal: {self.goal}

Analyze the UI screenshot and the listed interactive elements. Return JSON strictly matching this schema:
{{
  "thought": "Reasoning about current state toward goal",
  "action": "click" | "fill" | "press_enter" | "COMPLETE" | "BUG_FOUND",
  "selector": "CSS selector to interact with",
  "value": "Text value if action is 'fill', else null",
  "bug_description": "Detailed explanation if action is 'BUG_FOUND', else null"
}}"""

        dom_summary = json.dumps(state["interactive_elements"][:25], indent=2)

        # Give the model short-term memory of what it already did so it stops
        # repeating the same action on already-filled inputs.
        recent_lines = []
        for entry in self.history[-4:]:
            d = entry["decision"]
            line = f"- step {entry['step']}: {d.get('action')} on '{d.get('selector')}'"
            if d.get("value"):
                line += f" (value already set: {d.get('value')!r})"
            recent_lines.append(line)
        recent_summary = "\n".join(recent_lines) or "(none yet)"

        user_prompt = (
            f"Current URL: {state['url']}\n\n"
            f"Interactive Elements Summary:\n{dom_summary}\n\n"
            f"Actions already executed (NEVER repeat one of these on an "
            f"element whose value is already correct; perform the next "
            f"necessary action instead):\n{recent_summary}"
        )

        # List of candidate vision models to attempt in order.
        # gemini-1.5-flash is retired; gemini-2.0-flash now 404s; the API
        # recommends gemini-3.8-flash as the current fallback.
        candidate_models = ["gemini-2.5-flash", "gemini-3.8-flash"]
        last_exception = None
        retryable = ("503", "429", "500", "RESOURCE_EXHAUSTED")

        for model_name in candidate_models:
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Part.from_bytes(
                                data=state["screenshot"], mime_type="image/png"
                            ),
                            f"{system_prompt}\n\n{user_prompt}",
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        ),
                    )
                    return self._parse_decision(response.text)
                except Exception as e:
                    last_exception = e
                    if any(marker in str(e) for marker in retryable):
                        await asyncio.sleep(2 ** attempt)  # Backoff wait
                        continue
                    break  # Not retryable → try next candidate model

        raise last_exception

    @staticmethod
    def _parse_decision(text: Any) -> Dict[str, Any]:
        """Decode a model decision, tolerating markdown-fenced JSON."""
        if not isinstance(text, str):
            raise ValueError("Model response contained no text payload.")

        cleaned = text.strip()
        fence_match = _JSON_FENCE_RE.match(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Decision JSON must be an object.")
        return data
