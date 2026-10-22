"""Integration test: BrowserDriver against a real Chromium + local server."""
import asyncio

import pytest

from src.browser.driver import BrowserDriver
from tests.integration.helpers import MAIN_PAGE, StaticSite


@pytest.fixture()
def driver_url():
    with StaticSite({"/": MAIN_PAGE}) as url:
        yield url


def run(coro):
    return asyncio.run(
        asyncio.wait_for(coro, timeout=60)
    )


class TestDriverNavigationAndActions:
    def test_navigate_capture_fill_and_click(self, driver_url):
        async def scenario():
            driver = BrowserDriver(headless=True, timeout_ms=10_000)
            await driver.start()
            try:
                # -- navigate ------------------------------------------------
                await driver.navigate(driver_url)
                assert driver.page.url == driver_url + "/"

                # -- capture_state -------------------------------------------
                state = await driver.capture_state()
                assert state["title"] == "AutoQA test page"
                assert state["screenshot"][:8] == b"\x89PNG\r\n\x1a\n"
                selectors = [el["selector"] for el in state["interactive_elements"]]
                assert "#email" in selectors
                assert "#submit-btn" in selectors
                assert "#counter-btn" in selectors
                # duplicate id-based selectors are stable
                email_el = next(
                    el
                    for el in state["interactive_elements"]
                    if el["selector"] == "#email"
                )
                assert email_el["name"] == "email"
                assert email_el["placeholder"] == "Email address"

                # -- fill -----------------------------------------------------
                await driver.execute_action("fill", "#email", "qa@example.com")
                value = await driver.page.locator("#email").input_value()
                assert value == "qa@example.com"

                # -- click ----------------------------------------------------
                await driver.execute_action("click", "#submit-btn")
                result_text = await driver.page.locator("#result").inner_text()
                assert result_text == "submitted: qa@example.com"
            finally:
                await driver.close()

        run(scenario())

    def test_bad_actions_raise_descriptive_errors(self, driver_url):
        async def scenario():
            driver = BrowserDriver(headless=True, timeout_ms=10_000)
            await driver.start()
            try:
                await driver.navigate(driver_url)

                # unsupported action
                with pytest.raises(ValueError, match="Unsupported action"):
                    await driver.execute_action("scroll_down", "#x")

                # fill without text
                with pytest.raises(ValueError, match="Text is required"):
                    await driver.execute_action("fill", "#email")

                # LLM quirk selectors must never reach Playwright
                for bogus in ("null", "None", "", "  "):
                    with pytest.raises(ValueError, match="non-empty selector"):
                        await driver.execute_action("click", bogus)

                # non-string selector (None included)
                for bogus in (123, None):
                    with pytest.raises(ValueError, match="string selector"):
                        await driver.execute_action("click", bogus)

                # missing element times out within the driver timeout
                with pytest.raises(Exception):
                    await driver.execute_action("click", "#does-not-exist")
            finally:
                await driver.close()

        run(scenario())

    def test_navigation_errors_do_not_leak_the_browser(self, driver_url):
        """An earlier failed navigation must not prevent later use or cleanup."""

        async def scenario():
            driver = BrowserDriver(headless=True, timeout_ms=10_000)
            await driver.start()
            try:
                # Navigating to a closed port raises a network error…
                with pytest.raises(Exception):
                    await driver.navigate("http://127.0.0.1:1/")
                # …and the driver still works afterwards.
                await driver.navigate(driver_url)
                state = await driver.capture_state()
                assert state["title"] == "AutoQA test page"
            finally:
                await driver.close()

        run(scenario())


class TestDomReducerLimits:
    def test_max_dom_elements_is_respected(self, driver_url):
        async def scenario():
            # tiny limit exercises the JS slice + truncate flag
            driver = BrowserDriver(headless=True, max_dom_elements=2)
            await driver.start()
            try:
                await driver.navigate(driver_url)
                state = await driver.capture_state()
                assert len(state["interactive_elements"]) == 2
                assert state["dom_elements_truncated"] is True
            finally:
                await driver.close()

        run(scenario())


NUMERIC_ID_PAGE = """<!DOCTYPE html>
<html>
  <head><title>Numeric IDs</title></head>
  <body>
    <input id="9ca80820-numid" name="UserName" placeholder="user" />
    <input id="password\"odd-id" name="Password" placeholder="pass" />
    <button id="123btn" type="button" onclick="
      document.getElementById('status').textContent = 'clicked'
    ">Go</button>
    <div id="status">idle</div>
  </body>
</html>"""


class TestNumericAndOddIds:
    """Regression: IDs starting with a digit must not produce hex-escape
    CSS like '#\\39 ca...' (and must survive selector.strip())."""

    def test_numeric_and_quote_ids_are_usable(self):
        async def scenario():
            with StaticSite({"/": NUMERIC_ID_PAGE}) as base_url:
                driver = BrowserDriver(headless=True, timeout_ms=10_000)
                await driver.start()
                try:
                    await driver.navigate(base_url)
                    state = await driver.capture_state()
                    selectors = {
                        el["selector"]
                        for el in state["interactive_elements"]
                        if el["selector"]
                    }

                    # attribute-style selectors, no hex escapes anywhere
                    assert '[id="9ca80820-numid"]' in selectors
                    assert "#123btn" not in selectors
                    assert not any(
                        "#\\39" in s or "#\\32" in s for s in selectors
                    ), f"hex-escaped selector leaked: {selectors}"

                    # the generated selectors actually work end-to-end
                    await driver.execute_action(
                        "fill", '[id="9ca80820-numid"]', "qa-user"
                    )
                    value = await driver.page.locator(
                        '[id="9ca80820-numid"]'
                    ).input_value()
                    assert value == "qa-user"

                    await driver.execute_action("click", "[id=\"123btn\"]")
                    status = await driver.page.locator("#status").inner_text()
                    assert status == "clicked"
                finally:
                    await driver.close()

        run(scenario())
