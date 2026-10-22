"""Unit tests for src.browser.driver — validation and lifecycle guards.

These tests never launch a browser; lifecycle tests use asyncio.run wrappers.
"""
import asyncio

import pytest

from src.browser.driver import BrowserDriver


class TestConstructorValidation:
    def test_rejects_zero_width(self):
        with pytest.raises(ValueError):
            BrowserDriver(width=0)

    def test_rejects_negative_height(self):
        with pytest.raises(ValueError):
            BrowserDriver(height=-1)

    def test_rejects_zero_timeout(self):
        with pytest.raises(ValueError):
            BrowserDriver(timeout_ms=0)

    def test_rejects_zero_navigation_timeout(self):
        with pytest.raises(ValueError):
            BrowserDriver(navigation_timeout_ms=0)

    def test_rejects_zero_max_dom_elements(self):
        with pytest.raises(ValueError):
            BrowserDriver(max_dom_elements=0)

    def test_accepts_valid_defaults(self):
        driver = BrowserDriver()
        assert driver.headless is True
        assert driver.viewport == {"width": 1440, "height": 900}
        assert driver.page is None  # not started yet


class TestUrlValidation:
    @pytest.mark.parametrize(
        "bad_url",
        [
            "",
            "   ",
            None,
            "not-a-url",
            "ftp://example.com",
            "http://",  # no hostname
            "javascript:alert(1)",
            "file:///etc/hosts",
        ],
    )
    def test_rejects_invalid_urls(self, bad_url):
        driver = BrowserDriver()
        with pytest.raises(ValueError):
            asyncio.run(driver.navigate(bad_url))

    @pytest.mark.parametrize(
        "good_url",
        [
            "https://example.com",
            "http://127.0.0.1:8000/",
            "HTTP://EXAMPLE.COM/path",
        ],
    )
    def test_accepts_valid_urls(self, good_url):
        BrowserDriver._validate_url(good_url)  # must not raise

    def test_port_check_catches_invalid_port(self):
        with pytest.raises(ValueError):
            BrowserDriver._validate_url("http://example.com:99999")


class TestLifecycleGuards:
    def test_capture_state_requires_start(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            asyncio.run(BrowserDriver().capture_state())

    def test_execute_action_requires_start(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            asyncio.run(BrowserDriver().execute_action("click", "#x"))

    def test_navigate_requires_page_but_validates_url_first(self):
        driver = BrowserDriver()
        # Not started → lifecycle error for a valid URL…
        with pytest.raises(RuntimeError, match="not initialized"):
            asyncio.run(driver.navigate("https://example.com"))
        # …but URL validation errors still come back first for bad URLs.
        with pytest.raises(ValueError):
            asyncio.run(driver.navigate("not-a-url"))

    def test_close_before_start_is_a_noop(self):
        asyncio.run(BrowserDriver().close())  # must not raise

    def test_start_launches_and_is_idempotent_and_closes(self):
        """Real Chromium launch: start() twice must reuse/replace cleanly."""
        driver = BrowserDriver(headless=True, timeout_ms=10_000)

        async def scenario():
            await driver.start()
            assert driver.page is not None
            first_page = driver.page
            await driver.start()  # idempotent
            assert driver.page is first_page
            await driver.close()

        asyncio.run(scenario())
        assert driver.page is None
        assert driver.browser is None
        assert driver._playwright is None

    def test_context_manager_starts_and_closes(self):
        async def scenario():
            driver = BrowserDriver(headless=True)
            async with driver as started:
                assert started is driver
                assert driver.page is not None
            assert driver.page is None
            assert driver.browser is None

        asyncio.run(scenario())
