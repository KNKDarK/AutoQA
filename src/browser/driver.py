"""Reliable Playwright browser driver for AI browser agents."""

import asyncio
import os
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
)


class BrowserDriver:
    """Manage browser lifecycle, navigation, DOM snapshots, and actions."""

    def __init__(
        self,
        headless: bool = True,
        width: int = 1440,
        height: int = 900,
        timeout_ms: int = 15_000,
        navigation_timeout_ms: int = 30_000,
        max_dom_elements: int = 500,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Viewport width and height must be positive.")
        if timeout_ms <= 0 or navigation_timeout_ms <= 0:
            raise ValueError("Timeouts must be positive.")
        if max_dom_elements <= 0:
            raise ValueError("max_dom_elements must be positive.")

        self.headless = headless
        self.viewport = {"width": width, "height": height}
        self.timeout_ms = timeout_ms
        self.navigation_timeout_ms = navigation_timeout_ms
        self.max_dom_elements = max_dom_elements

        self._playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start Chromium and clean up partial startup on failure."""
        async with self._lock:
            if self.page and not self.page.is_closed():
                return

            await self._close_unlocked()

            try:
                self._playwright = await async_playwright().start()

                exec_path = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
                if exec_path and not os.path.isfile(exec_path):
                    raise FileNotFoundError(
                        "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH does not "
                        f"point to a file: {exec_path}"
                    )

                if not exec_path and os.path.isfile("/usr/bin/chromium"):
                    exec_path = "/usr/bin/chromium"

                launch_kwargs: Dict[str, Any] = {
                    "headless": self.headless,
                    "timeout": self.timeout_ms,
                    "args": ["--disable-setuid-sandbox"],
                }
                if exec_path:
                    launch_kwargs["executable_path"] = exec_path

                self.browser = await self._playwright.chromium.launch(
                    **launch_kwargs
                )
                self.context = await self.browser.new_context(
                    viewport=self.viewport
                )
                self.context.set_default_timeout(self.timeout_ms)
                self.context.set_default_navigation_timeout(
                    self.navigation_timeout_ms
                )
                self.page = await self.context.new_page()

            except Exception:
                await self._close_unlocked()
                raise

    def _require_page(self) -> Page:
        """Return the active page or raise a clear lifecycle error."""
        if self.page is None or self.page.is_closed():
            raise RuntimeError("Browser is not initialized. Call start() first.")
        return self.page

    @staticmethod
    def _validate_url(url: str) -> None:
        """Reject malformed URLs and unsupported schemes."""
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string.")

        try:
            parsed = urlsplit(url.strip())
            hostname = parsed.hostname
            _ = parsed.port
        except ValueError as exc:
            raise ValueError(f"Invalid URL: {url!r}") from exc

        if parsed.scheme.lower() not in {"http", "https"} or not hostname:
            raise ValueError("URL must be an absolute HTTP or HTTPS URL.")

    async def navigate(self, url: str) -> None:
        """Navigate to a URL without relying on network-idle detection."""
        self._validate_url(url)

        async with self._lock:
            page = self._require_page()
            try:
                await page.goto(
                    url.strip(),
                    wait_until="domcontentloaded",
                    timeout=self.navigation_timeout_ms,
                )
            except PlaywrightTimeoutError as exc:
                raise TimeoutError(
                    f"Navigation timed out after "
                    f"{self.navigation_timeout_ms} ms: {url}"
                ) from exc
            except Exception as exc:
                if "interrupted by another navigation" not in str(exc):
                    raise
                # After a network failure Chromium loads an internal
                # chrome-error:// page; wait for it to settle and retry once
                # so a failed navigation does not sabotage the next one.
                try:
                    await page.wait_for_load_state(
                        "domcontentloaded",
                        timeout=self.navigation_timeout_ms,
                    )
                except Exception:
                    pass  # already settled — the retry below decides the outcome
                await page.goto(
                    url.strip(),
                    wait_until="domcontentloaded",
                    timeout=self.navigation_timeout_ms,
                )

    async def capture_state(self) -> Dict[str, Any]:
        """Capture screenshot bytes and a bounded interactive DOM snapshot."""
        async with self._lock:
            page = self._require_page()

            for attempt in range(2):
                try:
                    screenshot = await page.screenshot(
                        full_page=False,
                        timeout=self.timeout_ms,
                    )
                    dom_snapshot = await page.evaluate(
                        """(maxElements) => {
                            const allElements = Array.from(
                                document.querySelectorAll(
                                    'a, button, input, select, textarea, ' +
                                    '[role="button"], [role="link"], ' +
                                    '[role="checkbox"], [role="radio"]'
                                )
                            );
                            const elements = allElements.slice(0, maxElements);

                            return {
                                elements: elements.map((el, idx) => {
                                    const tag = el.tagName.toLowerCase();
                                    const text = (
                                        el.innerText ||
                                        el.value ||
                                        el.getAttribute('aria-label') ||
                                        el.getAttribute('placeholder') ||
                                        ''
                                    ).trim().slice(0, 500);

                                    const id = el.id || '';
                                    const name = el.getAttribute('name') || '';
                                    const type = el.getAttribute('type') || '';

                                    const escapeAttr = (v) => String(v)
                                        .replace(/\\\\/g, '\\\\\\\\')
                                        .replace(/"/g, '\\\\"');

                                    let selector = null;
                                    if (id) {
                                        // IDs starting with a digit would need
                                        // CSS hex escapes (#\\39 ca...) which are
                                        // brittle; prefer [id="..."] instead.
                                        selector = /^[0-9]/.test(id)
                                            ? '[id="' + escapeAttr(id) + '"]'
                                            : '#' + CSS.escape(id);
                                    } else if (
                                        name &&
                                        ['input', 'select', 'textarea'].includes(tag)
                                    ) {
                                        selector = tag +
                                            '[name="' + escapeAttr(name) + '"]';
                                    }

                                    return {
                                        id: idx,
                                        tag,
                                        text,
                                        type,
                                        name,
                                        selector,
                                        role: el.getAttribute('role') || '',
                                        placeholder:
                                            el.getAttribute('placeholder') || '',
                                        aria_label:
                                            el.getAttribute('aria-label') || '',
                                        disabled: Boolean(el.disabled),
                                        visible: Boolean(el.getClientRects().length),
                                    };
                                }),
                                truncated: allElements.length > maxElements
                            };
                        }""",
                        self.max_dom_elements,
                    )
                    break
                except Exception as exc:
                    if (
                        "Execution context was destroyed" not in str(exc)
                        or attempt == 1
                    ):
                        raise

                    await page.wait_for_load_state(
                        "domcontentloaded",
                        timeout=self.navigation_timeout_ms,
                    )

            return {
                "url": page.url,
                "title": await page.title(),
                "screenshot": screenshot,
                "interactive_elements": dom_snapshot["elements"],
                "dom_elements_truncated": dom_snapshot["truncated"],
            }

    async def execute_action(
        self,
        action_type: str,
        selector: Optional[str] = None,
        text: Optional[str] = None,
    ) -> None:
        """Execute an interaction on the active page."""
        async with self._lock:
            page = self._require_page()

            # Enter is a global keyboard action and does not need a selector.
            if action_type == "press_enter":
                await page.keyboard.press("Enter")
                return

            if not isinstance(selector, str):
                raise ValueError(
                    f"A string selector is required for action: {action_type}"
                )
            normalized = selector.strip()
            if not normalized or normalized.lower() in {"none", "null"}:
                raise ValueError(
                    f"A non-empty selector is required for action: {action_type}"
                )
            selector = normalized

            if action_type == "click":
                await page.click(selector, timeout=self.timeout_ms)
            elif action_type == "fill":
                if text is None:
                    raise ValueError("Text is required for fill action.")
                await page.fill(selector, text, timeout=self.timeout_ms)
            else:
                raise ValueError(f"Unsupported action: {action_type}")

    async def _close_unlocked(self) -> None:
        """Release all resources, even if an earlier close step fails."""
        page, context, browser, playwright = (
            self.page,
            self.context,
            self.browser,
            self._playwright,
        )

        self.page = None
        self.context = None
        self.browser = None
        self._playwright = None

        errors = []

        for resource, method in (
            (page, "close"),
            (context, "close"),
            (browser, "close"),
            (playwright, "stop"),
        ):
            if resource is None:
                continue
            try:
                await getattr(resource, method)()
            except Exception as exc:
                errors.append(exc)

        if errors:
            raise RuntimeError(
                f"Browser cleanup encountered {len(errors)} error(s)."
            ) from errors[0]

    async def close(self) -> None:
        """Close the page, context, browser, and Playwright runtime."""
        async with self._lock:
            await self._close_unlocked()

    async def __aenter__(self) -> "BrowserDriver":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()
