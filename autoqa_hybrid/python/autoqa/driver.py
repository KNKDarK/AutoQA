"""Playwright browser controller for the hybrid AutoQA agent.

Same hardened lifecycle as the pure-Python `src.browser.driver`, extended to
flatten each captured DOM into a hashed token stream for the Rust delta
engine.
"""
import asyncio
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
)

try:  # native engine is optional at import time
    from autoqa import _native as native
except ImportError:  # pragma: no cover - uncompiled checkouts still work
    native = None

DOM_HASH_SEED = 0xCBF29CE484222325
DOM_HASH_PRIME = 0x100000001B3


def _fnv1a(tokens: List[str], seed: int = DOM_HASH_SEED) -> int:
    """Pure-Python FNV-1a; used only when the native module is unavailable."""
    digest = seed
    for token in tokens:
        for byte in token.encode("utf-8"):
            digest = ((digest ^ byte) * DOM_HASH_PRIME) & 0xFFFFFFFFFFFFFFFF
        digest = ((digest ^ 0xFF) * DOM_HASH_PRIME) & 0xFFFFFFFFFFFFFFFF
    return digest


def flatten_dom_tokens(elements: List[Dict[str, Any]]) -> List[str]:
    """Flatten an interactive-elements snapshot into a stable token stream."""
    tokens: List[str] = []
    for el in elements:
        tokens.append(
            "|".join(
                (
                    "tag:" + str(el.get("tag", "")),
                    "id" + str(el.get("id", "")),
                    "sel:" + str(el.get("selector") or ""),
                    "name:" + str(el.get("name") or ""),
                    "role:" + str(el.get("role") or ""),
                    "text:" + str(el.get("text") or "")[:120],
                    "dis" if el.get("disabled") else "ena",
                    "vis" if el.get("visible") else "hid",
                )
            )
        )
    return tokens


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

                self.browser = await self._playwright.chromium.launch(**launch_kwargs)
                self.context = await self.browser.new_context(viewport=self.viewport)
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
    def _validate_url(url: Any) -> None:
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
                # Chromium may insert an internal chrome-error:// navigation
                # right after a network failure; let it settle and retry once.
                try:
                    await page.wait_for_load_state(
                        "domcontentloaded", timeout=self.navigation_timeout_ms
                    )
                except Exception:
                    pass
                await page.goto(
                    url.strip(),
                    wait_until="domcontentloaded",
                    timeout=self.navigation_timeout_ms,
                )

    async def capture_state(self) -> Dict[str, Any]:
        """Capture a screenshot, DOM snapshot, and native-dom-hash tokens."""
        async with self._lock:
            page = self._require_page()

            for attempt in range(2):
                try:
                    screenshot = await page.screenshot(
                        full_page=False,
                        timeout=self.timeout_ms,
                    )
                    dom_snapshot = await self._snapshot_dom(page)
                    break
                except Exception as exc:
                    if (
                        "Execution context was destroyed" not in str(exc)
                        or attempt == 1
                    ):
                        raise
                    await page.wait_for_load_state(
                        "domcontentloaded", timeout=self.navigation_timeout_ms
                    )

            elements = dom_snapshot["elements"]
            tokens = flatten_dom_tokens(elements)
            if native is not None:
                dom_hash = native.hash_tokens(tokens)
            else:  # pragma: no cover
                dom_hash = _fnv1a(tokens)

            return {
                "url": page.url,
                "title": await page.title(),
                "screenshot": screenshot,
                "interactive_elements": elements,
                "dom_elements_truncated": dom_snapshot["truncated"],
                "dom_tokens": tokens,
                "dom_hash": dom_hash,
            }

    async def _snapshot_dom(self, page: Page) -> Dict[str, Any]:
        """JS-reduced bounded snapshot of interactive elements on the page."""
        return await page.evaluate(
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
                            // CSS hex escapes (#\\39 ...) which are
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

    async def execute_action(
        self,
        action_type: str,
        selector: Optional[str] = None,
        text: Optional[str] = None,
    ) -> None:
        """Execute an interaction on the active page."""
        async with self._lock:
            page = self._require_page()

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
