"""
scraper.py
==========
Responsible for fetching raw match data (`matchCentreData`) from WhoScored Match Centre URLs.

Key Improvements over the legacy scraper:
1. Primary JS Extraction: Queries `window.require.config.params['args'].matchCentreData`
   directly via Selenium's `execute_script` with explicit polling, avoiding fragile regexes.
2. Safe Balanced-Brace JSON Fallback: Replaces the broken non-greedy regex
   (`r"matchCentreData:\\s*(\\{.*?\\}),\\s*\\n"`) which truncated early on nested JSON objects.
3. Context Manager Support: Ensures the browser process is always closed cleanly.
4. Batch & Offline Support: Can scrape multiple match URLs in a single browser session.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from selenium import webdriver
from selenium.common.exceptions import JavascriptException, WebDriverException
from selenium.webdriver.chrome.options import Options

logger = logging.getLogger(__name__)


class WhoScoredScraperError(Exception):
    """Custom exception raised when match data cannot be extracted from WhoScored."""


class WhoScoredScraper:
    """
    Selenium-based scraper for extracting embedded Opta/WhoScored `matchCentreData` JSON.

    Example usage:
        >>> with WhoScoredScraper(headless=False) as scraper:
        ...     raw_data = scraper.scrape_match("https://www.whoscored.com/Matches/1821123/Live/...")
    """

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )

    # Keys of `require.config.params['args']` that are persisted into the raw JSON,
    # paired with their target key on the saved dictionary (`None` means root `matchCentreData`).
    SAVED_PAGE_ARG_MAP: dict[str, Optional[str]] = {
        "matchCentreData": None,
        "matchCentreEventTypeJson": "_matchCentreEventTypeJson",
        "formationIdNameMappings": "_formationIdNameMappings",
    }
    SAVED_PAGE_ARG_KEYS = tuple(SAVED_PAGE_ARG_MAP.keys())

    def __init__(
        self,
        headless: bool = False,
        timeout_seconds: int = 25,
        poll_interval: float = 1.0,
        user_agent: Optional[str] = None,
    ) -> None:
        """
        Initialize scraper configuration.

        Args:
            headless: Whether to run Chrome in headless mode. False is recommended
                because Cloudflare frequently blocks headless browsers on WhoScored.
            timeout_seconds: Maximum time to wait for `matchCentreData` to load on the page.
            poll_interval: Seconds between checks when waiting for JS data to populate.
            user_agent: Custom User-Agent header string.
        """
        self.headless = headless
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self._driver: Optional[webdriver.Chrome] = None

    def __enter__(self) -> WhoScoredScraper:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _build_chrome_options(self) -> Options:
        """Configures Chrome options to minimize automation detection."""
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")

        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(f"user-agent={self.user_agent}")
        options.add_argument("--disable-infobars")
        options.add_argument("--start-maximized")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        return options

    def start(self) -> None:
        """Starts the underlying Chrome WebDriver instance if not already running."""
        if self._driver is None:
            logger.info("Starting Chrome WebDriver (headless=%s)...", self.headless)
            options = self._build_chrome_options()
            self._driver = webdriver.Chrome(options=options)
            # Mask navigator.webdriver property to reduce bot-detection friction
            try:
                self._driver.execute_cdp_cmd(
                    "Page.addScriptToEvaluateOnNewDocument",
                    {
                        "source": (
                            "Object.defineProperty(navigator, 'webdriver', "
                            "{get: () => undefined})"
                        )
                    },
                )
            except Exception:
                pass

    def close(self) -> None:
        """Closes the Chrome WebDriver instance and releases system resources."""
        if self._driver is not None:
            logger.info("Closing Chrome WebDriver...")
            try:
                self._driver.quit()
            except Exception as exc:
                logger.warning("Error while closing WebDriver: %s", exc)
            finally:
                self._driver = None

    @staticmethod
    def extract_match_id_from_url(match_url: str) -> Optional[int]:
        """
        Extracts the numeric match ID from a WhoScored URL.

        Example:
            "https://www.whoscored.com/Matches/1821123/Live/..." -> 1821123
        """
        match = re.search(r"/Matches/(\d+)", match_url, re.IGNORECASE)
        return int(match.group(1)) if match else None

    @staticmethod
    def extract_json_object_from_html(
        html_source: str,
        key: str = "matchCentreData",
    ) -> Optional[dict[str, Any]]:
        """
        Safely extracts a nested JSON object assigned to `key` from raw HTML/JS source
        by counting balanced curly braces `{}` (respecting string literals and escapes).
        """
        pattern = re.compile(rf"{re.escape(key)}\s*:\s*\{{")
        match = pattern.search(html_source)
        if not match:
            return None

        start_idx = match.end() - 1  # Index of the opening '{'
        depth = 0
        in_string = False
        escape_next = False

        for idx in range(start_idx, len(html_source)):
            char = html_source[idx]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                if in_string:
                    escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = html_source[start_idx : idx + 1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                "Balanced brace extraction found block for '%s' but JSON parsing failed: %s",
                                key,
                                exc,
                            )
                            return None

        return None

    @classmethod
    def _attach_source_metadata(
        cls,
        match_data: dict[str, Any],
        match_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Attaches URL-derived `matchId` (if missing) and `_sourceUrl` to `match_data`."""
        if match_url:
            url_match_id = cls.extract_match_id_from_url(match_url)
            if url_match_id and "matchId" not in match_data:
                match_data["matchId"] = url_match_id
            match_data["_sourceUrl"] = match_url
        return match_data

    @classmethod
    def extract_all_from_html(
        cls,
        html_source: str,
        match_url: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Extracts `matchCentreData` plus all auxiliary lookup dicts defined in
        `SAVED_PAGE_ARG_MAP` from raw HTML source using balanced-brace parsing.
        """
        match_data = cls.extract_json_object_from_html(html_source, key="matchCentreData")
        if not match_data:
            return None

        for page_key, data_key in cls.SAVED_PAGE_ARG_MAP.items():
            if data_key is None:
                continue
            extra = cls.extract_json_object_from_html(html_source, key=page_key)
            if extra:
                match_data[data_key] = extra
            else:
                logger.warning("HTML fallback: '%s' not found in page source.", page_key)
        return cls._attach_source_metadata(match_data, match_url)

    @classmethod
    def find_unsaved_page_args(cls, page_args: Any) -> list[dict[str, Any]]:
        """Returns entries from `page_args` whose `key` is not in `SAVED_PAGE_ARG_KEYS`."""
        if not isinstance(page_args, list):
            return []
        return [
            a
            for a in page_args
            if isinstance(a, dict) and a.get("key") not in cls.SAVED_PAGE_ARG_KEYS
        ]

    @classmethod
    def _report_unsaved_page_args(cls, page_args: Any) -> None:
        """
        Logs every key of `require.config.params['args']` that is NOT persisted into the raw
        JSON, with its JS type, size and a short JSON preview, so nothing is dropped silently.
        """
        if not isinstance(page_args, list):
            logger.warning("Could not enumerate page args keys (got %r).", type(page_args).__name__)
            return

        all_keys = [str(a.get("key")) for a in page_args if isinstance(a, dict)]
        unsaved = cls.find_unsaved_page_args(page_args)
        logger.info("Page args keys found (%d): %s", len(all_keys), ", ".join(all_keys))

        if not unsaved:
            logger.info("All page args keys are saved to the raw JSON.")
            return

        logger.warning("%d page args key(s) are NOT saved to the raw JSON:", len(unsaved))
        for arg in unsaved:
            logger.warning(
                "  - %s | type=%s | size=%s | preview=%s",
                arg.get("key"),
                arg.get("type"),
                arg.get("size"),
                arg.get("preview"),
            )

    def _poll_js_match_data(self) -> Optional[dict[str, Any]]:
        """
        Polls the browser window for `matchCentreData` until it is available or times out.
        Also captures all auxiliary dictionaries listed in `SAVED_PAGE_ARG_MAP` if present
        on the page, and reports every other `args` key that is not being saved.
        """
        assert self._driver is not None
        js_query = """
        try {
            var savedKeys = arguments[0] || [];
            var args = window.require && window.require.config && window.require.config.params &&
                window.require.config.params['args'];
            if (args && args.matchCentreData) {
                var pageArgs = Object.keys(args).map(function (key) {
                    var v = args[key];
                    var type = v === null ? 'null' : (Array.isArray(v) ? 'array' : typeof v);
                    var size = null, preview = null;
                    try {
                        if (type === 'array') size = v.length;
                        else if (type === 'object') size = Object.keys(v).length;
                        else if (type === 'string') size = v.length;
                        var s = type === 'function' ? String(v) : JSON.stringify(v);
                        preview = s === undefined ? null : (s.length > 300 ? s.slice(0, 300) + '...' : s);
                    } catch (e) { preview = '<unserializable: ' + e + '>'; }
                    return {key: key, type: type, size: size, preview: preview};
                });
                var savedArgs = {};
                for (var i = 0; i < savedKeys.length; i++) {
                    var k = savedKeys[i];
                    savedArgs[k] = args[k] !== undefined ? args[k] : null;
                }
                return {
                    savedArgs: savedArgs,
                    pageArgs: pageArgs
                };
            }
        } catch (e) {}
        return null;
        """

        deadline = time.time() + self.timeout_seconds
        saved_keys = list(self.SAVED_PAGE_ARG_KEYS)
        while time.time() < deadline:
            try:
                result = self._driver.execute_script(js_query, saved_keys)
                saved_args = result.get("savedArgs") if isinstance(result, dict) else None
                if isinstance(saved_args, dict) and isinstance(
                    saved_args.get("matchCentreData"), dict
                ):
                    match_data: dict[str, Any] = saved_args["matchCentreData"]
                    self._report_unsaved_page_args(result.get("pageArgs"))
                    for page_key, data_key in self.SAVED_PAGE_ARG_MAP.items():
                        if data_key and saved_args.get(page_key):
                            match_data[data_key] = saved_args[page_key]
                    return match_data
            except (JavascriptException, WebDriverException):
                pass

            time.sleep(self.poll_interval)

        return None

    def scrape_match(self, match_url: str) -> dict[str, Any]:
        """
        Navigates to a WhoScored Match Centre Live URL and extracts the full `matchCentreData` payload.

        Args:
            match_url: Full URL to the WhoScored `/Live/` match page.

        Returns:
            Dictionary containing the complete `matchCentreData` structure.

        Raises:
            WhoScoredScraperError: If `matchCentreData` cannot be retrieved via JS or HTML fallback.
        """
        own_driver = self._driver is None
        if own_driver:
            self.start()

        assert self._driver is not None
        try:
            logger.info("Navigating to match URL: %s", match_url)
            self._driver.get(match_url)

            # Method 1 (Primary & Most Reliable): Poll the browser's parsed JS object
            match_data = self._poll_js_match_data()

            # Method 2 (Fallback): Parse HTML source with balanced-brace JSON extraction
            if not match_data:
                logger.info(
                    "JS global not populated within timeout; trying HTML balanced-brace fallback..."
                )
                html_source = self._driver.page_source
                match_data = self.extract_all_from_html(html_source)
                if match_data:
                    logger.warning(
                        "HTML fallback used: page args keys cannot be enumerated, so unsaved "
                        "keys are NOT reported for this match."
                    )

            if not match_data:
                raise WhoScoredScraperError(
                    f"Could not extract 'matchCentreData' from {match_url}. "
                    "Ensure the URL points to a valid WhoScored '/Live/' match centre page "
                    "and was not blocked by a Cloudflare CAPTCHA challenge."
                )

            self._attach_source_metadata(match_data, match_url)

            events_count = len(match_data.get("events", []))
            logger.info(
                "Successfully extracted match data (matchId=%s, total_events=%d).",
                match_data.get("matchId", "unknown"),
                events_count,
            )
            return match_data

        finally:
            if own_driver:
                self.close()


def scrape_whoscored_match_json(match_url: str, headless: bool = False) -> dict[str, Any]:
    """
    Convenience function (backward-compatible with original script signature)
    to scrape a single WhoScored match URL.
    """
    with WhoScoredScraper(headless=headless) as scraper:
        return scraper.scrape_match(match_url)
