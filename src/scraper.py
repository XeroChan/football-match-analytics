r"""
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
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, ClassVar, Self

from selenium import webdriver
from selenium.common.exceptions import JavascriptException, WebDriverException
from selenium.webdriver.common import utils as selenium_utils
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService

logger = logging.getLogger(__name__)


class WhoScoredScraperError(Exception):
    """Custom exception raised when match data cannot be extracted from WhoScored."""


class _FastFirefoxService(FirefoxService):
    """Firefox Service bound to IPv4 loopback (127.0.0.1) with fast teardown on Windows."""

    def __init__(
        self,
        executable_path: str | None = None,
        port: int = 0,
        service_args: list[str] | None = None,
    ) -> None:
        """Initialize GeckoDriver service forcing IPv4 host binding."""
        args = list(service_args or [])
        has_host = any(arg == "--host" or arg.startswith("--host=") for arg in args)
        if not has_host:
            args = ["--host", "127.0.0.1", *args]
        super().__init__(executable_path=executable_path, port=port, service_args=args)

    @property
    def service_url(self) -> str:
        """Returns the IPv4 loopback URL to avoid Windows `localhost` IPv6 lookup stalls."""
        return f"http://127.0.0.1:{self.port}"

    def is_connectable(self) -> bool:
        """Checks GeckoDriver `/status` directly over IPv4 loopback."""
        return selenium_utils.is_url_connectable(self.port, host="127.0.0.1")

    def send_remote_shutdown_command(self) -> None:
        """Skips the HTTP `/shutdown` poll loop; DELETE /session already closes Firefox."""
        return


class WhoScoredScraper:
    """
    Selenium-based scraper for extracting embedded Opta/WhoScored `matchCentreData` JSON
    using Firefox (GeckoDriver).

    Example usage:
        >>> with WhoScoredScraper(headless=False) as scraper:
        ...     raw_data = scraper.scrape_match(1729376)
    """

    # Keys of `require.config.params['args']` that are persisted into the raw JSON,
    # paired with their target key on the saved dictionary (`None` means root `matchCentreData`).
    SAVED_PAGE_ARG_MAP: ClassVar[dict[str, str | None]] = {
        "matchCentreData": None,
        "matchCentreEventTypeJson": "_matchCentreEventTypeJson",
        "formationIdNameMappings": "_formationIdNameMappings",
    }

    def __init__(
        self,
        headless: bool = False,
        timeout_seconds: int = 25,
        poll_interval: float = 1.0,
        user_agent: str | None = None,
    ) -> None:
        """
        Initialize scraper configuration.

        Args:
            headless: Whether to run Firefox in headless mode. False is recommended
                because Cloudflare frequently blocks headless browsers on WhoScored.
            timeout_seconds: Maximum time to wait for `matchCentreData` to load on the page.
            poll_interval: Seconds between checks when waiting for JS data to populate.
            user_agent: Optional custom User-Agent override. By default Firefox sends its
                native User-Agent so HTTP headers match its TLS/JS fingerprint.
        """
        self.headless = headless
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.user_agent = user_agent
        self._driver: webdriver.Firefox | None = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    @staticmethod
    def _find_cached_geckodriver() -> str | None:
        """
        Locates a cached or installed `geckodriver` binary so Selenium Manager does not
        have to run on every startup. Returns `None` if none is found locally.
        """
        env_driver = os.environ.get("SE_GECKODRIVER")
        if env_driver and Path(env_driver).is_file():
            return env_driver

        which_driver = shutil.which("geckodriver")
        if which_driver and Path(which_driver).is_file():
            return which_driver

        cache_root = Path(
            os.environ.get("SE_CACHE_PATH") or (Path.home() / ".cache" / "selenium"),
        )
        geckodriver_dir = cache_root / "geckodriver"
        if not geckodriver_dir.is_dir():
            return None

        binary_name = "geckodriver.exe" if os.name == "nt" else "geckodriver"
        candidates = [p for p in geckodriver_dir.rglob(binary_name) if p.is_file()]
        if not candidates:
            return None

        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(newest)

    def _build_firefox_options(self) -> FirefoxOptions:
        """Configures Firefox options for fast page loads and minimal background noise."""
        options = FirefoxOptions()
        options.page_load_strategy = "eager"
        if self.headless:
            options.add_argument("-headless")

        preferences: dict[str, str | int | bool] = {
            "dom.webdriver.enabled": False,
            "useAutomationExtension": False,
            "browser.startup.homepage_override.mstone": "ignore",
            "datareporting.policy.dataSubmissionEnabled": False,
            "toolkit.telemetry.enabled": False,
            "app.update.auto": False,
            "app.update.enabled": False,
        }
        if self.user_agent:
            preferences["general.useragent.override"] = self.user_agent

        for name, value in preferences.items():
            options.set_preference(name, value)
        return options

    def start(self) -> None:
        """Starts the underlying Firefox WebDriver instance if not already running."""
        if self._driver is None:
            logger.info("Starting Firefox WebDriver (headless=%s)...", self.headless)
            options = self._build_firefox_options()
            driver_path = self._find_cached_geckodriver()
            service = _FastFirefoxService(executable_path=driver_path)
            self._driver = webdriver.Firefox(options=options, service=service)
            if not self.headless:
                try:
                    self._driver.maximize_window()
                except WebDriverException as exc:
                    logger.debug("Could not maximize Firefox window: %s", exc)

    def close(self) -> None:
        """Closes the Firefox WebDriver instance and releases system resources."""
        if self._driver is not None:
            logger.info("Closing Firefox WebDriver...")
            try:
                self._driver.quit()
            except WebDriverException as exc:
                logger.warning("Error while closing WebDriver: %s", exc)
            finally:
                self._driver = None

    @staticmethod
    def build_match_url(match_id: int) -> str:
        """Builds the canonical WhoScored Match Centre Live URL for a given `match_id`."""
        return f"https://www.whoscored.com/matches/{int(match_id)}/live/"

    @staticmethod
    def extract_json_object_from_html(
        html_source: str,
        key: str = "matchCentreData",
    ) -> dict[str, Any] | None:
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
                                "Balanced brace extraction found block for '%s' "
                                "but JSON parsing failed: %s",
                                key,
                                exc,
                            )
                            return None

        return None

    @classmethod
    def _attach_source_metadata(
        cls,
        match_data: dict[str, Any],
        match_id: int | None = None,
    ) -> dict[str, Any]:
        """Attaches `matchId` and canonical `_sourceUrl` to `match_data`."""
        if match_id is not None:
            if "matchId" not in match_data:
                match_data["matchId"] = int(match_id)
            match_data["_sourceUrl"] = cls.build_match_url(match_id)
        return match_data

    @classmethod
    def extract_all_from_html(
        cls,
        html_source: str,
        match_id: int | None = None,
    ) -> dict[str, Any] | None:
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
        return cls._attach_source_metadata(match_data, match_id)

    @classmethod
    def find_unsaved_page_args(cls, page_args: Any) -> list[dict[str, Any]]:
        """Returns entries from `page_args` whose `key` is not in `SAVED_PAGE_ARG_MAP`."""
        if not isinstance(page_args, list):
            return []
        return [
            a
            for a in page_args
            if isinstance(a, dict) and a.get("key") not in cls.SAVED_PAGE_ARG_MAP
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

    def _poll_js_match_data(self) -> dict[str, Any] | None:
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
                        preview = s === undefined
                            ? null
                            : (s.length > 300 ? s.slice(0, 300) + '...' : s);
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
        saved_keys = list(self.SAVED_PAGE_ARG_MAP)
        while time.time() < deadline:
            try:
                result = self._driver.execute_script(js_query, saved_keys)
                saved_args = result.get("savedArgs") if isinstance(result, dict) else None
                if isinstance(saved_args, dict) and isinstance(
                    saved_args.get("matchCentreData"), dict,
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

    def scrape_match(self, match_id: int) -> dict[str, Any]:
        """
        Navigates to a WhoScored Match Centre Live URL for `match_id` and extracts
        the `matchCentreData` payload.

        Args:
            match_id: Numeric WhoScored match ID (e.g. 1729376).

        Returns:
            Dictionary containing the complete `matchCentreData` structure.

        Raises:
            WhoScoredScraperError: If `matchCentreData` cannot be retrieved via JS or HTML fallback.
        """
        match_url = self.build_match_url(match_id)
        assert self._driver is not None

        try:
            logger.info("Navigating to match URL: %s", match_url)
            self._driver.get(match_url)
        except WebDriverException as exc:
            raise WhoScoredScraperError(
                f"Browser navigation failed for {match_url}: {exc}",
            ) from exc

        # Method 1 (Primary & Most Reliable): Poll the browser's parsed JS object
        match_data = self._poll_js_match_data()

        # Method 2 (Fallback): Parse HTML source with balanced-brace JSON extraction
        if not match_data:
            logger.info(
                "JS global not populated within timeout; "
                "trying HTML balanced-brace fallback...",
            )
            try:
                html_source = self._driver.page_source
            except WebDriverException as exc:
                raise WhoScoredScraperError(
                    f"Failed to read page source from {match_url}: {exc}",
                ) from exc
            match_data = self.extract_all_from_html(html_source)
            if match_data:
                logger.warning(
                    "HTML fallback used: page args keys cannot be enumerated, so unsaved "
                    "keys are NOT reported for this match.",
                )

        if not match_data:
            raise WhoScoredScraperError(
                f"Could not extract 'matchCentreData' from {match_url}. "
                "Ensure the URL points to a valid WhoScored '/Live/' match centre page "
                "and was not blocked by a Cloudflare CAPTCHA challenge.",
            )

        self._attach_source_metadata(match_data, match_id)

        events_count = len(match_data.get("events", []))
        logger.info(
            "Successfully extracted match data (matchId=%s, total_events=%d).",
            match_data.get("matchId", "unknown"),
            events_count,
        )
        return match_data
