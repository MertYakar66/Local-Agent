"""MCP (Model Context Protocol) server for Claude Desktop integration.

IMPORTANT: This server makes NO LLM calls. Claude Desktop does all reasoning.
This is pure browser execution only, resulting in $0 API token costs.

Uses the official MCP Python SDK for proper protocol version negotiation.

Tools provided:
- Navigation: browse_to, go_back, go_forward
- Waiting: wait_for (selector/timeout/network) - CRITICAL for dynamic content
- Interaction: click, fill, press_key, scroll, select_option
- Extraction: get_page_info, get_dom_elements, extract_text, extract_table
- Debugging: take_screenshot
- Audit: get_action_log
- Multi-tab: new_tab, switch_tab, close_tab, list_tabs

Usage:
  1. Install: pip install "mcp>=1.0.0"
  2. Add to Claude Desktop's claude_desktop_config.json:
     {
       "mcpServers": {
         "browser-agent": {
           "command": "/path/to/venv/bin/python",
           "args": ["-m", "src.mcp_server"],
           "cwd": "/path/to/Local-Agent"
         }
       }
     }
  3. Restart Claude Desktop
  4. Ask Claude to browse websites, fill forms, extract data
"""

import base64
import json
import os
import sys
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Set

from pydantic import Field

# Official MCP SDK - handles protocol negotiation automatically
from mcp.server.fastmcp import FastMCP
from playwright.async_api import Page, async_playwright, Playwright, Browser, BrowserContext


# ============ Security: Domain Allowlist ============

DEFAULT_ALLOWED_DOMAINS: Set[str] = {
    # Search engines
    "google.com", "www.google.com",
    "bing.com", "www.bing.com",
    "duckduckgo.com",
    # Reference
    "wikipedia.org", "en.wikipedia.org",
    # Example/test
    "example.com", "example.org", "example.net",
}


def get_allowed_domains() -> Set[str]:
    """Get allowed domains from env or use defaults."""
    env_domains = os.getenv("MCP_ALLOWED_DOMAINS", "")
    if env_domains:
        return set(d.strip().lower() for d in env_domains.split(",") if d.strip())
    return DEFAULT_ALLOWED_DOMAINS


def is_domain_allowed(url: str) -> tuple[bool, str]:
    """Check if URL domain is in allowlist. Returns (allowed, reason)."""
    allowed = get_allowed_domains()
    if not allowed:  # Empty allowlist = allow all (for development)
        return True, ""

    try:
        from urllib.parse import urlparse
        parsed = urlparse(url if "://" in url else f"https://{url}")
        domain = parsed.netloc.lower()

        # Check exact match or subdomain match
        for allowed_domain in allowed:
            if domain == allowed_domain or domain.endswith(f".{allowed_domain}"):
                return True, ""

        return False, f"Domain '{domain}' not in allowlist. Add to MCP_ALLOWED_DOMAINS env var."
    except Exception as e:
        return False, f"Invalid URL: {e}"


# ============ Browser Server (Pure Execution, NO LLM calls) ============

class MCPBrowserServer:
    """
    Pure execution browser server. NO LLM calls are made here.
    Claude Desktop performs all reasoning; this just executes browser commands.
    """

    def __init__(self):
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._tabs: Dict[int, Page] = {}
        self._active_tab: int = 0
        self._initialized = False
        self._action_log: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize browser (headless=False so user can see)."""
        if self._initialized:
            return

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )

        # Create initial tab
        page = await self._context.new_page()
        self._tabs[0] = page
        self._active_tab = 0
        self._initialized = True

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._initialized = False

    def _get_page(self) -> Page:
        """Get active page."""
        if not self._initialized:
            raise RuntimeError("Browser not initialized. Call a navigation tool first.")
        return self._tabs[self._active_tab]

    def _log_action(self, tool: str, args: Dict, result: Dict) -> None:
        """Log action for audit trail. Sensitive data is redacted."""
        # Redact sensitive fields
        safe_args = {k: "[REDACTED]" if k in ("value", "password", "secret", "token") else v
                     for k, v in args.items()}
        self._action_log.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool,
            "args": safe_args,
            "success": result.get("success", False),
        })
        # Keep last 100 actions
        if len(self._action_log) > 100:
            self._action_log = self._action_log[-100:]

    async def _validate_current_url(self, page: Page) -> tuple[bool, str]:
        """Validate that current page URL is in allowlist. Used after redirects/history nav."""
        current_url = page.url
        if current_url in ("about:blank", ""):
            return True, ""
        allowed, reason = is_domain_allowed(current_url)
        if not allowed:
            # Navigate away from disallowed domain
            await page.goto("about:blank")
            return False, f"Navigation blocked: landed on disallowed domain. {reason}"
        return True, ""

    # ---- Navigation ----

    async def browse_to(self, url: str) -> Dict[str, Any]:
        await self.initialize()
        allowed, reason = is_domain_allowed(url)
        if not allowed:
            result = {"success": False, "error": reason}
            self._log_action("browse_to", {"url": url}, result)
            return result

        page = self._get_page()
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # Validate final URL after potential redirects
            valid, block_reason = await self._validate_current_url(page)
            if not valid:
                result = {"success": False, "error": block_reason}
            else:
                result = {
                    "success": True,
                    "url": page.url,
                    "title": await page.title(),
                }
        except Exception as e:
            result = {"success": False, "error": str(e)}

        self._log_action("browse_to", {"url": url}, result)
        return result

    async def go_back(self) -> Dict[str, Any]:
        await self.initialize()
        page = self._get_page()
        try:
            await page.go_back(wait_until="domcontentloaded", timeout=10000)
            # Validate URL after history navigation
            valid, block_reason = await self._validate_current_url(page)
            if not valid:
                result = {"success": False, "error": block_reason}
            else:
                result = {"success": True, "url": page.url}
        except Exception as e:
            result = {"success": False, "error": str(e)}
        self._log_action("go_back", {}, result)
        return result

    async def go_forward(self) -> Dict[str, Any]:
        await self.initialize()
        page = self._get_page()
        try:
            await page.go_forward(wait_until="domcontentloaded", timeout=10000)
            # Validate URL after history navigation
            valid, block_reason = await self._validate_current_url(page)
            if not valid:
                result = {"success": False, "error": block_reason}
            else:
                result = {"success": True, "url": page.url}
        except Exception as e:
            result = {"success": False, "error": str(e)}
        self._log_action("go_forward", {}, result)
        return result

    # ---- Waiting ----

    async def wait_for(
        self,
        selector: Optional[str] = None,
        state: str = "visible",
        timeout: int = 10000,
        wait_for_network: bool = False,
    ) -> Dict[str, Any]:
        await self.initialize()
        # Require at least one wait condition to prevent no-op calls
        if not selector and not wait_for_network:
            result = {
                "success": False,
                "error": "Must specify either 'selector' or 'wait_for_network=true'. No-op wait rejected.",
            }
            self._log_action("wait_for", {"selector": selector, "wait_for_network": wait_for_network}, result)
            return result

        page = self._get_page()
        try:
            if wait_for_network:
                await page.wait_for_load_state("networkidle", timeout=timeout)
            if selector:
                await page.wait_for_selector(selector, state=state, timeout=timeout)
            result = {"success": True, "waited_for": selector or "network"}
        except Exception as e:
            result = {"success": False, "error": str(e), "hint": "Element may not exist or page still loading"}
        self._log_action("wait_for", {"selector": selector, "wait_for_network": wait_for_network}, result)
        return result

    # ---- Interaction ----

    async def click(self, selector: str, timeout: int = 10000) -> Dict[str, Any]:
        await self.initialize()
        page = self._get_page()
        url_before = page.url
        try:
            await page.click(selector, timeout=timeout)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=3000)
            except:
                pass
            url_changed = page.url != url_before
            # Validate URL if navigation occurred (click might have navigated)
            if url_changed:
                valid, block_reason = await self._validate_current_url(page)
                if not valid:
                    result = {"success": False, "error": block_reason}
                    self._log_action("click", {"selector": selector}, result)
                    return result
            result = {
                "success": True,
                "url": page.url,
                "url_changed": url_changed,
            }
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "hint": "Use get_dom_elements to find correct selector, or wait_for first",
            }
        self._log_action("click", {"selector": selector}, result)
        return result

    async def fill(self, selector: str, value: str, clear_first: bool = True) -> Dict[str, Any]:
        await self.initialize()
        page = self._get_page()
        try:
            if clear_first:
                await page.fill(selector, value, timeout=10000)
            else:
                await page.type(selector, value, timeout=10000)
            result = {"success": True, "filled_length": len(value)}
        except Exception as e:
            result = {"success": False, "error": str(e)}
        # Log with value field - will be auto-redacted by _log_action
        self._log_action("fill", {"selector": selector, "value": value}, result)
        return result

    async def press_key(self, key: str = "Enter") -> Dict[str, Any]:
        await self.initialize()
        page = self._get_page()
        url_before = page.url
        try:
            await page.keyboard.press(key)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
            except:
                pass
            url_changed = page.url != url_before
            # Validate URL if navigation occurred
            if url_changed:
                valid, block_reason = await self._validate_current_url(page)
                if not valid:
                    result = {"success": False, "error": block_reason}
                    self._log_action("press_key", {"key": key}, result)
                    return result
            result = {
                "success": True,
                "key": key,
                "url": page.url,
                "url_changed": url_changed,
            }
        except Exception as e:
            result = {"success": False, "error": str(e)}
        self._log_action("press_key", {"key": key}, result)
        return result

    async def scroll(self, direction: str = "down", amount: int = 500) -> Dict[str, Any]:
        await self.initialize()
        page = self._get_page()
        delta = amount if direction == "down" else -amount
        await page.evaluate(f"window.scrollBy(0, {delta})")
        scroll_pos = await page.evaluate("window.scrollY")
        result = {"success": True, "direction": direction, "scroll_position": scroll_pos}
        self._log_action("scroll", {"direction": direction, "amount": amount}, result)
        return result

    async def select_option(self, selector: str, value: str) -> Dict[str, Any]:
        await self.initialize()
        page = self._get_page()
        try:
            await page.select_option(selector, value, timeout=10000)
            result = {"success": True, "selected": value}
        except Exception as e:
            result = {"success": False, "error": str(e)}
        self._log_action("select_option", {"selector": selector, "value": value}, result)
        return result

    # ---- Extraction ----

    async def get_page_info(self) -> Dict[str, Any]:
        await self.initialize()
        page = self._get_page()
        result = {
            "url": page.url,
            "title": await page.title(),
            "tab_id": self._active_tab,
        }
        self._log_action("get_page_info", {}, {"success": True})
        return result

    async def get_dom_elements(self, max_elements: int = 50) -> Dict[str, Any]:
        await self.initialize()
        page = self._get_page()
        dom_data = await page.evaluate("""(maxElements) => {
            const elements = [];
            const selectors = 'a, button, input, select, textarea, [role="button"], [onclick], [tabindex]';
            document.querySelectorAll(selectors).forEach((el, idx) => {
                if (idx >= maxElements) return;
                if (el.offsetParent === null && el.tagName !== 'INPUT') return;
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                let selector = '';
                if (el.id) {
                    selector = '#' + el.id;
                } else if (el.name) {
                    selector = `${el.tagName.toLowerCase()}[name="${el.name}"]`;
                } else if (el.className && typeof el.className === 'string') {
                    selector = el.tagName.toLowerCase() + '.' + el.className.split(' ')[0];
                } else {
                    selector = el.tagName.toLowerCase();
                }
                elements.push({
                    tag: el.tagName.toLowerCase(),
                    type: el.type || null,
                    text: (el.textContent || '').trim().slice(0, 50),
                    placeholder: el.placeholder || null,
                    value: el.value ? el.value.slice(0, 30) : null,
                    selector: selector,
                    aria_label: el.getAttribute('aria-label'),
                });
            });
            return {
                title: document.title,
                url: window.location.href,
                element_count: elements.length,
                elements: elements.slice(0, maxElements),
            };
        }""", max_elements)
        self._log_action("get_dom_elements", {"max_elements": max_elements}, {"success": True, "count": dom_data.get("element_count", 0)})
        return dom_data

    async def extract_text(self, selector: str, max_items: int = 20) -> Dict[str, Any]:
        await self.initialize()
        page = self._get_page()
        try:
            elements = await page.query_selector_all(selector)
            texts = []
            for el in elements[:max_items]:
                text = await el.text_content()
                if text and text.strip():
                    texts.append(text.strip())
            result = {"success": True, "selector": selector, "count": len(texts), "texts": texts}
        except Exception as e:
            result = {"success": False, "error": str(e)}
        self._log_action("extract_text", {"selector": selector, "max_items": max_items}, result)
        return result

    async def extract_table(self, selector: str = "table") -> Dict[str, Any]:
        await self.initialize()
        page = self._get_page()
        try:
            table_data = await page.evaluate("""(selector) => {
                const table = document.querySelector(selector);
                if (!table) return { error: 'Table not found' };
                const headers = [];
                const headerRow = table.querySelector('thead tr, tr:first-child');
                if (headerRow) {
                    headerRow.querySelectorAll('th, td').forEach(cell => {
                        headers.push(cell.textContent.trim());
                    });
                }
                const rows = [];
                const dataRows = table.querySelectorAll('tbody tr, tr:not(:first-child)');
                dataRows.forEach((row, idx) => {
                    if (idx >= 50) return;
                    const cells = row.querySelectorAll('td');
                    if (cells.length === 0) return;
                    const rowData = {};
                    cells.forEach((cell, i) => {
                        const key = headers[i] || `column_${i}`;
                        rowData[key] = cell.textContent.trim();
                    });
                    rows.push(rowData);
                });
                return { headers: headers, row_count: rows.length, rows: rows };
            }""", selector)
            if "error" in table_data:
                result = {"success": False, "error": table_data["error"]}
            else:
                result = {"success": True, **table_data}
        except Exception as e:
            result = {"success": False, "error": str(e)}
        self._log_action("extract_table", {"selector": selector}, result)
        return result

    # ---- Debugging ----

    async def take_screenshot(self, full_page: bool = False) -> Dict[str, Any]:
        await self.initialize()
        page = self._get_page()
        try:
            screenshot_bytes = await page.screenshot(type="png", full_page=full_page)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            result = {
                "success": True,
                "format": "png",
                "base64": screenshot_b64,
                "url": page.url,
            }
        except Exception as e:
            result = {"success": False, "error": str(e)}
        self._log_action("take_screenshot", {"full_page": full_page}, {"success": result.get("success", False)})
        return result

    async def get_action_log(self) -> Dict[str, Any]:
        return {
            "actions": self._action_log[-20:],
            "total_actions": len(self._action_log),
        }

    # ---- Multi-tab ----

    async def new_tab(self, url: Optional[str] = None) -> Dict[str, Any]:
        await self.initialize()
        if url:
            allowed, reason = is_domain_allowed(url)
            if not allowed:
                result = {"success": False, "error": reason}
                self._log_action("new_tab", {"url": url}, result)
                return result

        try:
            page = await self._context.new_page()
            tab_id = max(self._tabs.keys()) + 1
            self._tabs[tab_id] = page
            self._active_tab = tab_id

            if url:
                if not url.startswith(("http://", "https://")):
                    url = f"https://{url}"
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                # Validate final URL after potential redirects
                valid, block_reason = await self._validate_current_url(page)
                if not valid:
                    # Close the tab we just created since navigation was blocked
                    await page.close()
                    del self._tabs[tab_id]
                    self._active_tab = list(self._tabs.keys())[0] if self._tabs else 0
                    result = {"success": False, "error": block_reason}
                    self._log_action("new_tab", {"url": url}, result)
                    return result

            result = {"success": True, "tab_id": tab_id, "url": page.url if url else "about:blank"}
        except Exception as e:
            result = {"success": False, "error": str(e)}
        self._log_action("new_tab", {"url": url}, result)
        return result

    async def switch_tab(self, tab_id: int) -> Dict[str, Any]:
        if tab_id not in self._tabs:
            result = {"success": False, "error": f"Tab {tab_id} not found"}
            self._log_action("switch_tab", {"tab_id": tab_id}, result)
            return result
        self._active_tab = tab_id
        page = self._tabs[tab_id]
        await page.bring_to_front()
        result = {"success": True, "tab_id": tab_id, "url": page.url}
        self._log_action("switch_tab", {"tab_id": tab_id}, result)
        return result

    async def close_tab(self, tab_id: Optional[int] = None) -> Dict[str, Any]:
        tab_to_close = tab_id if tab_id is not None else self._active_tab
        if tab_to_close not in self._tabs:
            result = {"success": False, "error": f"Tab {tab_to_close} not found"}
            self._log_action("close_tab", {"tab_id": tab_to_close}, result)
            return result
        if len(self._tabs) <= 1:
            result = {"success": False, "error": "Cannot close last tab"}
            self._log_action("close_tab", {"tab_id": tab_to_close}, result)
            return result
        await self._tabs[tab_to_close].close()
        del self._tabs[tab_to_close]
        if self._active_tab == tab_to_close:
            self._active_tab = list(self._tabs.keys())[0]
        result = {"success": True, "closed_tab": tab_to_close, "active_tab": self._active_tab}
        self._log_action("close_tab", {"tab_id": tab_to_close}, result)
        return result

    async def list_tabs(self) -> Dict[str, Any]:
        tabs_info = []
        for tab_id, page in self._tabs.items():
            tabs_info.append({
                "tab_id": tab_id,
                "url": page.url,
                "title": await page.title(),
                "active": tab_id == self._active_tab,
            })
        result = {"tabs": tabs_info, "active_tab": self._active_tab}
        self._log_action("list_tabs", {}, {"success": True, "count": len(tabs_info)})
        return result


# ============ FastMCP Server (Official SDK) ============

# Load .env for domain allowlist config
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

mcp = FastMCP("browser-agent")

# Module-level browser server (lazy initialization on first tool call)
_server = MCPBrowserServer()


# ---- Navigation Tools ----

@mcp.tool()
async def browse_to(
    url: Annotated[str, Field(description="URL to navigate to")],
) -> str:
    """Navigate to a URL. Domain must be in allowlist (see MCP_ALLOWED_DOMAINS env var)."""
    result = await _server.browse_to(url)
    return json.dumps(result, indent=2)


@mcp.tool()
async def go_back() -> str:
    """Go back in browser history."""
    result = await _server.go_back()
    return json.dumps(result, indent=2)


@mcp.tool()
async def go_forward() -> str:
    """Go forward in browser history."""
    result = await _server.go_forward()
    return json.dumps(result, indent=2)


# ---- Waiting Tools (CRITICAL for dynamic content) ----

@mcp.tool()
async def wait_for(
    selector: Annotated[Optional[str], Field(description="CSS/Playwright selector to wait for")] = None,
    state: Annotated[Literal["visible", "hidden", "attached", "detached"], Field(description="Element state to wait for")] = "visible",
    timeout: Annotated[int, Field(description="Max wait time in milliseconds")] = 10000,
    wait_for_network: Annotated[bool, Field(description="If true, wait for network idle")] = False,
) -> str:
    """IMPORTANT: Wait for element or network before interacting. Use before click/fill on dynamic pages."""
    result = await _server.wait_for(selector, state, timeout, wait_for_network)
    return json.dumps(result, indent=2)


# ---- Interaction Tools ----

@mcp.tool()
async def click(
    selector: Annotated[str, Field(description="Playwright selector: 'text=Submit', '#id', '.class', 'button[name=x]'")],
    timeout: Annotated[int, Field(description="Timeout in milliseconds")] = 10000,
) -> str:
    """Click an element by Playwright selector."""
    result = await _server.click(selector, timeout)
    return json.dumps(result, indent=2)


@mcp.tool()
async def fill(
    selector: Annotated[str, Field(description="Input field selector")],
    value: Annotated[str, Field(description="Text to enter")],
    clear_first: Annotated[bool, Field(description="Clear field before filling")] = True,
) -> str:
    """Fill text into an input field."""
    result = await _server.fill(selector, value, clear_first)
    return json.dumps(result, indent=2)


@mcp.tool()
async def press_key(
    key: Annotated[str, Field(description="Key to press: Enter, Tab, Escape, ArrowDown, etc.")] = "Enter",
) -> str:
    """Press a keyboard key."""
    result = await _server.press_key(key)
    return json.dumps(result, indent=2)


@mcp.tool()
async def scroll(
    direction: Annotated[Literal["up", "down"], Field(description="Scroll direction")] = "down",
    amount: Annotated[int, Field(description="Pixels to scroll")] = 500,
) -> str:
    """Scroll the page up or down."""
    result = await _server.scroll(direction, amount)
    return json.dumps(result, indent=2)


@mcp.tool()
async def select_option(
    selector: Annotated[str, Field(description="Select element selector")],
    value: Annotated[str, Field(description="Option value to select")],
) -> str:
    """Select an option from a dropdown."""
    result = await _server.select_option(selector, value)
    return json.dumps(result, indent=2)


# ---- Extraction Tools ----

@mcp.tool()
async def get_page_info() -> str:
    """Get current page URL and title."""
    result = await _server.get_page_info()
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_dom_elements(
    max_elements: Annotated[int, Field(description="Max number of elements to return")] = 50,
) -> str:
    """Get interactive elements on the page (links, buttons, inputs). Use this to find selectors for click/fill."""
    result = await _server.get_dom_elements(max_elements)
    return json.dumps(result, indent=2)


@mcp.tool()
async def extract_text(
    selector: Annotated[str, Field(description="CSS selector to match elements")],
    max_items: Annotated[int, Field(description="Max number of text items")] = 20,
) -> str:
    """Extract text content from elements matching a CSS selector."""
    result = await _server.extract_text(selector, max_items)
    return json.dumps(result, indent=2)


@mcp.tool()
async def extract_table(
    selector: Annotated[str, Field(description="Table CSS selector")] = "table",
) -> str:
    """Extract table data as structured JSON with column headers as keys."""
    result = await _server.extract_table(selector)
    return json.dumps(result, indent=2)


# ---- Debugging Tools ----

@mcp.tool()
async def take_screenshot(
    full_page: Annotated[bool, Field(description="Capture full scrollable page")] = False,
) -> str:
    """Take a screenshot (returns base64 PNG). Use when a click fails to see what is on screen."""
    result = await _server.take_screenshot(full_page)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_action_log() -> str:
    """Get recent action log for debugging what happened."""
    result = await _server.get_action_log()
    return json.dumps(result, indent=2)


# ---- Multi-tab Tools ----

@mcp.tool()
async def new_tab(
    url: Annotated[Optional[str], Field(description="Optional URL to open in new tab")] = None,
) -> str:
    """Open a new browser tab, optionally navigating to a URL."""
    result = await _server.new_tab(url)
    return json.dumps(result, indent=2)


@mcp.tool()
async def switch_tab(
    tab_id: Annotated[int, Field(description="Tab ID to switch to")],
) -> str:
    """Switch to a different tab by ID."""
    result = await _server.switch_tab(tab_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def close_tab(
    tab_id: Annotated[Optional[int], Field(description="Tab ID to close (default: current tab)")] = None,
) -> str:
    """Close a tab. Defaults to closing the current active tab."""
    result = await _server.close_tab(tab_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_tabs() -> str:
    """List all open browser tabs with their URLs and titles."""
    result = await _server.list_tabs()
    return json.dumps(result, indent=2)


# ============ Entry Point ============

def main():
    """Run MCP server over stdio."""
    # Suppress print output that could corrupt JSON-RPC messages
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
