"""MCP (Model Context Protocol) server for Claude Desktop integration.

This allows Claude Desktop (using your Pro subscription) to orchestrate
the local browser agent. Cloud reasoning, local execution, $0 API tokens.

Usage:
  1. Add to Claude Desktop's claude_desktop_config.json:
     {
       "mcpServers": {
         "browser-agent": {
           "command": "python",
           "args": ["-m", "src.mcp_server"],
           "cwd": "/path/to/Local-Agent"
         }
       }
     }
  2. Restart Claude Desktop
  3. Ask Claude to browse websites, fill forms, extract data
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

from src.browser.tab_manager import TabManager
from src.agents.dom_actor import DOMActorAgent, DOMAction


class MCPServer:
    """
    MCP server exposing browser automation tools.

    Tools available:
    - browse_to: Navigate to a URL
    - get_page_info: Get current URL and title
    - get_dom_elements: Get interactive elements on page
    - click: Click an element by selector
    - fill: Fill an input field
    - press_key: Press a keyboard key
    - extract_text: Extract text from elements
    - scroll: Scroll the page
    """

    def __init__(self):
        self.tab_manager: Optional[TabManager] = None
        self.dom_actor: Optional[DOMActorAgent] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize browser and agents."""
        if self._initialized:
            return

        self.tab_manager = TabManager(headless=False)  # Show browser for user visibility
        await self.tab_manager.initialize()
        await self.tab_manager.create_tab(0)

        self.dom_actor = DOMActorAgent()
        self._initialized = True

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.tab_manager:
            await self.tab_manager.close()
        self._initialized = False

    def _get_page(self):
        """Get the active page."""
        if not self.tab_manager:
            raise RuntimeError("Browser not initialized")
        return self.tab_manager.get_page(0)

    # ============ MCP Tool Implementations ============

    async def browse_to(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL."""
        await self.initialize()
        await self.tab_manager.navigate(0, url)
        page = self._get_page()
        return {
            "success": True,
            "url": page.url,
            "title": await page.title(),
        }

    async def get_page_info(self) -> Dict[str, Any]:
        """Get current page URL and title."""
        await self.initialize()
        page = self._get_page()
        return {
            "url": page.url,
            "title": await page.title(),
        }

    async def get_dom_elements(self, max_elements: int = 50) -> Dict[str, Any]:
        """Get interactive elements on the current page."""
        await self.initialize()
        page = self._get_page()
        dom_snapshot = await self.dom_actor.get_dom_snapshot(page, max_length=15000)
        return {
            "url": page.url,
            "elements": dom_snapshot,
        }

    async def click(self, selector: str) -> Dict[str, Any]:
        """Click an element by Playwright selector."""
        await self.initialize()
        page = self._get_page()
        action = DOMAction(action_type="click", selector=selector)
        result = await self.dom_actor.execute_action(page, action)
        return {
            "success": result.get("success", False),
            "url": page.url,
            "error": result.get("error"),
        }

    async def fill(self, selector: str, value: str) -> Dict[str, Any]:
        """Fill an input field."""
        await self.initialize()
        page = self._get_page()
        action = DOMAction(action_type="fill", selector=selector, value=value)
        result = await self.dom_actor.execute_action(page, action)
        return {
            "success": result.get("success", False),
            "url": page.url,
            "error": result.get("error"),
        }

    async def press_key(self, key: str = "Enter") -> Dict[str, Any]:
        """Press a keyboard key (Enter, Tab, Escape, etc.)."""
        await self.initialize()
        page = self._get_page()
        await page.keyboard.press(key)
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass
        return {
            "success": True,
            "url": page.url,
        }

    async def extract_text(self, selector: str) -> Dict[str, Any]:
        """Extract text content from elements matching a selector."""
        await self.initialize()
        page = self._get_page()
        try:
            elements = await page.query_selector_all(selector)
            texts = []
            for el in elements[:20]:  # Limit to 20 elements
                text = await el.text_content()
                if text and text.strip():
                    texts.append(text.strip())
            return {
                "success": True,
                "count": len(texts),
                "texts": texts,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def scroll(self, direction: str = "down", amount: int = 500) -> Dict[str, Any]:
        """Scroll the page up or down."""
        await self.initialize()
        page = self._get_page()
        delta = amount if direction == "down" else -amount
        await page.evaluate(f"window.scrollBy(0, {delta})")
        return {"success": True, "direction": direction, "amount": amount}


# ============ MCP Protocol Handler ============

def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return MCP tool definitions."""
    return [
        {
            "name": "browse_to",
            "description": "Navigate the browser to a URL. Use this to open websites.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to (e.g., 'https://google.com')",
                    },
                },
                "required": ["url"],
            },
        },
        {
            "name": "get_page_info",
            "description": "Get the current page URL and title.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "get_dom_elements",
            "description": "Get a list of interactive elements (links, buttons, inputs) on the current page. Use this to understand what actions are available.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "max_elements": {
                        "type": "integer",
                        "description": "Maximum number of elements to return (default 50)",
                        "default": 50,
                    },
                },
            },
        },
        {
            "name": "click",
            "description": "Click an element on the page. Use Playwright selectors like 'text=Submit', '#button-id', 'button[name=search]', or 'role=button[name=Login]'.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Playwright selector for the element to click",
                    },
                },
                "required": ["selector"],
            },
        },
        {
            "name": "fill",
            "description": "Fill text into an input field. Use Playwright selectors like 'input[name=q]', '#search-box', or 'textarea'.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Playwright selector for the input field",
                    },
                    "value": {
                        "type": "string",
                        "description": "The text to type into the field",
                    },
                },
                "required": ["selector", "value"],
            },
        },
        {
            "name": "press_key",
            "description": "Press a keyboard key. Common keys: Enter, Tab, Escape, ArrowDown, ArrowUp.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to press (default: Enter)",
                        "default": "Enter",
                    },
                },
            },
        },
        {
            "name": "extract_text",
            "description": "Extract text content from elements matching a selector. Useful for scraping data from a page.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for elements to extract text from (e.g., 'h1', '.price', 'table tr')",
                    },
                },
                "required": ["selector"],
            },
        },
        {
            "name": "scroll",
            "description": "Scroll the page up or down to reveal more content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Direction to scroll",
                        "default": "down",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Pixels to scroll (default 500)",
                        "default": 500,
                    },
                },
            },
        },
    ]


async def handle_request(server: MCPServer, request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle an MCP JSON-RPC request."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                    },
                    "serverInfo": {
                        "name": "browser-agent",
                        "version": "1.0.0",
                    },
                },
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": get_tool_definitions(),
                },
            }

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})

            # Dispatch to tool implementation
            if tool_name == "browse_to":
                result = await server.browse_to(tool_args.get("url", ""))
            elif tool_name == "get_page_info":
                result = await server.get_page_info()
            elif tool_name == "get_dom_elements":
                result = await server.get_dom_elements(tool_args.get("max_elements", 50))
            elif tool_name == "click":
                result = await server.click(tool_args.get("selector", ""))
            elif tool_name == "fill":
                result = await server.fill(
                    tool_args.get("selector", ""),
                    tool_args.get("value", ""),
                )
            elif tool_name == "press_key":
                result = await server.press_key(tool_args.get("key", "Enter"))
            elif tool_name == "extract_text":
                result = await server.extract_text(tool_args.get("selector", ""))
            elif tool_name == "scroll":
                result = await server.scroll(
                    tool_args.get("direction", "down"),
                    tool_args.get("amount", 500),
                )
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}",
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2),
                        }
                    ],
                },
            }

        elif method == "notifications/initialized":
            # Client notification, no response needed
            return None

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }

    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32000,
                "message": str(e),
            },
        }


async def run_mcp_server():
    """Run the MCP server over stdio."""
    server = MCPServer()

    # Read from stdin, write to stdout (MCP protocol)
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            if not line:
                break

            request = json.loads(line)
            response = await handle_request(server, request)

            if response:  # Some notifications don't need responses
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

        except json.JSONDecodeError:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": str(e)},
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()

    await server.shutdown()


def main():
    """Entry point for MCP server."""
    asyncio.run(run_mcp_server())


if __name__ == "__main__":
    main()
