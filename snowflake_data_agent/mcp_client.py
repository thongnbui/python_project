"""MCP client for talking to a Snowflake MCP server from synchronous code.

The official MCP Python SDK is fully asynchronous, but Streamlit runs our code
synchronously and re-executes the script on every interaction. To bridge the two
worlds we host a single, long-lived asyncio event loop on a background thread and
keep one MCP session alive inside a single task for the whole app session.

Why a single task? ``anyio`` cancel scopes (used internally by ``stdio_client``
and ``ClientSession``) must be entered and exited in the *same* task. If we
opened the session in one coroutine and called tools from other coroutines we
would hit "Attempted to exit cancel scope in a different task" errors. Instead a
dedicated runner task owns the session and we feed it work over a queue.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class ToolResult:
    """Normalized result of an MCP tool call."""

    text: str
    is_error: bool = False
    structured: Any = None  # Parsed JSON payload when the text is valid JSON.

    @property
    def rows(self) -> Optional[list[dict]]:
        """Return the result as a list of row dicts when it looks tabular."""
        if isinstance(self.structured, list) and all(
            isinstance(item, dict) for item in self.structured
        ):
            return self.structured
        return None


@dataclass
class MCPTool:
    """Lightweight description of a tool exposed by the MCP server."""

    name: str
    description: str
    input_schema: dict = field(default_factory=dict)

    def to_openai_tool(self) -> dict:
        """Convert to the schema expected by OpenAI function calling."""
        parameters = self.input_schema or {"type": "object", "properties": {}}
        # OpenAI requires an object schema with a ``properties`` key.
        if parameters.get("type") != "object":
            parameters = {"type": "object", "properties": {}}
        parameters.setdefault("properties", {})
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description[:1024],
                "parameters": parameters,
            },
        }


class SnowflakeMCPClient:
    """Synchronous facade over an async MCP session running on a worker thread."""

    def __init__(self, server_params: StdioServerParameters, startup_timeout: float = 180.0):
        self._server_params = server_params
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._cmd_queue: Optional[asyncio.Queue] = None
        self._ready_evt: Optional[asyncio.Event] = None
        self._runner_task: Optional[asyncio.Task] = None
        self._startup_error: Optional[BaseException] = None
        self.tools: list[MCPTool] = []

        future = asyncio.run_coroutine_threadsafe(self._start(), self._loop)
        future.result(timeout=startup_timeout)
        if self._startup_error is not None:
            self.close()
            raise self._startup_error

    async def _start(self) -> None:
        """Create the queue and launch the session runner, then wait until ready."""
        self._cmd_queue = asyncio.Queue()
        self._ready_evt = asyncio.Event()
        self._runner_task = asyncio.create_task(self._session_runner())
        await self._ready_evt.wait()

    async def _session_runner(self) -> None:
        """Own the MCP session for its entire lifetime inside one task."""
        try:
            async with stdio_client(self._server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self.tools = [
                        MCPTool(
                            name=tool.name,
                            description=tool.description or "",
                            input_schema=tool.inputSchema or {},
                        )
                        for tool in listed.tools
                    ]
                    self._ready_evt.set()
                    await self._serve(session)
        except BaseException as exc:  # noqa: BLE001 - surface startup failures to caller
            self._startup_error = exc
            if self._ready_evt is not None:
                self._ready_evt.set()

    async def _serve(self, session: ClientSession) -> None:
        """Process tool-call requests until a shutdown sentinel arrives."""
        assert self._cmd_queue is not None
        while True:
            item = await self._cmd_queue.get()
            if item is None:
                break
            name, arguments, future = item
            try:
                result = await session.call_tool(name, arguments)
                future.set_result(_normalize_result(result))
            except BaseException as exc:  # noqa: BLE001 - forward to caller
                future.set_exception(exc)

    def call_tool(self, name: str, arguments: dict, timeout: float = 120.0) -> ToolResult:
        """Invoke an MCP tool and block until it returns."""
        if self._cmd_queue is None:
            raise RuntimeError("MCP client is not connected.")
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(
            self._cmd_queue.put_nowait, (name, arguments, future)
        )
        return future.result(timeout=timeout)

    def openai_tools(self) -> list[dict]:
        """Return all tools formatted for OpenAI function calling."""
        return [tool.to_openai_tool() for tool in self.tools]

    def close(self) -> None:
        """Tear down the session task and stop the event loop."""
        try:
            if self._cmd_queue is not None:
                self._loop.call_soon_threadsafe(self._cmd_queue.put_nowait, None)
        except Exception:  # noqa: BLE001 - best-effort shutdown
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


def _normalize_result(result: Any) -> ToolResult:
    """Flatten an MCP ``CallToolResult`` into text + parsed structure."""
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(str(item))
    text = "\n".join(parts).strip()

    structured: Any = None
    try:
        structured = json.loads(text)
    except (ValueError, TypeError):
        structured = None

    return ToolResult(
        text=text,
        is_error=bool(getattr(result, "isError", False)),
        structured=structured,
    )
