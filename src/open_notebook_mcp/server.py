from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import uuid
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from typing import Any, Literal, Optional

import httpx
from mcp.server.mcpserver import MCPServer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("open-notebook-mcp")

mcp = MCPServer("open-notebook-mcp", version=_pkg_version("open-notebook-mcp"))

# -----------------------------
# Configuration and Constants
# -----------------------------

# Maximum number of items to return in list operations.
# AGENTS.md 4.1 caps list/search tools at 200 items per page.
MAX_LIMIT = 200

# Ceiling the discovery endpoint accepts for one run (api/models.py
# DiscoverSourcesRequest.limit). The provider never returns more candidates
# than this, so a page of discovered hits cannot grow beyond it.
DISCOVER_LIMIT_MAX = 20

# Ceiling the discovery endpoint accepts for the excluded-domain list
# (api/models.py DiscoverSourcesRequest.exclude_domains).
DISCOVER_EXCLUDE_DOMAINS_MAX = 50

# Ceiling the sources endpoint accepts for one page (api/routers/sources.py
# get_sources). Asking for more makes the upstream answer 422.
SOURCES_LIMIT_MAX = 100


def _read_timeout_s(raw: str) -> float:
    """Return the HTTP timeout in seconds, refusing values httpx accepts silently.

    httpx takes zero, negative, infinite and NaN timeouts at face value, so a
    typo in the environment turned into requests that either hang forever or
    fail on the first byte. Invalid input raises instead of degrading quietly.

    Args:
        raw: Raw value of OPEN_NOTEBOOK_TIMEOUT_S, or the default when unset.

    Returns:
        The timeout in seconds, always positive and finite.

    Raises:
        ValueError: If raw is not a positive, finite number.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"OPEN_NOTEBOOK_TIMEOUT_S must be a number, got {raw!r}"
        ) from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"OPEN_NOTEBOOK_TIMEOUT_S must be a positive finite number, got {raw!r}"
        )
    return value


# Default timeout for HTTP requests (seconds)
# PATCHED-TIMEOUT-V1: 30s aborted slow RAG asks (40-50s); an httpx timeout
# carries an EMPTY message, so callers only saw "API request failed: ".
DEFAULT_TIMEOUT_S = _read_timeout_s(os.getenv("OPEN_NOTEBOOK_TIMEOUT_S", "300"))

def get_base_url() -> str:
    """Get the Open Notebook API base URL from environment."""
    return os.getenv("OPEN_NOTEBOOK_URL", "http://localhost:5055")

def get_auth_token() -> Optional[str]:
    """Get the authentication token from environment."""
    return os.getenv("OPEN_NOTEBOOK_PASSWORD")

def generate_request_id() -> str:
    """Generate a unique request ID for tracking."""
    return str(uuid.uuid4())


class ToolError(Exception):
    """Structured tool error with the fields AGENTS.md section 6 requires.

    Args:
        error_code: Stable machine-readable code, for example 'invalid_input'.
        message: What went wrong, in one sentence.
        hint: Actionable next step for the caller.
        request_id: Correlation ID for the server logs.
    """

    def __init__(
        self,
        error_code: str,
        message: str,
        hint: str,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.hint = hint
        self.request_id = request_id or generate_request_id()

    def to_dict(self) -> dict[str, str]:
        """Return the error as the structured payload of AGENTS.md section 6.

        Returns:
            The error code, message, hint and request ID.
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "hint": self.hint,
            "request_id": self.request_id,
        }

    def __str__(self) -> str:
        return (
            f"{self.error_code}: {self.message} "
            f"(hint: {self.hint}, request_id: {self.request_id})"
        )


class ToolInputError(ToolError, ValueError):
    """Input rejected before the request left the process."""


class ToolApiError(ToolError, RuntimeError):
    """The Open Notebook API refused or failed the request."""


class ToolTimeoutError(ToolError, TimeoutError):
    """The request exceeded its deadline."""


def _require_text(
    value: Optional[str],
    field: str,
    hint: str,
    max_length: Optional[int] = None,
) -> str:
    """Return a trimmed non-empty value, or raise a structured input error.

    Args:
        value: The value to check.
        field: Name of the field, used in the error message.
        hint: Actionable next step for the caller.
        max_length: Optional maximum number of characters accepted.

    Returns:
        The trimmed value.

    Raises:
        ToolInputError: If the value is missing, blank, not a string or too long.
    """
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(
            "invalid_input",
            f"{field} is required and must be a non-empty string",
            hint,
        )
    trimmed = value.strip()
    if max_length is not None and len(trimmed) > max_length:
        raise ToolInputError(
            "invalid_input",
            f"{field} must be at most {max_length} characters, got {len(trimmed)}",
            hint,
        )
    return trimmed


def _require_range(
    value: int, field: str, minimum: int, maximum: int, hint: str
) -> int:
    """Return the value when it is inside the documented range.

    Args:
        value: The value to check.
        field: Name of the field, used in the error message.
        minimum: Lowest accepted value.
        maximum: Highest accepted value.
        hint: Actionable next step for the caller.

    Returns:
        The value, unchanged.

    Raises:
        ToolInputError: If the value is not an integer or is out of range.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolInputError(
            "invalid_input",
            f"{field} must be an integer",
            hint,
        )
    if not minimum <= value <= maximum:
        raise ToolInputError(
            "invalid_input",
            f"{field} must be between {minimum} and {maximum}, got {value}",
            hint,
        )
    return value


# -----------------------------
# Capability index (source of truth)
# -----------------------------
# Note: CAPABILITIES is a tuple to ensure immutability and prevent accidental modification.
# This is the single source of truth for all available tools in the server.

Detail = Literal["name", "summary", "full"]

#: Detail levels accepted by every list/search tool.
DETAIL_LEVELS: tuple[str, ...] = ("name", "summary", "full")


def _require_detail(detail: str, levels: tuple[str, ...]) -> str:
    """Return the detail level when the tool publishes it.

    Args:
        detail: Requested detail level.
        levels: Detail levels the tool accepts.

    Returns:
        The detail level, unchanged.

    Raises:
        ToolInputError: If the level is not one the tool documents.
    """
    if detail not in levels:
        raise ToolInputError(
            "invalid_input",
            f"detail must be one of {', '.join(levels)}, got {detail!r}",
            "Use 'summary' for the default row, 'name' for the smallest one.",
        )
    return detail


def _require_fields(fields: Optional[list[str]]) -> Optional[list[str]]:
    """Return the requested row keys after trimming them.

    Args:
        fields: Row keys the caller wants to keep, or None.

    Returns:
        The trimmed keys, or None when nothing was asked for.

    Raises:
        ToolInputError: If a key is blank or not a string.
    """
    if fields is None:
        return None
    return [
        _require_text(
            field,
            "fields",
            "Pass field names as strings, for example ['id', 'title'].",
        )
        for field in fields
    ]


def _select_fields(
    record: Any,
    fields: Optional[list[str]],
    detail: str,
    presets: dict[str, tuple[str, ...]],
) -> Any:
    """Project one API row down to the keys the caller asked for.

    Args:
        record: One row returned by the API.
        fields: Explicit keys to keep, or None to use the detail preset.
        detail: Detail preset name.
        presets: Key sets per detail level; a missing level keeps the row whole.

    Returns:
        The projected row, or the row untouched when nothing was asked for.
    """
    if not isinstance(record, dict):
        return record
    keys = fields if fields else presets.get(detail)
    if not keys:
        return record
    return {key: record[key] for key in keys if key in record}


@dataclass(frozen=True)
class Capability:
    name: str                 # tool name
    summary: str              # one-liner
    tags: tuple[str, ...]     # searchable tags
    args: dict[str, str]      # param -> type (stringified)
    returns: str              # return type (stringified)
    example: dict[str, Any]   # example call args
    typical_bytes: int        # typical response size (rough)

CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        name="search_capabilities",
        summary="Search tools exposed by this server with progressive detail levels.",
        tags=("meta", "discovery", "progressive-disclosure"),
        args={"query": "str", "detail": "Literal['name','summary','full']", "limit": "int"},
        returns="dict[str, Any]",
        example={"query": "notebook", "detail": "summary", "limit": 10},
        typical_bytes=1200,
    ),
    # Notebooks API
    Capability(
        name="list_notebooks",
        summary="Get all notebooks with optional filtering and ordering.",
        tags=("notebooks", "list", "query"),
        args={"archived": "Optional[bool]", "order_by": "str", "limit": "int"},
        returns="dict[str, Any]",
        example={"archived": False, "order_by": "updated desc", "limit": 20},
        typical_bytes=2000,
    ),
    Capability(
        name="get_notebook",
        summary="Get a specific notebook by ID.",
        tags=("notebooks", "get", "read"),
        args={"notebook_id": "str"},
        returns="dict[str, Any]",
        example={"notebook_id": "notebook:abc123"},
        typical_bytes=500,
    ),
    Capability(
        name="create_notebook",
        summary="Create a new notebook.",
        tags=("notebooks", "create", "write"),
        args={"name": "str", "description": "Optional[str]"},
        returns="dict[str, Any]",
        example={"name": "My Research", "description": "AI research notebook"},
        typical_bytes=500,
    ),
    Capability(
        name="update_notebook",
        summary="Update a notebook.",
        tags=("notebooks", "update", "write"),
        args={"notebook_id": "str", "name": "Optional[str]", "description": "Optional[str]", "archived": "Optional[bool]"},
        returns="dict[str, Any]",
        example={"notebook_id": "notebook:abc123", "name": "Updated Name"},
        typical_bytes=500,
    ),
    Capability(
        name="delete_notebook",
        summary="Delete a notebook.",
        tags=("notebooks", "delete", "write"),
        args={"notebook_id": "str"},
        returns="dict[str, Any]",
        example={"notebook_id": "notebook:abc123"},
        typical_bytes=100,
    ),
    # Sources API
    Capability(
        name="list_sources",
        summary="Get sources with optional filtering, paged.",
        tags=("sources", "list", "query"),
        args={
            "notebook_id": "Optional[str]",
            "limit": "int",
            "offset": "int",
            "fields": "Optional[list[str]]",
            "detail": "Literal['name','summary','full']",
        },
        returns="dict[str, Any]",
        example={"notebook_id": "notebook:abc123", "limit": 25, "detail": "summary"},
        typical_bytes=3000,
    ),
    Capability(
        name="get_source",
        summary="Get a specific source by ID.",
        tags=("sources", "get", "read"),
        args={"source_id": "str"},
        returns="dict[str, Any]",
        example={"source_id": "source:abc123"},
        typical_bytes=2000,
    ),
    Capability(
        name="create_source",
        summary="Create a link or text source asynchronously.",
        tags=("sources", "create", "write"),
        args={
            "notebook_id": "str",
            "type": "Literal['link','text']",
            "url": "Optional[str]",
            "content": "Optional[str]",
            "title": "Optional[str]",
            "embed": "bool",
            "async_processing": "bool",
        },
        returns="dict[str, Any]",
        example={
            "notebook_id": "notebook:abc123",
            "type": "text",
            "content": "Research notes",
            "embed": True,
            "async_processing": True,
        },
        typical_bytes=2000,
    ),
    Capability(
        name="update_source",
        summary="Update a source.",
        tags=("sources", "update", "write"),
        args={"source_id": "str", "title": "Optional[str]", "topics": "Optional[list[str]]"},
        returns="dict[str, Any]",
        example={"source_id": "source:abc123", "title": "New Title"},
        typical_bytes=2000,
    ),
    Capability(
        name="delete_source",
        summary="Delete a source.",
        tags=("sources", "delete", "write"),
        args={"source_id": "str"},
        returns="dict[str, Any]",
        example={"source_id": "source:abc123"},
        typical_bytes=100,
    ),
    # Notes API
    Capability(
        name="list_notes",
        summary="Get all notes with optional filtering.",
        tags=("notes", "list", "query"),
        args={"notebook_id": "Optional[str]", "limit": "int", "offset": "int"},
        returns="dict[str, Any]",
        example={"notebook_id": "notebook:abc123", "limit": 20, "offset": 0},
        typical_bytes=2000,
    ),
    Capability(
        name="get_note",
        summary="Get a specific note by ID.",
        tags=("notes", "get", "read"),
        args={"note_id": "str"},
        returns="dict[str, Any]",
        example={"note_id": "note:abc123"},
        typical_bytes=1500,
    ),
    Capability(
        name="create_note",
        summary="Create a new note.",
        tags=("notes", "create", "write"),
        args={"notebook_id": "str", "title": "str", "content": "str", "topics": "Optional[list[str]]"},
        returns="dict[str, Any]",
        example={"notebook_id": "notebook:abc123", "title": "My Note", "content": "Note content"},
        typical_bytes=1500,
    ),
    Capability(
        name="update_note",
        summary="Update a note.",
        tags=("notes", "update", "write"),
        args={"note_id": "str", "title": "Optional[str]", "content": "Optional[str]", "topics": "Optional[list[str]]"},
        returns="dict[str, Any]",
        example={"note_id": "note:abc123", "title": "Updated Title"},
        typical_bytes=1500,
    ),
    Capability(
        name="delete_note",
        summary="Delete a note.",
        tags=("notes", "delete", "write"),
        args={"note_id": "str"},
        returns="dict[str, Any]",
        example={"note_id": "note:abc123"},
        typical_bytes=100,
    ),
    # Search API
    Capability(
        name="search",
        summary="Search content using vector or text search.",
        tags=("search", "query", "vector"),
        args={"query": "str", "type": "str", "notebook_id": "Optional[str]", "limit": "int"},
        returns="dict[str, Any]",
        example={"query": "AI research", "type": "vector", "limit": 10},
        typical_bytes=3000,
    ),
    Capability(
        name="ask_question",
        summary="Ask a question about your content with detailed control.",
        tags=("search", "ask", "ai", "question"),
        args={"question": "str", "strategy_model": "Optional[str]", "answer_model": "Optional[str]", "final_answer_model": "Optional[str]"},
        returns="dict[str, Any]",
        example={"question": "What are the main AI applications?", "strategy_model": "model:abc", "answer_model": "model:abc", "final_answer_model": "model:abc"},
        typical_bytes=5000,
    ),
    Capability(
        name="ask_simple",
        summary="Ask a question about your content with simplified interface.",
        tags=("search", "ask", "ai", "question", "simple"),
        args={"question": "str", "strategy_model": "Optional[str]", "answer_model": "Optional[str]", "final_answer_model": "Optional[str]"},
        returns="dict[str, Any]",
        example={"question": "Summarize my AI research", "strategy_model": "model:abc", "answer_model": "model:abc", "final_answer_model": "model:abc"},
        typical_bytes=4000,
    ),
    # Models API
    Capability(
        name="list_models",
        summary="Get all configured AI models.",
        tags=("models", "list", "ai"),
        args={"limit": "int"},
        returns="dict[str, Any]",
        example={"limit": 50},
        typical_bytes=2000,
    ),
    Capability(
        name="get_model",
        summary="Get a specific model by ID.",
        tags=("models", "get", "read", "ai"),
        args={"model_id": "str"},
        returns="dict[str, Any]",
        example={"model_id": "model:abc123"},
        typical_bytes=500,
    ),
    Capability(
        name="create_model",
        summary="Create a new AI model configuration.",
        tags=("models", "create", "write", "ai"),
        args={"name": "str", "provider": "str", "type": "str"},
        returns="dict[str, Any]",
        example={"name": "gpt-4", "provider": "openai", "type": "language"},
        typical_bytes=500,
    ),
    Capability(
        name="delete_model",
        summary="Delete a model configuration.",
        tags=("models", "delete", "write", "ai"),
        args={"model_id": "str"},
        returns="dict[str, Any]",
        example={"model_id": "model:abc123"},
        typical_bytes=100,
    ),
    Capability(
        name="get_default_models",
        summary="Get default model configurations.",
        tags=("models", "defaults", "ai"),
        args={},
        returns="dict[str, Any]",
        example={},
        typical_bytes=1000,
    ),
    # Chat API
    Capability(
        name="list_chat_sessions",
        summary="Get all chat sessions with optional filtering.",
        tags=("chat", "sessions", "list"),
        args={"notebook_id": "Optional[str]", "limit": "int"},
        returns="dict[str, Any]",
        example={"notebook_id": "notebook:abc123", "limit": 20},
        typical_bytes=2000,
    ),
    Capability(
        name="create_chat_session",
        summary="Create a new chat session.",
        tags=("chat", "sessions", "create", "write"),
        args={"notebook_id": "str", "title": "str"},
        returns="dict[str, Any]",
        example={"notebook_id": "notebook:abc123", "title": "Research Discussion"},
        typical_bytes=500,
    ),
    Capability(
        name="get_chat_session",
        summary="Get a specific chat session by ID.",
        tags=("chat", "sessions", "get", "read"),
        args={"session_id": "str"},
        returns="dict[str, Any]",
        example={"session_id": "session:abc123"},
        typical_bytes=3000,
    ),
    Capability(
        name="update_chat_session",
        summary="Update a chat session.",
        tags=("chat", "sessions", "update", "write"),
        args={"session_id": "str", "title": "Optional[str]"},
        returns="dict[str, Any]",
        example={"session_id": "session:abc123", "title": "Updated Title"},
        typical_bytes=500,
    ),
    Capability(
        name="delete_chat_session",
        summary="Delete a chat session.",
        tags=("chat", "sessions", "delete", "write"),
        args={"session_id": "str"},
        returns="dict[str, Any]",
        example={"session_id": "session:abc123"},
        typical_bytes=100,
    ),
    Capability(
        name="execute_chat",
        summary="Send a message in a chat session.",
        tags=("chat", "execute", "message", "ai"),
        args={"session_id": "str", "message": "str", "context": "Optional[dict]"},
        returns="dict[str, Any]",
        example={"session_id": "session:abc123", "message": "What are the key insights?"},
        typical_bytes=3000,
    ),
    Capability(
        name="get_chat_context",
        summary="Build context for a chat conversation.",
        tags=("chat", "context", "build"),
        args={"notebook_id": "str", "context_config": "Optional[dict]"},
        returns="dict[str, Any]",
        example={"notebook_id": "notebook:abc123"},
        typical_bytes=5000,
    ),
    # Settings API
    Capability(
        name="get_settings",
        summary="Get application settings.",
        tags=("settings", "get", "config"),
        args={},
        returns="dict[str, Any]",
        example={},
        typical_bytes=1000,
    ),
    Capability(
        name="update_settings",
        summary="Update application settings.",
        tags=("settings", "update", "write", "config"),
        args={"settings": "dict"},
        returns="dict[str, Any]",
        example={"settings": {"theme": "dark"}},
        typical_bytes=1000,
    ),
    # Web discovery API (upstream #973) - requires an instance that serves it
    Capability(
        name="discover_sources",
        summary="Search the web and attach the hits to a notebook as sources.",
        tags=("discover", "web", "search", "sources", "create", "native"),
        args={
            "query": "str",
            "notebook_id": "str",
            "limit": "int",
            "provider": "Optional[str]",
            "embed": "bool",
            "exclude_domains": "Optional[list[str]]",
            "dry_run": "bool",
            "fields": "Optional[list[str]]",
            "detail": "Literal['name','summary','full']",
        },
        returns="dict[str, Any]",
        example={
            "query": "DGX Spark local LLM inference performance",
            "notebook_id": "notebook:abc123",
            "limit": 5,
            "dry_run": True,
            "detail": "summary",
        },
        typical_bytes=4000,
    ),
    # Artifacts API (upstream #203) - requires an instance that serves it
    Capability(
        name="generate_artifact",
        summary="Generate a report or deck from a notebook (a deck adds diagrams).",
        tags=("artifacts", "report", "deck", "diagrams", "mermaid", "generate", "native"),
        args={
            "notebook_id": "str",
            "kind": "str",
            "variant": "Optional[str]",
            "formats": "Optional[list[str]]",
            "language": "str",
            "title": "Optional[str]",
            "instructions": "Optional[str]",
            "sections": "int",
            "model_id": "Optional[str]",
        },
        returns="dict[str, Any]",
        example={
            "notebook_id": "notebook:abc123",
            "kind": "deck",
            "formats": ["md", "docx", "pptx"],
            "language": "Italian",
        },
        typical_bytes=600,
    ),
    Capability(
        name="list_artifacts",
        summary="List generated artifacts, newest first.",
        tags=("artifacts", "list", "query", "native"),
        args={
            "notebook_id": "Optional[str]",
            "limit": "int",
            "offset": "int",
            "fields": "Optional[list[str]]",
            "detail": "Literal['name','summary','full']",
        },
        returns="dict[str, Any]",
        example={"notebook_id": "notebook:abc123", "limit": 25, "detail": "summary"},
        typical_bytes=3000,
    ),
    Capability(
        name="get_artifact",
        summary="Get one artifact with the status of its generation job.",
        tags=("artifacts", "get", "read", "poll", "native"),
        args={"artifact_id": "str"},
        returns="dict[str, Any]",
        example={"artifact_id": "artifact:abc123"},
        typical_bytes=1500,
    ),
    Capability(
        name="delete_artifact",
        summary="Delete an artifact and its rendered files.",
        tags=("artifacts", "delete", "write", "native"),
        args={"artifact_id": "str"},
        returns="dict[str, Any]",
        example={"artifact_id": "artifact:abc123"},
        typical_bytes=100,
    ),
)

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def _match_score(q: str, cap: Capability) -> int:
    if not q:
        return 0
    qn = _normalize(q)
    hay = " ".join([cap.name, cap.summary, " ".join(cap.tags)])
    hn = _normalize(hay)
    score = 0
    for token in qn.split():
        if token in hn:
            score += 1
    return score

@mcp.tool()
def search_capabilities(query: str = "", detail: Detail = "summary", limit: int = 20) -> dict[str, Any]:
    """Search tools exposed by this server with progressive detail levels.
    
    Args:
        query: Search query to filter tools
        detail: Level of detail - 'name' (minimal), 'summary' (default), or 'full' (complete)
        limit: Maximum number of results (1-50)
    
    Returns:
        Dictionary with request_id, query, detail, count, matches, and hint fields
    """
    request_id = generate_request_id()
    limit = max(1, min(int(limit), 50))

    scored = [(c, _match_score(query, c)) for c in CAPABILITIES]
    if query.strip():
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda t: (-t[1], t[0].name))
    else:
        scored.sort(key=lambda t: t[0].name)

    matches = []
    for cap, _score in scored[:limit]:
        if detail == "name":
            matches.append({"name": cap.name})
        elif detail == "summary":
            matches.append({"name": cap.name, "summary": cap.summary, "tags": list(cap.tags)})
        else:
            matches.append({
                "name": cap.name,
                "summary": cap.summary,
                "tags": list(cap.tags),
                "args": cap.args,
                "returns": cap.returns,
                "example": cap.example,
                "typical_bytes": cap.typical_bytes,
            })

    return {
        "request_id": request_id,
        "query": query,
        "detail": detail,
        "count": len(matches),
        "matches": matches,
        "hint": "Use detail='name' for minimal context; detail='full' only when implementing a call.",
    }

# -----------------------------
# HTTP helper functions
# -----------------------------
# HTTP helper functions
# -----------------------------

async def make_request(
    method: str,
    endpoint: str,
    *,
    json_data: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Make an HTTP request to the Open Notebook API.
    
    Args:
        method: HTTP method (GET, POST, PUT, DELETE)
        endpoint: API endpoint path (e.g., '/api/notebooks')
        json_data: Optional JSON body for POST/PUT requests
        params: Optional query parameters
    
    Returns:
        Response JSON as dictionary
    
    Raises:
        Exception: If request fails with formatted error message
    """
    base_url = get_base_url()
    url = f"{base_url}{endpoint}"
    
    headers = {"Content-Type": "application/json"}
    auth_token = get_auth_token()
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    async with httpx.AsyncClient(follow_redirects=True, timeout=DEFAULT_TIMEOUT_S) as client:
        try:
            if method == "GET":
                r = await client.get(url, headers=headers, params=params)
            elif method == "POST":
                r = await client.post(url, headers=headers, json=json_data, params=params)
            elif method == "PUT":
                r = await client.put(url, headers=headers, json=json_data, params=params)
            elif method == "DELETE":
                r = await client.delete(url, headers=headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            r.raise_for_status()
            
            # Handle empty responses
            if not r.text:
                return {"message": "Success"}
            
            try:
                return r.json()
            except (ValueError, Exception) as json_err:
                # If response is not valid JSON, return text content
                log.warning(f"Non-JSON response from {endpoint}: {json_err}")
                return {"message": "Success", "content": r.text}
            
        except httpx.HTTPError as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg = error_detail.get("detail", error_msg)
                except (ValueError, AttributeError):
                    # If error response is not JSON, use text or default message
                    error_msg = e.response.text or error_msg

            raise ToolApiError(
                "api_error",
                f"API request failed: {error_msg}",
                "Check that OPEN_NOTEBOOK_URL points at a reachable Open Notebook API "
                "and that the payload matches the endpoint contract.",
            ) from e

# -----------------------------
# Notebooks API Tools
# -----------------------------

@mcp.tool()
async def list_notebooks(
    archived: Optional[bool] = None,
    order_by: str = "updated desc",
    limit: int = 20,
) -> dict[str, Any]:
    """Get all notebooks with optional filtering and ordering.
    
    Args:
        archived: Filter by archived status (None = all, True = archived only, False = active only)
        order_by: Order by field and direction (e.g., 'created desc', 'name asc')
        limit: Maximum number of results (1-200)
    
    Returns:
        Dictionary with notebooks list and metadata
    """
    limit = max(1, min(limit, MAX_LIMIT))
    params = {"order_by": order_by}
    if archived is not None:
        params["archived"] = archived
    
    notebooks = await make_request("GET", "/api/notebooks", params=params)
    
    # Limit results
    if isinstance(notebooks, list):
        notebooks = notebooks[:limit]
    
    return {
        "request_id": generate_request_id(),
        "count": len(notebooks) if isinstance(notebooks, list) else 0,
        "notebooks": notebooks,
    }

@mcp.tool()
async def get_notebook(notebook_id: str) -> dict[str, Any]:
    """Get a specific notebook by ID.
    
    Args:
        notebook_id: Notebook ID (e.g., 'notebook:abc123')
    
    Returns:
        Notebook details
    """
    notebook = await make_request("GET", f"/api/notebooks/{notebook_id}")
    return {
        "request_id": generate_request_id(),
        "notebook": notebook,
    }

@mcp.tool()
async def create_notebook(name: str, description: Optional[str] = None) -> dict[str, Any]:
    """Create a new notebook.
    
    Args:
        name: Notebook name
        description: Optional notebook description
    
    Returns:
        Created notebook details
    """
    data = {"name": name}
    if description is not None:
        data["description"] = description
    
    notebook = await make_request("POST", "/api/notebooks", json_data=data)
    return {
        "request_id": generate_request_id(),
        "notebook": notebook,
    }

@mcp.tool()
async def update_notebook(
    notebook_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    archived: Optional[bool] = None,
) -> dict[str, Any]:
    """Update a notebook.
    
    Args:
        notebook_id: Notebook ID
        name: Optional new name
        description: Optional new description
        archived: Optional archived status
    
    Returns:
        Updated notebook details
    """
    data = {}
    if name is not None:
        data["name"] = name
    if description is not None:
        data["description"] = description
    if archived is not None:
        data["archived"] = archived
    
    notebook = await make_request("PUT", f"/api/notebooks/{notebook_id}", json_data=data)
    return {
        "request_id": generate_request_id(),
        "notebook": notebook,
    }

@mcp.tool()
async def delete_notebook(notebook_id: str) -> dict[str, Any]:
    """Delete a notebook.
    
    Args:
        notebook_id: Notebook ID
    
    Returns:
        Success message
    """
    result = await make_request("DELETE", f"/api/notebooks/{notebook_id}")
    return {
        "request_id": generate_request_id(),
        "result": result,
    }

# -----------------------------
# Sources API Tools
# -----------------------------

#: Detail presets for the source rows (api/models.py SourceListResponse).
#: 'full' is absent on purpose: it keeps the whole row.
SOURCE_DETAIL_FIELDS: dict[str, tuple[str, ...]] = {
    "name": ("id", "title"),
    "summary": (
        "id",
        "title",
        "topics",
        "embedded",
        "insights_count",
        "status",
        "updated",
    ),
}


@mcp.tool()
async def list_sources(
    notebook_id: Optional[str] = None,
    limit: int = 25,
    offset: int = 0,
    fields: Optional[list[str]] = None,
    detail: Detail = "summary",
) -> dict[str, Any]:
    """Get sources with optional filtering, paged.

    Args:
        notebook_id: Optional notebook ID to filter by.
        limit: Maximum number of results (1-100, default 25). The API refuses
            more than 100 per page.
        offset: Pagination offset, 0 for the first page.
        fields: Optional list of row keys to keep, for example ['id', 'title'].
            Wins over detail when both are given.
        detail: Preset for how big a row is: 'name' (id and title), 'summary'
            (adds topics, embedding state, insight count, status and date) or
            'full' (the whole row).

    Returns:
        The page of sources plus count, offset, limit and the next_offset to
        pass for the following page (None once the API returns a short page).
    """
    limit = _require_range(
        limit,
        "limit",
        1,
        SOURCES_LIMIT_MAX,
        f"Ask for between 1 and {SOURCES_LIMIT_MAX} sources per call: the API "
        "pages at 100 and answers 422 above it.",
    )
    offset = _require_range(
        offset,
        "offset",
        0,
        1_000_000,
        "Pass offset 0 for the first page.",
    )
    if notebook_id is not None:
        notebook_id = _require_text(
            notebook_id,
            "notebook_id",
            "Pass a notebook ID, or leave it out to list every notebook.",
        )
    detail = _require_detail(detail, DETAIL_LEVELS)
    fields = _require_fields(fields)

    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if notebook_id is not None:
        params["notebook_id"] = notebook_id

    sources = await make_request("GET", "/api/sources", params=params)
    rows = sources if isinstance(sources, list) else []
    page = [_select_fields(row, fields, detail, SOURCE_DETAIL_FIELDS) for row in rows]
    return {
        "request_id": generate_request_id(),
        "count": len(page),
        "offset": offset,
        "limit": limit,
        "next_offset": offset + len(page) if len(rows) == limit else None,
        "sources": page,
    }

@mcp.tool()
async def get_source(source_id: str) -> dict[str, Any]:
    """Get a specific source by ID.
    
    Args:
        source_id: Source ID (e.g., 'source:abc123')
    
    Returns:
        Source details
    """
    source = await make_request("GET", f"/api/sources/{source_id}")
    return {
        "request_id": generate_request_id(),
        "source": source,
    }

def _build_source_payload(
    notebook_id: str,
    type: str,
    url: Optional[str],
    content: Optional[str],
    title: Optional[str],
    embed: bool,
    async_processing: bool,
) -> dict[str, Any]:
    """Validate the source fields and build the API payload.

    Args:
        notebook_id: Notebook the source belongs to.
        type: Source type, 'link' or 'text'.
        url: Page to add, for a link source.
        content: Text to add, for a text source.
        title: Optional title.
        embed: Whether the source must be embedded.
        async_processing: Whether the source is processed in the background.

    Returns:
        The payload for POST /api/sources/json.

    Raises:
        ToolInputError: If a required field is missing or invalid.
    """
    if type not in ("link", "text"):
        raise ToolInputError(
            "invalid_input",
            "upload sources are not supported because MCP cannot transfer files",
            "Use type='link' with a url, or type='text' with content.",
        )
    notebook_id = _require_text(
        notebook_id,
        "notebook_id",
        "Pass a notebook ID, or create one first with create_notebook.",
    )
    if type == "link":
        url = _require_text(
            url,
            "url",
            "Pass the page to add, for example 'https://example.com/post'.",
        )
    elif type == "text":
        content = _require_text(
            content,
            "content",
            "Pass the text to add, or use type='link' with a url.",
        )
    if title is not None:
        title = _require_text(
            title,
            "title",
            "Pass a non-empty title, or omit it to let the API pick one.",
        )

    data: dict[str, Any] = {
        "notebook_id": notebook_id,
        "type": type,
        "embed": embed,
        "async_processing": async_processing,
    }
    if url is not None:
        data["url"] = url
    if content is not None:
        data["content"] = content
    if title is not None:
        data["title"] = title
    return data


@mcp.tool()
async def create_source(
    notebook_id: str,
    type: Literal["link", "text"],
    url: str | None = None,
    content: str | None = None,
    title: str | None = None,
    embed: bool = True,
    async_processing: bool = True,
) -> dict[str, Any]:
    """Create a link or text source.

    Args:
        notebook_id: Notebook ID to add source to
        type: Source type ('link' or 'text')
        url: URL for link type sources
        content: Text for text type sources
        title: Optional title
        embed: Whether to generate embeddings (default: True)
        async_processing: Whether to process the source asynchronously

    Returns:
        Created source details
    """
    data = _build_source_payload(
        notebook_id,
        type,
        url,
        content,
        title,
        embed,
        async_processing,
    )

    source = await make_request("POST", "/api/sources/json", json_data=data)
    return {
        "request_id": generate_request_id(),
        "source": source,
    }


@mcp.tool()
async def update_source(
    source_id: str,
    title: Optional[str] = None,
    topics: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Update a source.
    
    Args:
        source_id: Source ID
        title: Optional new title
        topics: Optional list of topics
    
    Returns:
        Updated source details
    """
    data = {}
    if title is not None:
        data["title"] = title
    if topics is not None:
        data["topics"] = topics
    
    source = await make_request("PUT", f"/api/sources/{source_id}", json_data=data)
    return {
        "request_id": generate_request_id(),
        "source": source,
    }

@mcp.tool()
async def delete_source(source_id: str) -> dict[str, Any]:
    """Delete a source.
    
    Args:
        source_id: Source ID
    
    Returns:
        Success message
    """
    result = await make_request("DELETE", f"/api/sources/{source_id}")
    return {
        "request_id": generate_request_id(),
        "result": result,
    }

# -----------------------------
# Notes API Tools
# -----------------------------

@mcp.tool()
async def list_notes(
    notebook_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Get all notes with optional filtering.
    
    Args:
        notebook_id: Optional notebook ID to filter by
        limit: Maximum number of results (1-200)
        offset: Pagination offset
    
    Returns:
        Dictionary with notes list and metadata
    """
    limit = max(1, min(limit, MAX_LIMIT))
    params = {"limit": limit, "offset": offset}
    if notebook_id is not None:
        params["notebook_id"] = notebook_id
    
    notes = await make_request("GET", "/api/notes", params=params)
    return {
        "request_id": generate_request_id(),
        "count": len(notes) if isinstance(notes, list) else 0,
        "notes": notes,
    }

@mcp.tool()
async def get_note(note_id: str) -> dict[str, Any]:
    """Get a specific note by ID.
    
    Args:
        note_id: Note ID (e.g., 'note:abc123')
    
    Returns:
        Note details
    """
    note = await make_request("GET", f"/api/notes/{note_id}")
    return {
        "request_id": generate_request_id(),
        "note": note,
    }

@mcp.tool()
async def create_note(
    notebook_id: str,
    title: str,
    content: str,
    topics: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Create a new note.
    
    Args:
        notebook_id: Notebook ID to add note to
        title: Note title
        content: Note content
        topics: Optional list of topics
    
    Returns:
        Created note details
    """
    data = {
        "notebook_id": notebook_id,
        "title": title,
        "content": content,
    }
    if topics is not None:
        data["topics"] = topics
    
    note = await make_request("POST", "/api/notes", json_data=data)
    return {
        "request_id": generate_request_id(),
        "note": note,
    }

@mcp.tool()
async def update_note(
    note_id: str,
    title: Optional[str] = None,
    content: Optional[str] = None,
    topics: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Update a note.
    
    Args:
        note_id: Note ID
        title: Optional new title
        content: Optional new content
        topics: Optional list of topics
    
    Returns:
        Updated note details
    """
    data = {}
    if title is not None:
        data["title"] = title
    if content is not None:
        data["content"] = content
    if topics is not None:
        data["topics"] = topics
    
    note = await make_request("PUT", f"/api/notes/{note_id}", json_data=data)
    return {
        "request_id": generate_request_id(),
        "note": note,
    }

@mcp.tool()
async def delete_note(note_id: str) -> dict[str, Any]:
    """Delete a note.
    
    Args:
        note_id: Note ID
    
    Returns:
        Success message
    """
    result = await make_request("DELETE", f"/api/notes/{note_id}")
    return {
        "request_id": generate_request_id(),
        "result": result,
    }

# -----------------------------
# Search API Tools
# -----------------------------

@mcp.tool()
async def search(
    query: str,
    type: str = "vector",
    notebook_id: Optional[str] = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search content using vector or text search.
    
    Args:
        query: Search query
        type: Search type ('vector' or 'text')
        notebook_id: Optional notebook ID to limit search
        limit: Maximum number of results (1-50)
    
    Returns:
        Search results
    """
    limit = max(1, min(limit, 50))
    data = {
        "query": query,
        "type": type,
        "limit": limit,
    }
    if notebook_id is not None:
        data["notebook_id"] = notebook_id
    
    results = await make_request("POST", "/api/search", json_data=data)
    return {
        "request_id": generate_request_id(),
        "results": results,
    }

async def _resolve_models(
    strategy_model: Optional[str] = None,
    answer_model: Optional[str] = None,
    final_answer_model: Optional[str] = None,
) -> tuple[str, str, str]:
    """PATCHED-DEFAULT-V1: fall back to the app default chat model.

    Settings -> Modelli is the single source of truth, so the MCP follows
    it when the caller omits a model id instead of demanding one.
    """
    if strategy_model and answer_model and final_answer_model:
        return strategy_model, answer_model, final_answer_model
    defaults = await make_request("GET", "/api/models/defaults")
    fallback = (
        defaults.get("default_chat_model")
        if isinstance(defaults, dict) else None
    )
    if not fallback:
        raise Exception(
            "no model supplied and no default_chat_model configured"
        )
    return (
        strategy_model or fallback,
        answer_model or fallback,
        final_answer_model or fallback,
    )


def _consume_ask_event(payload: str, result: dict[str, Any]) -> bool:
    """Apply one decoded SSE event and return whether it completed the stream.

    Args:
        payload: JSON payload from an SSE data field.
        result: Mutable accumulator for the ask response.

    Returns:
        True when the event is the stream completion marker.

    Raises:
        ToolApiError: If the event is malformed or reports an ask error.
    """
    try:
        event = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ToolApiError(
            "invalid_stream",
            "Open Notebook returned an invalid SSE event",
            "The ask endpoint must send one JSON object per 'data:' line.",
        ) from exc
    if not isinstance(event, dict):
        raise ToolApiError(
            "invalid_stream",
            "Open Notebook returned an invalid SSE event",
            "The ask endpoint must send one JSON object per 'data:' line.",
        )

    event_type = event.get("type")
    if event_type == "strategy":
        result["strategy"] = event.get("reasoning")
        result["searches"] = event.get("searches", [])
    elif event_type == "answer":
        result["answers"].append(event.get("content", ""))
    elif event_type == "final_answer":
        result["final_answer"] = event.get("content")
    elif event_type == "complete":
        result["final_answer"] = event.get("final_answer") or result["final_answer"]
        return True
    elif event_type == "error":
        message = event.get("message") or "Open Notebook ask stream failed"
        raise ToolApiError(
            "ask_failed",
            str(message),
            "The upstream ask job failed; check the notebook's models and sources.",
        )
    return False


async def _stream_ask(data: dict[str, Any]) -> dict[str, Any]:
    """Read and decode the Open Notebook ask endpoint as Server-Sent Events.

    Args:
        data: Validated ask request body.

    Returns:
        Strategy, searches, partial answers, and final answer.

    Raises:
        ToolApiError: If the upstream response or event stream is invalid.
        ToolTimeoutError: If the complete request exceeds the configured deadline.
    """
    base_url = get_base_url()
    url = f"{base_url}/api/search/ask"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    auth_token = get_auth_token()
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    result: dict[str, Any] = {
        "strategy": None,
        "searches": [],
        "answers": [],
        "final_answer": None,
    }
    completed = False

    try:
        async with (
            asyncio.timeout(DEFAULT_TIMEOUT_S),
            httpx.AsyncClient(
                follow_redirects=True, timeout=DEFAULT_TIMEOUT_S
            ) as client,
            client.stream("POST", url, headers=headers, json=data) as response,
        ):
            if not response.is_success:
                body = await response.aread()
                message = body.decode("utf-8", "replace").strip()
                try:
                    error_body = json.loads(message)
                    if isinstance(error_body, dict):
                        message = str(
                            error_body.get("detail")
                            or error_body.get("message")
                            or message
                        )
                except json.JSONDecodeError:
                    pass
                raise ToolApiError(
                    "api_error",
                    f"API request failed ({response.status_code}): "
                    f"{message or response.reason_phrase}",
                    "The ask endpoint refused the request; check the payload and the "
                    "notebook's models.",
                )

            content_type = response.headers.get("content-type", "")
            if not content_type.lower().startswith("text/event-stream"):
                raise ToolApiError(
                    "invalid_stream",
                    "Open Notebook ask endpoint did not return an SSE stream",
                    "Point OPEN_NOTEBOOK_URL at the API port, not the web UI port, "
                    "or use an instance that serves /api/search/ask as SSE.",
                )

            event_data: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    if event_data:
                        completed = _consume_ask_event("\n".join(event_data), result)
                        event_data.clear()
                        if completed:
                            break
                elif line.startswith(":"):
                    continue
                elif line.startswith("data:"):
                    value = line[5:]
                    event_data.append(value.removeprefix(" "))

            if event_data and not completed:
                completed = _consume_ask_event("\n".join(event_data), result)
    except TimeoutError as exc:
        raise ToolTimeoutError(
            "timeout",
            f"Ask request exceeded the {DEFAULT_TIMEOUT_S:g}s overall deadline",
            "Raise OPEN_NOTEBOOK_TIMEOUT_S, or ask a shorter question.",
        ) from exc
    except httpx.TimeoutException as exc:
        raise ToolTimeoutError(
            "timeout",
            f"Ask request timed out after {DEFAULT_TIMEOUT_S:g}s",
            "Raise OPEN_NOTEBOOK_TIMEOUT_S, or ask a shorter question.",
        ) from exc
    except httpx.RequestError as exc:
        raise ToolApiError(
            "network_error",
            f"Could not reach the Open Notebook ask endpoint: {exc}",
            "Check that OPEN_NOTEBOOK_URL points to a reachable API instance.",
        ) from exc

    if not completed:
        raise ToolApiError(
            "incomplete_stream",
            "Open Notebook ask stream ended before completion",
            "The upstream stream closed without a 'complete' event; retry the question.",
        )
    if not result["final_answer"]:
        raise ToolApiError(
            "incomplete_stream",
            "Open Notebook ask stream completed without a final answer",
            "The upstream stream carried no final answer; retry the question.",
        )
    return result


@mcp.tool()
async def ask_question(
    question: str,
    strategy_model: Optional[str] = None,
    answer_model: Optional[str] = None,
    final_answer_model: Optional[str] = None,
    notebook_id: Optional[str] = None,
) -> dict[str, Any]:
    """Ask a question about your content with detailed control.

    Args:
        question: Question to ask
        strategy_model: Model ID for strategy generation
        answer_model: Model ID for answering
        final_answer_model: Model ID for final answer synthesis
        notebook_id: Optional notebook ID to limit context

    Returns:
        Answer with sources and reasoning
    """
    # PATCHED-DEFAULT-V1: models optional -> fall back to the app default.
    strategy_model, answer_model, final_answer_model = await _resolve_models(
        strategy_model, answer_model, final_answer_model
    )
    data = {
        "question": question,
        "strategy_model": strategy_model,
        "answer_model": answer_model,
        "final_answer_model": final_answer_model,
    }
    if notebook_id is not None:
        data["notebook_id"] = notebook_id

    streamed = await _stream_ask(data)
    return {
        "request_id": generate_request_id(),
        "result": {
            "question": question,
            "strategy": streamed["strategy"],
            "searches": streamed["searches"],
            "answers": streamed["answers"],
            "final_answer": streamed["final_answer"],
        },
    }


@mcp.tool()
async def ask_simple(
    question: str,
    strategy_model: Optional[str] = None,
    answer_model: Optional[str] = None,
    final_answer_model: Optional[str] = None,
    notebook_id: Optional[str] = None,
) -> dict[str, Any]:
    """Ask a question about your content with simplified interface.

    Args:
        question: Question to ask
        strategy_model: Model ID for strategy generation
        answer_model: Model ID for answering
        final_answer_model: Model ID for final answer synthesis
        notebook_id: Optional notebook ID to limit context

    Returns:
        Simple answer
    """
    # PATCHED-DEFAULT-V1: models optional -> fall back to the app default.
    strategy_model, answer_model, final_answer_model = await _resolve_models(
        strategy_model, answer_model, final_answer_model
    )
    data = {
        "question": question,
        "strategy_model": strategy_model,
        "answer_model": answer_model,
        "final_answer_model": final_answer_model,
    }
    if notebook_id is not None:
        data["notebook_id"] = notebook_id

    streamed = await _stream_ask(data)
    return {
        "request_id": generate_request_id(),
        "result": {
            "question": question,
            "answer": streamed["final_answer"],
        },
    }

# -----------------------------
# Models API Tools
# -----------------------------

@mcp.tool()
async def list_models(limit: int = 50) -> dict[str, Any]:
    """Get all configured AI models.
    
    Args:
        limit: Maximum number of results (1-200)
    
    Returns:
        Dictionary with models list and metadata
    """
    limit = max(1, min(limit, MAX_LIMIT))
    models = await make_request("GET", "/api/models")
    
    if isinstance(models, list):
        models = models[:limit]
    
    return {
        "request_id": generate_request_id(),
        "count": len(models) if isinstance(models, list) else 0,
        "models": models,
    }

@mcp.tool()
async def get_model(model_id: str) -> dict[str, Any]:
    """Get a specific model by ID.
    
    Args:
        model_id: Model ID (e.g., 'model:abc123')
    
    Returns:
        Model details
    """
    model = await make_request("GET", f"/api/models/{model_id}")
    return {
        "request_id": generate_request_id(),
        "model": model,
    }

@mcp.tool()
async def create_model(name: str, provider: str, type: str) -> dict[str, Any]:
    """Create a new AI model configuration.
    
    Args:
        name: Model name (e.g., 'gpt-4', 'claude-3-opus')
        provider: Provider name (e.g., 'openai', 'anthropic')
        type: Model type (e.g., 'language', 'embedding')
    
    Returns:
        Created model details
    """
    data = {
        "name": name,
        "provider": provider,
        "type": type,
    }
    
    model = await make_request("POST", "/api/models", json_data=data)
    return {
        "request_id": generate_request_id(),
        "model": model,
    }

@mcp.tool()
async def delete_model(model_id: str) -> dict[str, Any]:
    """Delete a model configuration.
    
    Args:
        model_id: Model ID
    
    Returns:
        Success message
    """
    result = await make_request("DELETE", f"/api/models/{model_id}")
    return {
        "request_id": generate_request_id(),
        "result": result,
    }

@mcp.tool()
async def get_default_models() -> dict[str, Any]:
    """Get default model configurations.
    
    Returns:
        Default models configuration
    """
    defaults = await make_request("GET", "/api/models/defaults")
    return {
        "request_id": generate_request_id(),
        "defaults": defaults,
    }

# -----------------------------
# Chat API Tools
# -----------------------------

@mcp.tool()
async def list_chat_sessions(
    notebook_id: Optional[str] = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Get all chat sessions with optional filtering.
    
    Args:
        notebook_id: Optional notebook ID to filter by
        limit: Maximum number of results (1-200)
    
    Returns:
        Dictionary with sessions list and metadata
    """
    limit = max(1, min(limit, MAX_LIMIT))
    params = {}
    if notebook_id is not None:
        params["notebook_id"] = notebook_id
    
    sessions = await make_request("GET", "/api/chat/sessions", params=params)
    
    if isinstance(sessions, list):
        sessions = sessions[:limit]
    
    return {
        "request_id": generate_request_id(),
        "count": len(sessions) if isinstance(sessions, list) else 0,
        "sessions": sessions,
    }

@mcp.tool()
async def create_chat_session(notebook_id: str, title: str) -> dict[str, Any]:
    """Create a new chat session.
    
    Args:
        notebook_id: Notebook ID for the session
        title: Session title
    
    Returns:
        Created session details
    """
    data = {
        "notebook_id": notebook_id,
        "title": title,
    }
    
    session = await make_request("POST", "/api/chat/sessions", json_data=data)
    return {
        "request_id": generate_request_id(),
        "session": session,
    }

@mcp.tool()
async def get_chat_session(session_id: str) -> dict[str, Any]:
    """Get a specific chat session by ID.
    
    Args:
        session_id: Session ID (e.g., 'session:abc123')
    
    Returns:
        Session details with message history
    """
    session = await make_request("GET", f"/api/chat/sessions/{session_id}")
    return {
        "request_id": generate_request_id(),
        "session": session,
    }

@mcp.tool()
async def update_chat_session(
    session_id: str,
    title: Optional[str] = None,
) -> dict[str, Any]:
    """Update a chat session.
    
    Args:
        session_id: Session ID
        title: Optional new title
    
    Returns:
        Updated session details
    """
    data = {}
    if title is not None:
        data["title"] = title
    
    session = await make_request("PUT", f"/api/chat/sessions/{session_id}", json_data=data)
    return {
        "request_id": generate_request_id(),
        "session": session,
    }

@mcp.tool()
async def delete_chat_session(session_id: str) -> dict[str, Any]:
    """Delete a chat session.
    
    Args:
        session_id: Session ID
    
    Returns:
        Success message
    """
    result = await make_request("DELETE", f"/api/chat/sessions/{session_id}")
    return {
        "request_id": generate_request_id(),
        "result": result,
    }

@mcp.tool()
async def execute_chat(
    session_id: str,
    message: str,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Send a message in a chat session.
    
    Args:
        session_id: Session ID
        message: Message to send
        context: Optional context data for the conversation
    
    Returns:
        Chat response with AI message
    """
    data = {
        "session_id": session_id,
        "message": message,
    }
    # PATCHED-CONTRACT-V1: the API requires "context" (Field(...) with no
    # default, api/routers/chat.py:69); omitting it returns 422. Empty dict
    # means "no sources/notes", which the chat graph accepts.
    data["context"] = context if context is not None else {}
    
    response = await make_request("POST", "/api/chat/execute", json_data=data)
    return {
        "request_id": generate_request_id(),
        "response": response,
    }

@mcp.tool()
async def get_chat_context(
    notebook_id: str,
    context_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build context for a chat conversation.
    
    Args:
        notebook_id: Notebook ID
        context_config: Optional context configuration
    
    Returns:
        Built context data
    """
    data = {
        "notebook_id": notebook_id,
    }
    # PATCHED-CONTRACT-V1: the API requires "context_config"
    # (api/routers/chat.py:84); omitting it returns 422.
    data["context_config"] = context_config if context_config is not None else {}
    
    context = await make_request("POST", "/api/chat/context", json_data=data)
    return {
        "request_id": generate_request_id(),
        "context": context,
    }

# -----------------------------
# Settings API Tools
# -----------------------------

@mcp.tool()
async def get_settings() -> dict[str, Any]:
    """Get application settings.
    
    Returns:
        Application settings
    """
    settings = await make_request("GET", "/api/settings")
    return {
        "request_id": generate_request_id(),
        "settings": settings,
    }

@mcp.tool()
async def update_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Update application settings.
    
    Args:
        settings: Settings dictionary to update
    
    Returns:
        Updated settings
    """
    result = await make_request("PUT", "/api/settings", json_data=settings)
    return {
        "request_id": generate_request_id(),
        "settings": result,
    }

# -----------------------------
# Native web discovery (#973) and artifacts (#203)
# -----------------------------
# These two capabilities are not CRUD on an existing resource: discovery is a
# search that creates sources, and an artifact is an async generation job.
# They only exist on an instance built from the branch that adds them; against
# an instance that does not serve them these tools return 404.

#: Detail presets for the discovery hits (api/models.py DiscoveredSource).
#: 'full' is absent on purpose: it keeps the whole row.
DISCOVER_DETAIL_FIELDS: dict[str, tuple[str, ...]] = {
    "name": ("title", "url"),
    "summary": ("title", "url", "score", "status", "source_id"),
}


def _build_discover_payload(
    query: str,
    notebook_id: str,
    limit: int,
    provider: Optional[str],
    embed: bool,
    exclude_domains: Optional[list[str]],
    dry_run: bool,
) -> dict[str, Any]:
    """Validate the discovery fields and build the API payload.

    Args:
        query: The topic to search for.
        notebook_id: Notebook the discovered sources land in.
        limit: How many candidates to ask the provider for.
        provider: Optional search backend name.
        embed: Whether the created sources must be embedded.
        exclude_domains: Optional denylist of domains.
        dry_run: Whether to preview without creating anything.

    Returns:
        The payload for POST /api/sources/discover.

    Raises:
        ToolInputError: If a field is missing, blank or outside the API range.
    """
    limit = _require_range(
        limit,
        "limit",
        1,
        DISCOVER_LIMIT_MAX,
        f"Ask for between 1 and {DISCOVER_LIMIT_MAX} sources per call.",
    )
    query = _require_text(
        query,
        "query",
        "Give a non-empty topic to search for, for example 'MCP protocol specification'.",
        max_length=400,
    )
    notebook_id = _require_text(
        notebook_id,
        "notebook_id",
        "Pass a notebook ID, or create one first with create_notebook.",
    )
    if provider is not None:
        provider = _require_text(
            provider,
            "provider",
            "Pass the exact backend name, for example 'tavily'.",
        )
    if exclude_domains is not None:
        exclude_domains = [
            _require_text(
                domain,
                "exclude_domains",
                "Pass domains as bare names, without scheme, path or port.",
            ).lower()
            for domain in exclude_domains
        ]
        if len(exclude_domains) > DISCOVER_EXCLUDE_DOMAINS_MAX:
            raise ToolInputError(
                "invalid_input",
                "exclude_domains accepts at most "
                f"{DISCOVER_EXCLUDE_DOMAINS_MAX} domains, got {len(exclude_domains)}",
                "Drop the least important domains and retry.",
            )

    data: dict[str, Any] = {
        "query": query,
        "notebook_id": notebook_id,
        "limit": limit,
        "embed": embed,
        "dry_run": dry_run,
        "exclude_domains": exclude_domains or [],
    }
    if provider is not None:
        data["provider"] = provider
    return data


@mcp.tool()
async def discover_sources(
    query: str,
    notebook_id: str,
    limit: int = 5,
    provider: Optional[str] = None,
    embed: bool = False,
    exclude_domains: Optional[list[str]] = None,
    dry_run: bool = False,
    fields: Optional[list[str]] = None,
    detail: Detail = "summary",
) -> dict[str, Any]:
    """Search the web and attach the hits to a notebook as sources.

    This is a search that creates sources, not a listing: the endpoint runs one
    provider query and returns that single ranked batch. There is no cursor,
    so the tool takes no offset. Ask again with a narrower query to see other
    hits, or use dry_run to inspect the batch before anything is created.

    Args:
        query: What to search for (the topic to find sources about).
        notebook_id: Notebook to attach the discovered sources to. Create one
            first with create_notebook if the notebook does not exist yet.
        limit: Maximum number of sources to create (1-20).
        provider: Search backend name; omit to use the instance default.
        embed: Embed the created sources so they are usable for vector search.
        exclude_domains: Domains the provider must not return (max 50).
        dry_run: Preview the ranked candidates without creating anything.
        fields: Optional list of hit keys to keep, for example ['title', 'url'].
            Wins over detail when both are given.
        detail: Preset for how big a hit is: 'name' (title and url), 'summary'
            (adds score, status and the created source id) or 'full' (the whole
            hit, including the snippet and any error).

    Returns:
        The provider used, how many sources were created, how many were skipped
        because the notebook already had them, and every hit with its outcome.
    """
    data = _build_discover_payload(
        query,
        notebook_id,
        limit,
        provider,
        embed,
        exclude_domains,
        dry_run,
    )
    detail = _require_detail(detail, DETAIL_LEVELS)
    fields = _require_fields(fields)

    result = await make_request("POST", "/api/sources/discover", json_data=data)
    hits = result.get("results") if isinstance(result, dict) else None
    if isinstance(hits, list):
        result["results"] = [
            _select_fields(hit, fields, detail, DISCOVER_DETAIL_FIELDS)
            for hit in hits
        ]
    return {
        "request_id": generate_request_id(),
        "result": result,
    }


#: Formats the API can render (api/routers/artifacts.py MEDIA_TYPES).
ARTIFACT_FORMATS: tuple[str, ...] = ("md", "html", "docx", "pptx")

#: Valid artifact kinds; 'slides' is the legacy alias of 'deck'.
ARTIFACT_KINDS: tuple[str, ...] = ("report", "deck", "slides")

#: Valid variant per kind (open_notebook/artifacts/outline.py VARIANTS).
ARTIFACT_VARIANTS: dict[str, tuple[str, ...]] = {
    "report": ("document", "illustrated"),
    "deck": ("presenter", "detailed"),
}


def _build_artifact_payload(
    notebook_id: str,
    kind: str,
    variant: Optional[str],
    formats: Optional[list[str]],
    language: str,
    title: Optional[str],
    instructions: Optional[str],
    sections: int,
    model_id: Optional[str],
) -> dict[str, Any]:
    """Validate the artifact fields and build the API payload.

    Args:
        notebook_id: Notebook whose sources feed the artifact.
        kind: 'report', 'deck' or the 'slides' alias.
        variant: Sub-variant of the kind.
        formats: Rendered formats.
        language: Language name for the output.
        title: Optional title override.
        instructions: Optional extra guidance for the writer.
        sections: How many sections to outline and write.
        model_id: Optional model id override.

    Returns:
        The payload for POST /api/artifacts.

    Raises:
        ToolInputError: If a required field is missing or invalid.
    """
    notebook_id = _require_text(
        notebook_id,
        "notebook_id",
        "Pass a notebook ID, or create one first with create_notebook.",
    )
    kind = _require_text(
        kind,
        "kind",
        "Use 'report' for prose or 'deck' for slides.",
    ).lower()
    if kind not in ARTIFACT_KINDS:
        raise ToolInputError(
            "invalid_input",
            f"kind must be one of {', '.join(ARTIFACT_KINDS)}, got {kind!r}",
            "Use 'report' for prose or 'deck' for slides.",
        )
    normalized_kind = "deck" if kind == "slides" else kind
    if variant is not None:
        variant = _require_text(
            variant,
            "variant",
            f"Use one of {', '.join(ARTIFACT_VARIANTS[normalized_kind])}.",
        ).lower()
        if variant not in ARTIFACT_VARIANTS[normalized_kind]:
            raise ToolInputError(
                "invalid_input",
                f"variant {variant!r} is not valid for a {normalized_kind}",
                f"Use one of {', '.join(ARTIFACT_VARIANTS[normalized_kind])}.",
            )
    if formats is None:
        formats = ["md", "docx"]
    else:
        formats = [
            _require_text(
                fmt,
                "formats",
                "Pass formats as bare names, for example 'md'.",
            )
            for fmt in formats
        ]
        unsupported = [fmt for fmt in formats if fmt not in ARTIFACT_FORMATS]
        if unsupported:
            raise ToolInputError(
                "invalid_input",
                f"unsupported artifact formats: {', '.join(unsupported)}",
                f"Supported formats are {', '.join(ARTIFACT_FORMATS)}.",
            )
    sections = _require_range(
        sections,
        "sections",
        1,
        50,
        "Ask for between 1 and 50 sections.",
    )
    language = _require_text(
        language,
        "language",
        "Pass a language name, for example 'en'.",
    )
    if title is not None:
        title = _require_text(
            title,
            "title",
            "Pass a non-empty title, or omit it to let the API pick one.",
        )
    if instructions is not None:
        instructions = _require_text(
            instructions,
            "instructions",
            "Pass non-empty guidance for the writer, or omit it.",
        )
    if model_id is not None:
        model_id = _require_text(
            model_id,
            "model_id",
            "Pass the model id to use, for example 'openai/gpt-4o-mini'.",
        )
    data: dict[str, Any] = {
        "notebook_id": notebook_id,
        "kind": kind,
        "formats": formats,
        "language": language,
        "sections": sections,
    }
    if variant is not None:
        data["variant"] = variant
    if title is not None:
        data["title"] = title
    if instructions is not None:
        data["instructions"] = instructions
    if model_id is not None:
        data["model_id"] = model_id
    return data


@mcp.tool()
async def generate_artifact(
    notebook_id: str,
    kind: str = "report",
    variant: Optional[str] = None,
    formats: Optional[list[str]] = None,
    language: str = "en",
    title: Optional[str] = None,
    instructions: Optional[str] = None,
    sections: int = 5,
    model_id: Optional[str] = None,
) -> dict[str, Any]:
    """Generate a report or slide artifact from a notebook.

    Args:
        notebook_id: Notebook whose sources feed the artifact.
        kind: 'report' or 'deck'. 'slides' is accepted as an alias for deck.
            Any other value is rejected; the comparison ignores case, so
            'Deck' works. A deck is written as bullets, a report as prose.
            Which FILES come out is a separate question, see formats.
        variant: Sub-variant of the kind. A report has 'document' (plain
            prose, no diagrams) and 'illustrated' (prose plus ```mermaid```
            diagrams rendered to images); a deck has 'presenter' (concise
            bullets) and 'detailed' (full sentences under each bullet).
            Defaults: 'document' for a report, 'presenter' for a deck. A
            variant of the other kind is rejected locally with an
            invalid_input error, before any request is sent. Diagrams are
            off for a plain report and on for everything else, so an
            'illustrated' report is the way to get prose AND diagrams.
        formats: Rendered formats ('md', 'html', 'docx', 'pptx'). This is
            independent of kind: a deck is NOT automatically a pptx. Pass
            'pptx' explicitly if you want the slide file.
        language: Language name for the output.
        title: Optional title override. Putting "with diagrams" in the title
            does NOT enable diagrams - use variant='illustrated' or a deck.
        instructions: Optional extra guidance for the writer.
        sections: How many sections to outline and write.
        model_id: Optional model id override for the generation.

    Returns:
        A command_id and an artifact_id. Generation runs in the background, so
        poll get_artifact with the artifact_id until job_status settles.
    """
    data = _build_artifact_payload(
        notebook_id,
        kind,
        variant,
        formats,
        language,
        title,
        instructions,
        sections,
        model_id,
    )

    result = await make_request("POST", "/api/artifacts", json_data=data)
    return {
        "request_id": generate_request_id(),
        "result": result,
    }

#: Preset key sets for the detail levels of the artifact list, so a listing can
#: stay cheap in the model's context. 'full' is absent: it keeps the whole row.
ARTIFACT_DETAIL_FIELDS: dict[str, tuple[str, ...]] = {
    "name": ("id", "title"),
    "summary": (
        "id",
        "notebook_id",
        "title",
        "kind",
        "variant",
        "job_status",
        "created",
    ),
}

@mcp.tool()
async def list_artifacts(
    notebook_id: Optional[str] = None,
    limit: int = 25,
    offset: int = 0,
    fields: Optional[list[str]] = None,
    detail: Detail = "summary",
) -> dict[str, Any]:
    """List generated artifacts, newest first.

    The upstream endpoint returns the whole list, so the page is cut here: a
    tool that can return a list has to stay small by default (AGENTS.md 4.1).

    Args:
        notebook_id: Optional notebook ID to filter by.
        limit: How many artifacts to return (1-200, default 25).
        offset: How many artifacts to skip, for paging.
        fields: Optional list of row keys to keep, for example ['id', 'title'].
            Wins over detail when both are given.
        detail: Preset for how big a row is: 'name' (id and title), 'summary'
            (adds notebook, kind, variant, job status and date) or 'full' (the
            whole row).

    Returns:
        The page of artifacts plus total, offset, limit and the next_offset to
        pass for the following page (None once the list is exhausted).
    """
    limit = _require_range(
        limit,
        "limit",
        1,
        MAX_LIMIT,
        f"Ask for between 1 and {MAX_LIMIT} artifacts per call.",
    )
    offset = _require_range(
        offset,
        "offset",
        0,
        1_000_000,
        "Pass offset 0 for the first page.",
    )
    if notebook_id is not None:
        notebook_id = _require_text(
            notebook_id,
            "notebook_id",
            "Pass a notebook ID, or leave it out to list every notebook.",
        )
    detail = _require_detail(detail, DETAIL_LEVELS)
    fields = _require_fields(fields)
    params: dict[str, Any] = {}
    if notebook_id is not None:
        params["notebook_id"] = notebook_id

    artifacts = await make_request("GET", "/api/artifacts", params=params)
    rows: list[Any] = artifacts if isinstance(artifacts, list) else []
    total = len(rows)
    page = [
        _select_fields(row, fields, detail, ARTIFACT_DETAIL_FIELDS)
        for row in rows[offset : offset + limit]
    ]
    consumed = min(offset + len(page), total)
    return {
        "request_id": generate_request_id(),
        "count": len(page),
        "total": total,
        "offset": offset,
        "limit": limit,
        "next_offset": consumed if consumed < total else None,
        "artifacts": page,
    }

@mcp.tool()
async def get_artifact(artifact_id: str) -> dict[str, Any]:
    """Get one artifact, including the status of its generation job.

    Args:
        artifact_id: Artifact ID (e.g. 'artifact:abc123').

    Returns:
        Artifact details with job_status, error_message and download_url.
    """
    artifact_id = _require_text(
        artifact_id,
        "artifact_id",
        "Pass the artifact ID, for example 'artifact:abc123'.",
    )
    artifact = await make_request("GET", f"/api/artifacts/{artifact_id}")
    return {
        "request_id": generate_request_id(),
        "artifact": artifact,
    }

@mcp.tool()
async def delete_artifact(artifact_id: str) -> dict[str, Any]:
    """Delete an artifact and its rendered files.

    Args:
        artifact_id: Artifact ID.

    Returns:
        Success message.
    """
    artifact_id = _require_text(
        artifact_id,
        "artifact_id",
        "Pass the artifact ID, for example 'artifact:abc123'.",
    )
    result = await make_request("DELETE", f"/api/artifacts/{artifact_id}")
    return {
        "request_id": generate_request_id(),
        "result": result,
    }

# -----------------------------
# Dual transport entrypoint
# -----------------------------

def main() -> None:
    """Main entry point for the MCP server."""
    # Default: stdio (Codex/local)
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()

    if transport == "stdio":
        mcp.run(transport="stdio")
        return

    # Streamable HTTP (prod/remote)
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    path = os.getenv("MCP_PATH", "/mcp")
    stateless_http = os.getenv("STATELESS_HTTP", "1") == "1"
    json_response = os.getenv("JSON_RESPONSE", "1") == "1"

    # DNS-rebinding protection: auto-enabled by mcp 2.x only when the bind
    # address is loopback.  When the server listens on 0.0.0.0 (the default
    # for remote access), the protection is off by default.  MCP_ALLOWED_HOSTS
    # lets operators opt back in with an explicit allowlist.
    transport_security = None
    extra_hosts = [
        h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()
    ]
    if extra_hosts:
        from mcp.server.transport_security import TransportSecuritySettings

        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"] + extra_hosts,
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]
            + [f"http://{h}" for h in extra_hosts],
        )

    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        streamable_http_path=path,
        stateless_http=stateless_http,
        json_response=json_response,
        transport_security=transport_security,
    )

if __name__ == "__main__":
    main()
