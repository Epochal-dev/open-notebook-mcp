"""Tests for the Open Notebook MCP server."""

import json
from collections.abc import AsyncIterator

import respx
from httpx import AsyncByteStream, Response

from open_notebook_mcp.server import CAPABILITIES, search_capabilities

#: The host the tools are pointed at during tests, so respx can intercept.
API_BASE = "http://api.test"


def _mock_api(monkeypatch):
    """Point the server at a test host so respx sees the real HTTP path."""
    monkeypatch.setenv("OPEN_NOTEBOOK_URL", API_BASE)
    return respx.mock(assert_all_called=True)


def test_capabilities_defined():
    """Test that all capabilities are properly defined."""
    assert len(CAPABILITIES) > 0, "No capabilities defined"

    # Check that all capabilities have required fields
    for cap in CAPABILITIES:
        assert cap.name, "Missing name for capability"
        assert cap.summary, f"Missing summary for {cap.name}"
        assert cap.tags, f"Missing tags for {cap.name}"
        assert cap.args is not None, f"Missing args for {cap.name}"
        assert cap.returns, f"Missing returns for {cap.name}"
        assert cap.example is not None, f"Missing example for {cap.name}"
        assert cap.typical_bytes > 0, f"Missing typical_bytes for {cap.name}"


def test_search_capabilities_basic():
    """Test basic search_capabilities functionality."""
    result = search_capabilities(query="", detail="name", limit=50)

    assert "request_id" in result
    assert "count" in result
    assert "matches" in result
    assert result["count"] > 0
    assert len(result["matches"]) == result["count"]


def test_search_capabilities_query():
    """Test search_capabilities with query."""
    # Search for notebook-related tools
    result = search_capabilities(query="notebook", detail="summary", limit=10)
    assert result["count"] > 0

    # All matches should have notebook-related content
    for match in result["matches"]:
        assert "name" in match
        assert "summary" in match
        assert "tags" in match


def test_search_capabilities_detail_levels():
    """Test different detail levels."""
    # Name only
    result = search_capabilities(query="", detail="name", limit=1)
    assert len(result["matches"]) == 1
    match = result["matches"][0]
    assert "name" in match
    assert "summary" not in match

    # Summary
    result = search_capabilities(query="", detail="summary", limit=1)
    match = result["matches"][0]
    assert "name" in match
    assert "summary" in match
    assert "tags" in match
    assert "args" not in match

    # Full
    result = search_capabilities(query="", detail="full", limit=1)
    match = result["matches"][0]
    assert "name" in match
    assert "summary" in match
    assert "tags" in match
    assert "args" in match
    assert "returns" in match
    assert "example" in match


def test_search_capabilities_limit():
    """Test that limit parameter is respected."""
    result = search_capabilities(query="", detail="name", limit=5)
    assert result["count"] <= 5
    assert len(result["matches"]) <= 5

    result = search_capabilities(query="", detail="name", limit=100)
    # Should be capped at 50
    assert result["count"] <= 50


def test_generate_artifact_asks_for_a_handful_of_sections():
    """A document is a book: few broad chapters, so the default must match the API."""
    import inspect

    from open_notebook_mcp.server import generate_artifact

    default = inspect.signature(generate_artifact).parameters["sections"].default
    assert default == 5


def test_generate_artifact_accepts_a_variant():
    """The kind sub-variant is part of the contract and must reach the API."""
    import inspect

    from open_notebook_mcp.server import generate_artifact

    parameters = inspect.signature(generate_artifact).parameters
    assert "variant" in parameters
    assert parameters["variant"].default is None

    capability = next(cap for cap in CAPABILITIES if cap.name == "generate_artifact")
    assert "variant" in capability.args


def test_create_source_posts_to_the_json_endpoint(monkeypatch):
    """POST /api/sources is multipart-only; the JSON contract lives at /sources/json."""
    import asyncio

    from open_notebook_mcp import server

    with _mock_api(monkeypatch) as mock:
        route = mock.post(f"{API_BASE}/api/sources/json").mock(
            return_value=Response(200, json={"id": "source:1"})
        )
        result = asyncio.run(
            server.create_source(
                notebook_id="notebook:1",
                type="link",
                url="https://example.com",
                embed=False,
            )
        )

    sent = json.loads(route.calls.last.request.content)
    assert sent["notebook_id"] == "notebook:1"
    assert sent["type"] == "link"
    assert sent["embed"] is False
    assert sent["async_processing"] is True
    assert result["source"]["id"] == "source:1"


def test_create_source_sends_text_content_and_async_processing(monkeypatch):
    """Text sources need content and should use the non-blocking API path."""
    import asyncio

    from open_notebook_mcp import server

    with _mock_api(monkeypatch) as mock:
        route = mock.post(f"{API_BASE}/api/sources/json").mock(
            return_value=Response(200, json={"id": "source:text"})
        )
        result = asyncio.run(
            server.create_source(
                notebook_id="notebook:1",
                type="text",
                content="Source body",
                title="Text source",
            )
        )

    sent = json.loads(route.calls.last.request.content)
    assert sent["type"] == "text"
    assert sent["content"] == "Source body"
    assert sent["async_processing"] is True
    assert result["source"]["id"] == "source:text"


def test_create_source_rejects_text_without_content(monkeypatch):
    """Reject text sources before making a request when content is missing."""
    import asyncio

    import pytest

    from open_notebook_mcp import server

    with _mock_api(monkeypatch), pytest.raises(ValueError, match="content is required"):
        asyncio.run(server.create_source(notebook_id="notebook:1", type="text"))


def test_create_source_rejects_link_without_url(monkeypatch):
    """Reject links before making a request when URL is missing."""
    import asyncio

    import pytest

    from open_notebook_mcp import server

    with _mock_api(monkeypatch), pytest.raises(ValueError, match="url is required"):
        asyncio.run(server.create_source(notebook_id="notebook:1", type="link"))


def test_create_source_rejects_uploads_without_a_file_transfer_contract(monkeypatch):
    """MCP cannot upload a local file because the tool has no file input."""
    import asyncio

    import pytest

    from open_notebook_mcp import server

    with (
        _mock_api(monkeypatch),
        pytest.raises(ValueError, match="upload sources are not supported"),
    ):
        asyncio.run(server.create_source(notebook_id="notebook:1", type="upload"))


def test_create_source_capability_documents_text_and_async_inputs():
    """Tool discovery must expose the supported source input contract."""
    capability = next(cap for cap in CAPABILITIES if cap.name == "create_source")

    assert capability.args["content"] == "Optional[str]"
    assert capability.args["async_processing"] == "bool"
    assert "upload" not in capability.summary


def test_ask_question_parses_streamed_strategy_answers_and_final_answer(monkeypatch):
    """The detailed ask tool returns decoded SSE events instead of raw frames."""
    import asyncio

    from open_notebook_mcp import server

    events = [
        {
            "type": "strategy",
            "reasoning": "Find evidence",
            "searches": [{"term": "AI"}],
        },
        {"type": "answer", "content": "Partial answer"},
        {"type": "final_answer", "content": "Final answer"},
        {"type": "complete", "final_answer": "Final answer"},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    with _mock_api(monkeypatch) as mock:
        route = mock.post(f"{API_BASE}/api/search/ask").mock(
            return_value=Response(
                200,
                text=body,
                headers={"content-type": "text/event-stream"},
            )
        )
        result = asyncio.run(
            server.ask_question(
                question="What is AI?",
                strategy_model="model:1",
                answer_model="model:1",
                final_answer_model="model:1",
                notebook_id="notebook:1",
            )
        )

    sent = json.loads(route.calls.last.request.content)
    assert route.calls.last.request.headers["accept"] == "text/event-stream"
    assert sent["question"] == "What is AI?"
    assert sent["notebook_id"] == "notebook:1"
    assert result["result"] == {
        "question": "What is AI?",
        "strategy": "Find evidence",
        "searches": [{"term": "AI"}],
        "answers": ["Partial answer"],
        "final_answer": "Final answer",
    }


def test_ask_simple_returns_final_answer_from_stream(monkeypatch):
    """The simple ask tool keeps its compact answer contract over SSE."""
    import asyncio

    from open_notebook_mcp import server

    events = [
        {"type": "final_answer", "content": "Concise answer"},
        {"type": "complete", "final_answer": "Concise answer"},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    with _mock_api(monkeypatch) as mock:
        route = mock.post(f"{API_BASE}/api/search/ask").mock(
            return_value=Response(
                200,
                text=body,
                headers={"content-type": "text/event-stream"},
            )
        )
        result = asyncio.run(
            server.ask_simple(
                question="Summarize this",
                strategy_model="model:1",
                answer_model="model:1",
                final_answer_model="model:1",
            )
        )

    assert route.calls.last.request.headers["accept"] == "text/event-stream"
    assert result["result"] == {
        "question": "Summarize this",
        "answer": "Concise answer",
    }


def test_ask_tools_raise_actionable_error_from_error_event(monkeypatch):
    """An SSE error event must not be returned as if the ask succeeded."""
    import asyncio

    import pytest

    from open_notebook_mcp import server

    body = f"data: {json.dumps({'type': 'error', 'message': 'Model unavailable'})}\n\n"
    with _mock_api(monkeypatch) as mock:
        mock.post(f"{API_BASE}/api/search/ask").mock(
            return_value=Response(
                200,
                text=body,
                headers={"content-type": "text/event-stream"},
            )
        )
        with pytest.raises(RuntimeError, match="Model unavailable"):
            asyncio.run(
                server.ask_simple(
                    question="Question",
                    strategy_model="model:1",
                    answer_model="model:1",
                    final_answer_model="model:1",
                )
            )


def test_ask_tools_reject_streams_that_end_without_completion(monkeypatch):
    """A partial stream must not be reported as a successful answer."""
    import asyncio

    import pytest

    from open_notebook_mcp import server

    body = f"data: {json.dumps({'type': 'final_answer', 'content': 'Partial'})}\n\n"
    with _mock_api(monkeypatch) as mock:
        mock.post(f"{API_BASE}/api/search/ask").mock(
            return_value=Response(
                200,
                text=body,
                headers={"content-type": "text/event-stream"},
            )
        )
        with pytest.raises(RuntimeError, match="ended before completion"):
            asyncio.run(
                server.ask_simple(
                    question="Question",
                    strategy_model="model:1",
                    answer_model="model:1",
                    final_answer_model="model:1",
                )
            )


def test_ask_tools_enforce_an_overall_stream_deadline(monkeypatch):
    """Frequent SSE heartbeats cannot extend the total request deadline."""
    import asyncio

    import pytest

    from open_notebook_mcp import server

    class HeartbeatStream(AsyncByteStream):
        """Emit regular heartbeats until the overall deadline expires."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for _ in range(10):
                await asyncio.sleep(0.02)
                yield b": keep-alive\n\n"

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(server, "DEFAULT_TIMEOUT_S", 0.05)
    with _mock_api(monkeypatch) as mock:
        mock.post(f"{API_BASE}/api/search/ask").mock(
            return_value=Response(
                200,
                stream=HeartbeatStream(),
                headers={"content-type": "text/event-stream"},
            )
        )
        with pytest.raises(TimeoutError, match="overall deadline"):
            asyncio.run(
                server.ask_simple(
                    question="Question",
                    strategy_model="model:1",
                    answer_model="model:1",
                    final_answer_model="model:1",
                )
            )


def test_timeout_reader_accepts_a_positive_finite_value():
    """A sane OPEN_NOTEBOOK_TIMEOUT_S is read as a float."""
    from open_notebook_mcp.server import _read_timeout_s

    assert _read_timeout_s("45") == 45.0
    assert _read_timeout_s("0.5") == 0.5


def test_timeout_reader_rejects_values_httpx_would_accept():
    """Zero, negative, infinite and NaN timeouts must not reach httpx."""
    import pytest

    from open_notebook_mcp.server import _read_timeout_s

    for raw in ("0", "-1", "nan", "inf", "-inf", "abc", ""):
        with pytest.raises(ValueError):
            _read_timeout_s(raw)


def test_default_timeout_is_positive_and_finite():
    """The timeout the client actually uses must be usable."""
    import math

    from open_notebook_mcp.server import DEFAULT_TIMEOUT_S

    assert math.isfinite(DEFAULT_TIMEOUT_S)
    assert DEFAULT_TIMEOUT_S > 0


def test_ask_tools_publish_optional_models():
    """The capability metadata has to match the Optional[str] signatures."""
    for name in ("ask_question", "ask_simple"):
        capability = next(cap for cap in CAPABILITIES if cap.name == name)
        for key in ("strategy_model", "answer_model", "final_answer_model"):
            assert capability.args[key] == "Optional[str]", f"{name}.{key}"


def test_list_artifacts_defaults_to_a_small_page(monkeypatch):
    """A list tool must not dump everything: 25 by default, with a cursor."""
    import asyncio

    import open_notebook_mcp.server as server

    rows = [{"id": f"artifact:{i}", "title": f"A{i}"} for i in range(60)]

    with _mock_api(monkeypatch) as mock:
        mock.get(f"{API_BASE}/api/artifacts").mock(
            return_value=Response(200, json=rows)
        )
        page = asyncio.run(server.list_artifacts())

    assert [a["id"] for a in page["artifacts"]] == [f"artifact:{i}" for i in range(25)]
    assert page["total"] == 60
    assert page["offset"] == 0
    assert page["next_offset"] == 25


def test_list_artifacts_caps_the_limit(monkeypatch):
    """The cap has to stay at the 200 ceiling of AGENTS.md 4.1."""
    import asyncio

    import open_notebook_mcp.server as server

    rows = [{"id": f"artifact:{i}"} for i in range(300)]

    with _mock_api(monkeypatch) as mock:
        mock.get(f"{API_BASE}/api/artifacts").mock(
            return_value=Response(200, json=rows)
        )
        page = asyncio.run(server.list_artifacts(limit=200))

    assert len(page["artifacts"]) == 200
    assert page["limit"] == 200


def test_list_artifacts_closes_the_cursor(monkeypatch):
    """Paging has to end: no next_offset once the list is exhausted."""
    import asyncio

    import open_notebook_mcp.server as server

    rows = [{"id": f"artifact:{i}"} for i in range(60)]

    with _mock_api(monkeypatch) as mock:
        mock.get(f"{API_BASE}/api/artifacts").mock(
            return_value=Response(200, json=rows)
        )
        page = asyncio.run(server.list_artifacts(limit=25, offset=50))

    assert [a["id"] for a in page["artifacts"]] == [f"artifact:{i}" for i in range(50, 60)]
    assert page["next_offset"] is None


def test_list_artifacts_shrinks_rows_on_request(monkeypatch):
    """detail and fields exist to keep the model's context small."""
    import asyncio

    import open_notebook_mcp.server as server

    row = {
        "id": "artifact:1",
        "notebook_id": "notebook:1",
        "title": "Report",
        "kind": "report",
        "variant": "document",
        "job_status": "done",
        "created": "2026-01-01",
        "files": ["report.md"],
    }

    with _mock_api(monkeypatch) as mock:
        mock.get(f"{API_BASE}/api/artifacts").mock(
            return_value=Response(200, json=[row])
        )
        full = asyncio.run(server.list_artifacts(detail="full"))["artifacts"][0]
        name = asyncio.run(server.list_artifacts(detail="name"))["artifacts"][0]
        summary = asyncio.run(server.list_artifacts(detail="summary"))["artifacts"][0]
        picked = asyncio.run(server.list_artifacts(fields=["kind", "id"]))["artifacts"][0]

    assert full == row
    assert name == {"id": "artifact:1", "title": "Report"}
    assert "id" in summary and "job_status" in summary and "files" not in summary
    assert picked == {"kind": "report", "id": "artifact:1"}


def test_ask_tools_translate_a_network_failure_into_a_structured_error(monkeypatch):
    """A raw HTTPX error must not escape the ask tools."""
    import asyncio

    import httpx
    import pytest

    from open_notebook_mcp import server

    with _mock_api(monkeypatch) as mock:
        mock.post(f"{API_BASE}/api/search/ask").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with pytest.raises(server.ToolApiError) as excinfo:
            asyncio.run(
                server.ask_simple(
                    question="Question",
                    strategy_model="model:1",
                    answer_model="model:1",
                    final_answer_model="model:1",
                )
            )

    assert excinfo.value.error_code == "network_error"
    assert "connection refused" in excinfo.value.message
    assert excinfo.value.hint
    assert excinfo.value.request_id


def test_ask_tools_translate_a_read_timeout_into_a_structured_error(monkeypatch):
    """A raw HTTPX timeout must not escape the ask tools."""
    import asyncio

    import httpx
    import pytest

    from open_notebook_mcp import server

    with _mock_api(monkeypatch) as mock:
        mock.post(f"{API_BASE}/api/search/ask").mock(
            side_effect=httpx.ReadTimeout("read timed out")
        )
        with pytest.raises(server.ToolTimeoutError) as excinfo:
            asyncio.run(
                server.ask_simple(
                    question="Question",
                    strategy_model="model:1",
                    answer_model="model:1",
                    final_answer_model="model:1",
                )
            )

    assert excinfo.value.error_code == "timeout"
    assert excinfo.value.hint


def test_list_artifacts_publishes_its_paging_knobs():
    """The capability index has to describe the knobs the tool really has."""
    capability = next(cap for cap in CAPABILITIES if cap.name == "list_artifacts")
    for key in ("notebook_id", "limit", "offset", "fields", "detail"):
        assert key in capability.args, key


def test_list_sources_publishes_its_paging_knobs():
    """The capability index has to describe the knobs the tool really has."""
    capability = next(cap for cap in CAPABILITIES if cap.name == "list_sources")
    for key in ("notebook_id", "limit", "offset", "fields", "detail"):
        assert key in capability.args, key


def test_discover_sources_publishes_its_projection_knobs():
    """Discovery has no cursor upstream, but it does shrink rows on request."""
    capability = next(cap for cap in CAPABILITIES if cap.name == "discover_sources")
    for key in ("limit", "fields", "detail"):
        assert key in capability.args, key


def test_list_sources_forwards_the_page_to_the_api(monkeypatch):
    """The API pages for real, so limit and offset travel as query params."""
    import asyncio

    import open_notebook_mcp.server as server

    rows = [{"id": "source:1", "title": "S1", "topics": [], "embedded": True}]

    with _mock_api(monkeypatch) as mock:
        route = mock.get(f"{API_BASE}/api/sources").mock(
            return_value=Response(200, json=rows)
        )
        page = asyncio.run(server.list_sources(limit=10, offset=20))

    params = route.calls[0].request.url.params
    assert params["limit"] == "10"
    assert params["offset"] == "20"
    assert page["count"] == 1
    assert page["offset"] == 20
    assert page["next_offset"] is None


def test_list_sources_keeps_the_cursor_open_on_a_full_page(monkeypatch):
    """A full page may have more behind it, so the caller gets a next_offset."""
    import asyncio

    import open_notebook_mcp.server as server

    rows = [{"id": f"source:{i}"} for i in range(3)]

    with _mock_api(monkeypatch) as mock:
        mock.get(f"{API_BASE}/api/sources").mock(
            return_value=Response(200, json=rows)
        )
        page = asyncio.run(server.list_sources(limit=3, offset=0))

    assert page["next_offset"] == 3


def test_list_sources_shrinks_rows_on_request(monkeypatch):
    """detail and fields exist to keep the model's context small."""
    import asyncio

    import open_notebook_mcp.server as server

    row = {
        "id": "source:1",
        "title": "Paper",
        "topics": ["mcp"],
        "embedded": True,
        "embedded_chunks": 12,
        "insights_count": 2,
        "status": "done",
        "updated": "2026-01-01",
    }

    with _mock_api(monkeypatch) as mock:
        mock.get(f"{API_BASE}/api/sources").mock(
            return_value=Response(200, json=[row])
        )
        full = asyncio.run(server.list_sources(detail="full"))["sources"][0]
        name = asyncio.run(server.list_sources(detail="name"))["sources"][0]
        summary = asyncio.run(server.list_sources(detail="summary"))["sources"][0]
        picked = asyncio.run(server.list_sources(fields=["title"]))["sources"][0]

    assert full == row
    assert name == {"id": "source:1", "title": "Paper"}
    assert "id" in summary and "embedded_chunks" not in summary
    assert picked == {"title": "Paper"}


def test_discover_sources_projects_the_hits(monkeypatch):
    """A hit carries a snippet; only detail='full' should pay for it."""
    import asyncio

    import open_notebook_mcp.server as server

    hit = {
        "title": "MCP spec",
        "url": "https://example.com/mcp",
        "snippet": "long provider excerpt",
        "score": 0.9,
        "source_id": None,
        "status": "candidate",
        "error": None,
    }

    def _payload():
        return {
            "query": "mcp",
            "provider": "tavily",
            "notebook_id": "notebook:1",
            "created_count": 0,
            "skipped_count": 0,
            "results": [dict(hit)],
        }

    with _mock_api(monkeypatch) as mock:
        mock.post(f"{API_BASE}/api/sources/discover").mock(
            return_value=Response(200, json=_payload())
        )
        full = asyncio.run(
            server.discover_sources(
                query="mcp", notebook_id="notebook:1", dry_run=True, detail="full"
            )
        )["result"]["results"][0]
        name = asyncio.run(
            server.discover_sources(
                query="mcp", notebook_id="notebook:1", dry_run=True, detail="name"
            )
        )["result"]["results"][0]
        summary = asyncio.run(
            server.discover_sources(
                query="mcp", notebook_id="notebook:1", dry_run=True
            )
        )["result"]["results"][0]

    assert full == hit
    assert name == {"title": "MCP spec", "url": "https://example.com/mcp"}
    assert "score" in summary and "snippet" not in summary


if __name__ == "__main__":
    # Run tests
    test_capabilities_defined()
    print("✓ test_capabilities_defined passed")

    test_search_capabilities_basic()
    print("✓ test_search_capabilities_basic passed")

    test_search_capabilities_query()
    print("✓ test_search_capabilities_query passed")

    test_search_capabilities_detail_levels()
    print("✓ test_search_capabilities_detail_levels passed")

    test_search_capabilities_limit()
    print("✓ test_search_capabilities_limit passed")

    print("\n✅ All tests passed!")
