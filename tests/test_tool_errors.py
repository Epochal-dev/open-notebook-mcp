"""Tests for the structured error contract and the input validation.

AGENTS.md section 6 asks every tool to validate inputs early and to fail with
an error carrying `error_code`, `message`, `hint` and `request_id`.
"""

import asyncio
import json

import pytest
import respx
from httpx import Response

from open_notebook_mcp import server

#: The host the tools are pointed at during tests, so respx can intercept.
API_BASE = "http://api.test"


def _mock_api(monkeypatch):
    """Point the server at a test host so respx can intercept.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        A respx mock router. With no route registered, any HTTP call raises,
        so a test that expects a local rejection proves nothing was sent.
    """
    monkeypatch.setenv("OPEN_NOTEBOOK_URL", API_BASE)
    return respx.mock(assert_all_called=True)


def test_tool_error_carries_the_agenda_fields():
    """The error payload has to name the code, the hint and the request."""
    error = server.ToolError(
        "invalid_input", "query is required", "Pass a topic to search for."
    )

    payload = error.to_dict()

    assert payload["error_code"] == "invalid_input"
    assert payload["message"] == "query is required"
    assert payload["hint"] == "Pass a topic to search for."
    assert payload["request_id"]
    assert "invalid_input" in str(error)
    assert payload["request_id"] in str(error)


def test_tool_input_error_is_a_value_error():
    """Callers that catch ValueError keep working with the structured error."""
    assert issubclass(server.ToolInputError, ValueError)
    assert issubclass(server.ToolApiError, RuntimeError)
    assert issubclass(server.ToolTimeoutError, TimeoutError)


def test_discover_sources_rejects_a_blank_query(monkeypatch):
    """A blank topic must fail before any request leaves the process."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.discover_sources(query="   ", notebook_id="notebook:1"))

    assert excinfo.value.error_code == "invalid_input"
    assert excinfo.value.hint


def test_discover_sources_rejects_a_blank_notebook_id(monkeypatch):
    """The notebook is where the sources land: it cannot be empty."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.discover_sources(query="mcp", notebook_id=""))

    assert excinfo.value.error_code == "invalid_input"


@pytest.mark.parametrize("limit", [0, -1, 21, 1000])
def test_discover_sources_rejects_a_limit_out_of_range(monkeypatch, limit):
    """The API accepts 1 to 20 sources, so anything else is a local error."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(
            server.discover_sources(query="mcp", notebook_id="notebook:1", limit=limit)
        )

    assert excinfo.value.error_code == "invalid_input"
    assert "limit" in excinfo.value.message


def test_discover_sources_sends_the_validated_payload(monkeypatch):
    """A valid call must reach the API with the trimmed values."""
    with _mock_api(monkeypatch) as mock:
        route = mock.post(f"{API_BASE}/api/sources/discover").mock(
            return_value=Response(200, json={"created_count": 0, "results": []})
        )
        result = asyncio.run(
            server.discover_sources(
                query="  mcp protocol  ",
                notebook_id="notebook:1",
                limit=3,
                provider="tavily",
                exclude_domains=["example.com"],
            )
        )

    payload = json.loads(route.calls[0].request.content)
    assert payload["query"] == "mcp protocol"
    assert payload["notebook_id"] == "notebook:1"
    assert payload["limit"] == 3
    assert payload["provider"] == "tavily"
    assert payload["exclude_domains"] == ["example.com"]
    assert result["result"]["created_count"] == 0


def test_discover_sources_rejects_a_query_above_the_upstream_limit(monkeypatch):
    """The API caps the topic at 400 characters, so we fail before the call."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.discover_sources(query="x" * 401, notebook_id="notebook:1"))

    assert excinfo.value.error_code == "invalid_input"
    assert "query" in excinfo.value.message


def test_generate_artifact_rejects_an_unknown_kind(monkeypatch):
    """An unknown kind would silently become a report upstream."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.generate_artifact(notebook_id="notebook:1", kind="poem"))

    assert excinfo.value.error_code == "invalid_input"
    assert "kind" in excinfo.value.message


def test_generate_artifact_rejects_a_variant_of_the_other_kind(monkeypatch):
    """A report has no 'presenter' variant: the API would answer 400."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(
            server.generate_artifact(
                notebook_id="notebook:1", kind="report", variant="presenter"
            )
        )

    assert excinfo.value.error_code == "invalid_input"
    assert "variant" in excinfo.value.message


def test_generate_artifact_rejects_an_unsupported_format(monkeypatch):
    """Only the rendered formats are accepted."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(
            server.generate_artifact(notebook_id="notebook:1", formats=["md", "pdf"])
        )

    assert excinfo.value.error_code == "invalid_input"
    assert "pdf" in excinfo.value.message


@pytest.mark.parametrize("sections", [0, -3, 500])
def test_generate_artifact_rejects_out_of_range_sections(monkeypatch, sections):
    """A document has a bounded number of sections."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(
            server.generate_artifact(notebook_id="notebook:1", sections=sections)
        )

    assert excinfo.value.error_code == "invalid_input"
    assert "sections" in excinfo.value.message


def test_generate_artifact_sends_the_validated_payload(monkeypatch):
    """The default formats and the chosen variant must reach the API."""
    with _mock_api(monkeypatch) as mock:
        route = mock.post(f"{API_BASE}/api/artifacts").mock(
            return_value=Response(200, json={"artifact_id": "artifact:1"})
        )
        asyncio.run(
            server.generate_artifact(
                notebook_id="notebook:1",
                kind="slides",
                variant="Detailed",
                formats=["md", "pptx"],
                sections=3,
            )
        )

    payload = json.loads(route.calls[0].request.content)
    assert payload["kind"] == "slides"
    assert payload["variant"] == "detailed"
    assert payload["formats"] == ["md", "pptx"]
    assert payload["sections"] == 3


def test_list_artifacts_rejects_a_limit_above_the_ceiling(monkeypatch):
    """AGENTS.md 4.1 caps a page at 200 items."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.list_artifacts(limit=201))

    assert excinfo.value.error_code == "invalid_input"
    assert "limit" in excinfo.value.message


def test_list_artifacts_rejects_a_negative_offset(monkeypatch):
    """Paging starts at zero."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.list_artifacts(offset=-1))

    assert excinfo.value.error_code == "invalid_input"
    assert "offset" in excinfo.value.message


def test_list_artifacts_rejects_an_unknown_detail(monkeypatch):
    """Only the documented detail levels are accepted."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.list_artifacts(detail="everything"))

    assert excinfo.value.error_code == "invalid_input"
    assert "detail" in excinfo.value.message


def test_get_artifact_rejects_a_blank_id(monkeypatch):
    """An empty ID would ask the API for /api/artifacts/."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.get_artifact(artifact_id="  "))

    assert excinfo.value.error_code == "invalid_input"


def test_delete_artifact_rejects_a_blank_id(monkeypatch):
    """An empty ID must not reach a DELETE."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.delete_artifact(artifact_id=""))

    assert excinfo.value.error_code == "invalid_input"


def test_create_source_rejects_a_link_without_url(monkeypatch):
    """A link source without a url is a structured input error."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.create_source(notebook_id="notebook:1", type="link"))

    assert excinfo.value.error_code == "invalid_input"
    assert excinfo.value.hint


def test_create_source_rejects_a_blank_url(monkeypatch):
    """A whitespace-only url is as empty as a missing one."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(
            server.create_source(notebook_id="notebook:1", type="link", url="   ")
        )

    assert excinfo.value.error_code == "invalid_input"
    assert "url" in excinfo.value.message


def test_create_source_rejects_a_blank_content(monkeypatch):
    """A whitespace-only text is as empty as a missing one."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(
            server.create_source(notebook_id="notebook:1", type="text", content="  \n ")
        )

    assert excinfo.value.error_code == "invalid_input"
    assert "content" in excinfo.value.message


def test_create_source_sends_the_validated_payload(monkeypatch):
    """The trimmed values and the async flag must reach the API."""
    with _mock_api(monkeypatch) as mock:
        route = mock.post(f"{API_BASE}/api/sources/json").mock(
            return_value=Response(200, json={"id": "source:1"})
        )
        asyncio.run(
            server.create_source(
                notebook_id="  notebook:1  ",
                type="text",
                content="  hello  ",
                title="  Title  ",
                async_processing=True,
            )
        )

    payload = json.loads(route.calls[0].request.content)
    assert payload["notebook_id"] == "notebook:1"
    assert payload["content"] == "hello"
    assert payload["title"] == "Title"
    assert payload["async_processing"] is True


def test_generate_artifact_rejects_a_blank_title(monkeypatch):
    """An empty title override would create an unnamed artifact."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.generate_artifact(notebook_id="notebook:1", title="  "))

    assert excinfo.value.error_code == "invalid_input"
    assert "title" in excinfo.value.message


def test_api_failures_become_structured_errors(monkeypatch):
    """An HTTP failure has to carry the code, the hint and the request id."""
    with _mock_api(monkeypatch) as mock:
        mock.get(f"{API_BASE}/api/artifacts/artifact:1").mock(
            return_value=Response(500, json={"detail": "boom"})
        )
        with pytest.raises(server.ToolApiError) as excinfo:
            asyncio.run(server.get_artifact(artifact_id="artifact:1"))

    error = excinfo.value
    assert error.error_code == "api_error"
    assert "boom" in error.message
    assert error.hint
    assert error.request_id
    assert "boom" in str(error)


def test_api_failures_on_discovery_become_structured_errors(monkeypatch):
    """A refused discovery call has to say what to do next."""
    with _mock_api(monkeypatch) as mock:
        mock.post(f"{API_BASE}/api/sources/discover").mock(
            return_value=Response(404, json={"detail": "not found"})
        )
        with pytest.raises(server.ToolApiError) as excinfo:
            asyncio.run(server.discover_sources(query="mcp", notebook_id="notebook:1"))

    assert excinfo.value.error_code == "api_error"
    assert "not found" in excinfo.value.message
    assert excinfo.value.request_id


def test_discover_sources_rejects_too_many_excluded_domains(monkeypatch):
    """The API caps the denylist at 50 entries, so we fail before the call."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(
            server.discover_sources(
                query="mcp",
                notebook_id="notebook:1",
                exclude_domains=[f"d{i}.example" for i in range(51)],
            )
        )

    assert excinfo.value.error_code == "invalid_input"
    assert "exclude_domains" in excinfo.value.message


def test_discover_sources_lowercases_the_excluded_domains(monkeypatch):
    """Hostnames are case-insensitive, so the denylist is normalized."""
    with _mock_api(monkeypatch) as mock:
        route = mock.post(f"{API_BASE}/api/sources/discover").mock(
            return_value=Response(200, json={"created_count": 0, "results": []})
        )
        asyncio.run(
            server.discover_sources(
                query="mcp",
                notebook_id="notebook:1",
                exclude_domains=["Example.COM"],
                dry_run=True,
            )
        )

    payload = json.loads(route.calls[0].request.content)
    assert payload["exclude_domains"] == ["example.com"]


def test_list_sources_rejects_a_limit_above_the_api_page(monkeypatch):
    """The API answers 422 above 100, so the wrapper fails early instead."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.list_sources(limit=101))

    assert excinfo.value.error_code == "invalid_input"
    assert "limit" in excinfo.value.message


def test_list_sources_rejects_a_negative_offset(monkeypatch):
    """Paging starts at zero."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.list_sources(offset=-1))

    assert excinfo.value.error_code == "invalid_input"
    assert "offset" in excinfo.value.message


def test_list_sources_rejects_an_unknown_detail(monkeypatch):
    """Only the documented detail levels are accepted."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.list_sources(detail="everything"))

    assert excinfo.value.error_code == "invalid_input"
    assert "detail" in excinfo.value.message


def test_list_sources_rejects_a_blank_field_name(monkeypatch):
    """A blank projection key would silently drop out of the row."""
    with (
        _mock_api(monkeypatch),
        pytest.raises(server.ToolInputError) as excinfo,
    ):
        asyncio.run(server.list_sources(fields=["id", "  "]))

    assert excinfo.value.error_code == "invalid_input"
    assert "fields" in excinfo.value.message
