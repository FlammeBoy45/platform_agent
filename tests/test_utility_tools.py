import re

import httpx
import pytest

from src.infrastructure.tools.calculator_tool import CalculatorTool
from src.infrastructure.tools.datetime_tool import DateTimeTool
from src.infrastructure.tools.google_search_tool import GoogleSearchTool
from src.infrastructure.tools.python_code_tool import PythonCodeTool
from src.infrastructure.tools.text_stats_tool import TextStatsTool
from src.infrastructure.tools.web_fetch_tool import WebFetchTool


@pytest.mark.asyncio
async def test_calculator_evaluates_safe_math_expression() -> None:
    tool = CalculatorTool()

    result = await tool.execute(expression="2 + 3 * 4")

    assert result == "14"


@pytest.mark.asyncio
async def test_calculator_rejects_unsupported_expression() -> None:
    tool = CalculatorTool()

    result = await tool.execute(expression="__import__('os').system('dir')")

    assert result.startswith("Calcul impossible:")


@pytest.mark.asyncio
async def test_datetime_returns_current_utc_timestamp() -> None:
    tool = DateTimeTool()

    result = await tool.execute(timezone="UTC")

    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", result)


@pytest.mark.asyncio
async def test_datetime_accepts_utc_offset() -> None:
    tool = DateTimeTool()

    result = await tool.execute(timezone="UTC+2")

    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC\+02:00", result)


@pytest.mark.asyncio
async def test_datetime_accepts_common_iana_timezone_on_windows() -> None:
    tool = DateTimeTool()

    result = await tool.execute(timezone="Europe/Paris")

    assert "Timezone inconnue" not in result
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ", result)


@pytest.mark.asyncio
async def test_text_stats_counts_text() -> None:
    tool = TextStatsTool()

    result = await tool.execute(text="bonjour agent\nca va")

    assert result == "Caracteres: 19; Mots: 4; Lignes: 2"


@pytest.mark.asyncio
async def test_python_code_tool_executes_print() -> None:
    tool = PythonCodeTool()

    result = await tool.execute(code="print(2 + 3)")

    assert "Code sortie: 0" in result
    assert "stdout:\n5" in result


@pytest.mark.asyncio
async def test_python_code_tool_reports_runtime_errors() -> None:
    tool = PythonCodeTool()

    result = await tool.execute(code="raise ValueError('boom')")

    assert "Code sortie: 1" in result
    assert "ValueError: boom" in result


@pytest.mark.asyncio
async def test_python_code_tool_times_out_infinite_loops() -> None:
    tool = PythonCodeTool()
    tool.timeout_seconds = 0.2

    result = await tool.execute(code="while True:\n    pass")

    assert "TimeoutError" in result


@pytest.mark.asyncio
async def test_python_code_tool_blocks_attribute_introspection() -> None:
    result = await PythonCodeTool().execute(code="print((1).__class__)")

    assert "accès aux attributs interdit" in result


@pytest.mark.asyncio
async def test_python_code_tool_blocks_dangerous_imports() -> None:
    tool = PythonCodeTool()

    result = await tool.execute(code="import os\nprint(os.getcwd())")

    assert result == "Import interdit: os"


@pytest.mark.asyncio
async def test_web_fetch_rejects_non_http_urls() -> None:
    tool = WebFetchTool()

    result = await tool.execute(url="file:///etc/passwd")

    assert result == "URL refusee: utilise http:// ou https://."


@pytest.mark.asyncio
async def test_web_fetch_reads_html_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def get(self, url: str) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"<html><title>x</title><body><h1>Hello web</h1></body></html>",
                headers={"content-type": "text/html"},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    tool = WebFetchTool()

    result = await tool.execute(url="https://example.com")

    assert "Hello web" in result


@pytest.mark.asyncio
async def test_google_search_uses_duckduckgo_without_serpapi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html = """
    <a class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ffloby">
      Floby - artiste
    </a>
    <a class="result__snippet">Un artiste burkinabè.</a>
    """

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def get(self, url: str, params: dict[str, str]) -> httpx.Response:
            assert url == "https://html.duckduckgo.com/html/"
            assert params == {"q": "Floby"}
            return httpx.Response(
                200,
                text=html,
                request=httpx.Request("GET", url),
            )

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    result = await GoogleSearchTool().execute(query="Floby")

    assert "DuckDuckGo, sans SerpApi" in result
    assert "Floby - artiste" in result
    assert "Un artiste burkinabè." in result
    assert "https://example.com/floby" in result


@pytest.mark.asyncio
async def test_google_search_reports_when_public_search_requires_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def get(self, url: str, params: dict[str, str]) -> httpx.Response:
            return httpx.Response(
                202,
                text="verification required",
                request=httpx.Request("GET", url),
            )

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    result = await GoogleSearchTool().execute(query="Floby")

    assert "DuckDuckGo a demande une verification" in result
