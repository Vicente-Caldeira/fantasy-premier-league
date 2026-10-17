import httpx
import pytest

from fpl_core.fpl_api import FplApiClient, FplApiError


def make_client(handler, **kwargs):
    sleeps: list[float] = []
    client = FplApiClient(
        transport=httpx.MockTransport(handler), sleep=sleeps.append, backoff_seconds=1, **kwargs
    )
    return client, sleeps


def responder(*responses):
    calls = iter(responses)

    def handler(request):
        status, body, headers = next(calls)
        return httpx.Response(status, json=body, headers=headers)

    return handler


def test_returns_json_on_success():
    client, sleeps = make_client(responder((200, {"ok": True}, {})))
    assert client.get_json("/bootstrap-static/") == {"ok": True}
    assert sleeps == []


def test_retries_transient_errors_with_exponential_backoff():
    client, sleeps = make_client(
        responder((503, {}, {}), (502, {}, {}), (500, {}, {}), (200, [1], {}))
    )
    assert client.get_json("/fixtures/") == [1]
    assert sleeps == [1, 2, 4]


def test_honours_retry_after_on_rate_limit():
    client, sleeps = make_client(responder((429, {}, {"Retry-After": "7"}), (200, {}, {})))
    client.get_json("/fixtures/")
    assert sleeps == [7]


def test_gives_up_after_max_retries():
    client, sleeps = make_client(responder(*[(503, {}, {})] * 3), max_retries=2)
    with pytest.raises(FplApiError, match="HTTP 503 after 2 retries"):
        client.get_json("/fixtures/")
    assert len(sleeps) == 2


def test_does_not_retry_client_errors():
    client, sleeps = make_client(responder((404, {}, {})))
    with pytest.raises(FplApiError, match="HTTP 404"):
        client.get_json("/element-summary/99999/")
    assert sleeps == []


def test_retries_transport_errors():
    attempts = iter([httpx.ConnectError("boom"), None])

    def handler(request):
        if (err := next(attempts)) is not None:
            raise err
        return httpx.Response(200, json={"ok": 1})

    client, sleeps = make_client(handler)
    assert client.get_json("/x/") == {"ok": 1}
    assert sleeps == [1]


def test_sends_identifying_user_agent():
    seen = {}

    def handler(request):
        seen["ua"] = request.headers["User-Agent"]
        return httpx.Response(200, json={})

    client, _ = make_client(handler)
    client.get_json("/x/")
    assert "fpl-xpts" in seen["ua"]


def test_element_summaries_fetches_each_player_once():
    requested = []

    def handler(request):
        requested.append(request.url.path)
        element_id = int(request.url.path.rstrip("/").split("/")[-1])
        return httpx.Response(200, json={"history": [], "id": element_id})

    client, _ = make_client(handler)
    result = client.element_summaries([3, 1, 2])
    assert {k: v["id"] for k, v in result.items()} == {3: 3, 1: 1, 2: 2}
    assert sorted(requested) == [f"/api/element-summary/{i}/" for i in (1, 2, 3)]
