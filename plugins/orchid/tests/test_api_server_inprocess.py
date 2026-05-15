from fastapi.testclient import TestClient

from orchid.llmscheduler.server.config import ServerConfig
from orchid.llmscheduler.server.app_factory import create_app


def _test_config() -> ServerConfig:
    return ServerConfig(
        host="127.0.0.1",
        port=0,
        test_mode=True,
        model_path="",
        tokenizer_path="",
        use_fp16=True,
        engine_path=None,
        num_layers=None,
        num_heads=None,
        kv_num_heads=None,
        head_dim=None,
        page_size=16,
        max_pages=None,
    )


def test_chat_completions_test_mode():
    app = create_app(_test_config())
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ready"] is True

    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": False, "max_tokens": 4}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["choices"][0]["message"]["content"] == "ok"


def test_chat_completions_stream_test_mode():
    app = create_app(_test_config())
    client = TestClient(app)
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 4}
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        assert r.status_code == 200
        chunks = [line for line in r.iter_lines() if line]
    assert chunks[0].startswith("data: ")
    assert '"content":"ok"' in chunks[0]
    assert '"finish_reason":"stop"' in chunks[1]
    assert chunks[-1] == "data: [DONE]"
