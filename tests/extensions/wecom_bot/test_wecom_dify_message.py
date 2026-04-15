import json
import importlib.util
from pathlib import Path
import sys

from werkzeug.test import EnvironBuilder
from werkzeug.wrappers import Request


REPO_ROOT = Path(__file__).resolve().parents[3]
WECOM_ROOT = REPO_ROOT / "extensions" / "wecom_bot"

sys.path.insert(0, str(WECOM_ROOT / ".venv" / "Lib" / "site-packages"))
sys.path.insert(0, str(WECOM_ROOT))

MODULE_PATH = WECOM_ROOT / "endpoints" / "wecom_dify_message.py"
MODULE_SPEC = importlib.util.spec_from_file_location("wecom_dify_message", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
WECOM_DIFY_MESSAGE_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(WECOM_DIFY_MESSAGE_MODULE)
WeComDifyMessageEndpoint = WECOM_DIFY_MESSAGE_MODULE.WeComDifyMessageEndpoint
WeComCryptor = WECOM_DIFY_MESSAGE_MODULE.WeComCryptor


class FakeStorage:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.data.get(key)

    def set(self, key: str, value: bytes) -> None:
        self.data[key] = value

    def delete(self, key: str) -> None:
        self.data.pop(key, None)

    def exist(self, key: str) -> bool:
        return key in self.data


class FakeSession:
    def __init__(self) -> None:
        self.storage = FakeStorage()


def _make_endpoint() -> WeComDifyMessageEndpoint:
    endpoint = WeComDifyMessageEndpoint.__new__(WeComDifyMessageEndpoint)
    endpoint._upload_wecom_media_to_dify = lambda **kwargs: {
        "type": "image" if kwargs["msgtype"] == "image" else "document",
        "transfer_method": "local_file",
        "upload_file_id": f"upload-{kwargs['index']}",
    }
    return endpoint


def _invoke_endpoint_with_stream_events(events: list[dict]) -> tuple[dict, FakeStorage]:
    endpoint = WeComDifyMessageEndpoint.__new__(WeComDifyMessageEndpoint)
    endpoint.session = FakeSession()
    endpoint._normalize_base_url = lambda base_url: base_url.rstrip("/")
    endpoint._build_dify_query_and_files = lambda **kwargs: ("hello", [])
    endpoint._stream_dify_chat_events = lambda **kwargs: iter(events)

    settings = {
        "token": "test-token",
        "encoding_aes_key": "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
        "api_key": "test-api-key",
        "base_url": "https://example.com/v1",
    }
    payload = {
        "msgid": "agent-msg-test",
        "aibotid": "bot-1",
        "chattype": "single",
        "from": {"userid": "user-1"},
        "msgtype": "text",
        "text": {"content": "hello"},
    }

    cryptor = WeComCryptor(
        token=settings["token"],
        encoding_aes_key=settings["encoding_aes_key"],
    )
    encrypted = cryptor.encrypt_response(
        plain=json.dumps(payload, ensure_ascii=False), timestamp="1", nonce="2"
    )
    builder = EnvironBuilder(
        method="POST",
        path="/dify-api",
        query_string={
            "msg_signature": encrypted["msgsignature"],
            "timestamp": str(encrypted["timestamp"]),
            "nonce": encrypted["nonce"],
        },
        json={"encrypt": encrypted["encrypt"]},
    )
    request = Request(builder.get_environ())
    response = endpoint._invoke(request, {}, settings)

    body = json.loads(response.get_data(as_text=True))
    plain = cryptor.decrypt_echostr(
        signature=body["msgsignature"],
        timestamp=str(body["timestamp"]),
        nonce=body["nonce"],
        echostr=body["encrypt"],
    )
    return (json.loads(plain), endpoint.session.storage)


def test_build_dify_query_and_files_keeps_plain_text() -> None:
    endpoint = _make_endpoint()

    query, files = endpoint._build_dify_query_and_files(
        payload={
            "msgtype": "text",
            "msgid": "plain-text",
            "from": {"userid": "user-1"},
            "text": {"content": "hello"},
        },
        message_id="plain-text",
        cryptor=object(),
        normalized_base_url="https://example.com/v1",
        api_key="test-key",
    )

    assert query == "hello"
    assert files == []


def test_build_dify_query_and_files_uses_quote_file_fallback_query() -> None:
    endpoint = _make_endpoint()

    query, files = endpoint._build_dify_query_and_files(
        payload={
            "msgtype": "text",
            "msgid": "quote-file",
            "from": {"userid": "user-1"},
            "text": {"content": ""},
            "quote": {
                "msgtype": "file",
                "file": {"url": "https://example.com/file"},
            },
        },
        message_id="quote-file",
        cryptor=object(),
        normalized_base_url="https://example.com/v1",
        api_key="test-key",
    )

    assert query == endpoint._default_query_for_msgtype("file")
    assert files == [
        {
            "type": "document",
            "transfer_method": "local_file",
            "upload_file_id": "upload-0",
        }
    ]


def test_build_dify_query_and_files_uses_quote_mixed_fallback_query() -> None:
    endpoint = _make_endpoint()

    query, files = endpoint._build_dify_query_and_files(
        payload={
            "msgtype": "text",
            "msgid": "quote-mixed",
            "from": {"userid": "user-1"},
            "text": {"content": ""},
            "quote": {
                "msgtype": "mixed",
                "mixed": {
                    "msg_item": [
                        {
                            "msgtype": "image",
                            "image": {"url": "https://example.com/image"},
                        }
                    ]
                },
            },
        },
        message_id="quote-mixed",
        cryptor=object(),
        normalized_base_url="https://example.com/v1",
        api_key="test-key",
    )

    assert query == endpoint._default_query_for_msgtype("mixed")
    assert files == [
        {
            "type": "image",
            "transfer_method": "local_file",
            "upload_file_id": "upload-0",
        }
    ]


def test_build_dify_query_and_files_prefixes_quote_text() -> None:
    endpoint = _make_endpoint()

    query, files = endpoint._build_dify_query_and_files(
        payload={
            "msgtype": "text",
            "msgid": "quote-text",
            "from": {"userid": "user-1"},
            "text": {"content": "Question"},
            "quote": {
                "msgtype": "text",
                "text": {"content": "Original context"},
            },
        },
        message_id="quote-text",
        cryptor=object(),
        normalized_base_url="https://example.com/v1",
        api_key="test-key",
    )

    assert query == "Question\nQuoted content: Original context"
    assert files == []


def test_extract_dify_event_conversation_id_reads_metadata_fallback() -> None:
    endpoint = _make_endpoint()

    conversation_id = endpoint._extract_dify_event_conversation_id(
        {
            "event": "message_end",
            "metadata": {"conversation_id": "conv-from-metadata"},
        }
    )

    assert conversation_id == "conv-from-metadata"


def test_format_dify_agent_thought_renders_visible_block() -> None:
    endpoint = _make_endpoint()

    rendered = endpoint._format_dify_agent_thought(
        {
            "event": "agent_thought",
            "position": 1,
            "thought": "先分析天气问题",
            "tool": "天气查询",
            "tool_input": {"city": "马鞍山"},
            "observation": "已获取实时天气",
        }
    )

    assert rendered == "› 已使用 天气查询"


def test_build_dify_agent_thought_key_prefers_position() -> None:
    endpoint = _make_endpoint()

    key = endpoint._build_dify_agent_thought_key(
        {
            "event": "agent_thought",
            "position": 2,
            "thought": "继续分析",
        }
    )

    assert key == "position:2"


def test_render_dify_answer_text_extracts_multiple_think_blocks() -> None:
    endpoint = _make_endpoint()

    think_text, rendered = endpoint._render_dify_answer_text(
        "<think>先分析问题</think><think>再整理答案</think>最终回复"
    )

    assert think_text == "先分析问题\n\n再整理答案"
    assert rendered == "最终回复"


def test_render_dify_answer_text_hides_unclosed_think_block() -> None:
    endpoint = _make_endpoint()

    think_text, rendered = endpoint._render_dify_answer_text(
        "可见内容<think>未结束的思考"
    )

    assert think_text == ""
    assert rendered == "可见内容"


def test_format_tool_observation_summary_strips_json_and_html_noise() -> None:
    endpoint = _make_endpoint()

    summary = endpoint._format_tool_observation_summary(
        '{"天气查询（UI）":{"result":"南京：当前温度21°，阴\\n<p>今天 14/21℃ 多云转小雨</p>"}}'
    )

    assert summary == "南京：当前温度21°，阴 今天 14/21℃ 多云转小雨"


def test_invoke_consumes_agent_message_and_message_replace() -> None:
    response_payload, storage = _invoke_endpoint_with_stream_events(
        [
            {
                "event": "agent_thought",
                "position": 1,
                "conversation_id": "conv-agent-1",
                "thought": "tool calling",
                "tool": "weather.lookup",
                "tool_input": {"city": "Maanshan"},
                "observation": "晴，21C",
            },
            {
                "event": "agent_thought",
                "position": 2,
                "conversation_id": "conv-agent-1",
                "thought": "summarizing",
                "tool": "weather.format",
            },
            {
                "event": "agent_message",
                "answer": "<think>先查询天气</think>draft reply",
                "conversation_id": "conv-agent-1",
            },
            {
                "event": "message_replace",
                "answer": "<think>先查询天气</think><think>再整理结果</think>final reply",
                "conversation_id": "conv-agent-1",
            },
            {
                "event": "message_end",
                "metadata": {"conversation_id": "conv-agent-1"},
            },
        ]
    )

    assert response_payload["stream"]["content"] == (
        "› 已使用 weather.lookup\n\n"
        "› 已使用 weather.format\n\n"
        "> 深度思考\n"
        "> 先查询天气\n"
        "> 再整理结果\n\n"
        "final reply"
    )
    assert any(value == b"conv-agent-1" for value in storage.data.values())
