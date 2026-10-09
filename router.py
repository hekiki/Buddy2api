"""Bind /v1 requests to a single channel. Does not construct upstream HTTP clients."""

from __future__ import annotations

import copy
import json
from typing import Optional

from fastapi import HTTPException

import providers
import responses
from reasoning_controls import normalize_chat_reasoning
from model_capacity import clamp_output_tokens
from providers.protocol import (
    BindResult,
    InvalidModel,
    KeyChannelMismatch,
    KNOWN_CHANNEL_SET,
    UnknownChannel,
    UnknownModel,
)


def _key_channel(api_key_info: dict | None) -> str:
    if not api_key_info:
        return "workbuddy"
    value = str(api_key_info.get("default_channel") or "workbuddy").strip()
    return value or "workbuddy"


def _other_channel_ids(inner: str, except_channel: str) -> bool:
    for channel in providers.enabled_provider_ids():
        if channel == except_channel:
            continue
        provider = providers.get_provider(channel)
        if provider and provider.accepts_model(inner):
            return True
    return False


def bind(payload: dict, api_key_info: dict | None) -> BindResult:
    original = payload.get("model", "auto")
    if original is None or original == "":
        original = "auto"
    if not isinstance(original, str):
        raise InvalidModel(str(original), "model must be a string")
    original = original.strip() or "auto"
    key_channel = _key_channel(api_key_info)

    first, sep, rest = original.partition("/")
    if sep and first in KNOWN_CHANNEL_SET:
        channel = first
        inner = rest
        if not inner:
            raise InvalidModel(original, f"model '{original}' is missing the inner id")
        if not providers.is_channel_enabled(channel) or providers.get_provider(channel) is None:
            raise UnknownChannel(channel)
        provider = providers.get_provider(channel)
        if not provider.accepts_model(inner):
            raise InvalidModel(original)
        if key_channel != channel:
            raise KeyChannelMismatch(channel, key_channel)
        return BindResult(channel=channel, inner=inner, original=original)

    # Unprefixed: skip step 2, bind key.default_channel
    channel = key_channel
    if not providers.is_channel_enabled(channel) or providers.get_provider(channel) is None:
        raise UnknownChannel(channel)
    provider = providers.get_provider(channel)
    inner = original
    if inner == "auto" or provider.accepts_model(inner):
        return BindResult(channel=channel, inner=inner, original=original)
    if _other_channel_ids(inner, channel):
        raise UnknownModel(
            original,
            f"Model '{original}' belongs to another channel; "
            f"switch this API key's channel or send a namespaced id",
        )
    raise UnknownModel(original)


def bind_http(payload: dict, api_key_info: dict | None) -> BindResult:
    try:
        return bind(payload, api_key_info)
    except UnknownChannel as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "unknown_channel",
                    "code": "unknown_channel",
                    "channel": exc.channel,
                }
            },
        ) from exc
    except KeyChannelMismatch as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "key_channel_mismatch",
                    "code": "key_channel_mismatch",
                    "channel": exc.channel,
                    "key_channel": exc.key_channel,
                }
            },
        ) from exc
    except InvalidModel as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "invalid_model",
                    "code": "invalid_model",
                }
            },
        ) from exc
    except UnknownModel as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "unknown_model",
                    "code": "unknown_model",
                }
            },
        ) from exc


async def ensure_usable(channel: str) -> None:
    provider = providers.get_provider(channel)
    if provider is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Unknown or disabled channel '{channel}'",
                    "type": "unknown_channel",
                    "code": "unknown_channel",
                    "channel": channel,
                }
            },
        )
    if await provider.has_usable_account():
        return
    raise HTTPException(
        status_code=503,
        detail={
            "error": {
                "message": f"No usable accounts for channel '{channel}'",
                "type": "channel_unavailable",
                "code": "channel_unavailable",
                "channel": channel,
            }
        },
    )


def _estimate_text_tokens(text: str) -> int:
    """Count CJK and fullwidth characters as one token; other text as about four chars."""
    cjk = 0
    other = 0
    for char in text:
        code = ord(char)
        if (
            0x3400 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0x3000 <= code <= 0x303F
            or 0xFF00 <= code <= 0xFFEF
        ):
            cjk += 1
        else:
            other += 1
    return cjk + (other + 3) // 4


def _estimate_value(value) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return _estimate_text_tokens(value)
    if isinstance(value, list):
        return sum(_estimate_value(item) for item in value)
    if isinstance(value, dict):
        return sum(_estimate_value(key) + _estimate_value(item) for key, item in value.items())
    return _estimate_text_tokens(str(value))


def estimate_prompt_tokens(payload: dict) -> int:
    """Rough prompt size used only for the Codex pre-flight gate."""
    return max(1, _estimate_value(payload))


CODEX_PROMPT_TOKEN_CAP = 500000


def codex_prompt_limit(channel: str, model: str) -> tuple[int, int]:
    """Return the Codex-only input budget and the window used to derive it."""
    from model_capacity import capacity_fields
    import catalog

    row = next(
        (item for item in catalog.current_models(channel) if item.get("id") == model),
        {},
    )
    confirmed = capacity_fields(row).get("context_window")
    window = min(confirmed, CODEX_PROMPT_TOKEN_CAP) if confirmed else CODEX_PROMPT_TOKEN_CAP
    return max(1, window - 32768), window


def _is_system_item(item) -> bool:
    return isinstance(item, dict) and item.get("role") in ("system", "developer")


def _is_tool_carrier(item) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") in ("function_call", "function_call_output"):
        return True
    if item.get("role") == "tool":
        return True
    return item.get("role") == "assistant" and bool(item.get("tool_calls"))


def _history_segments(items: list) -> list[list]:
    """Keep a tool call and its results in one segment so trimming cannot split them."""
    segments = []
    index = 0
    while index < len(items):
        item = items[index]
        if _is_tool_carrier(item) or (isinstance(item, dict) and item.get("type") == "reasoning"):
            group = []
            while index < len(items) and (
                _is_tool_carrier(items[index])
                or (isinstance(items[index], dict) and items[index].get("type") == "reasoning")
            ):
                group.append(items[index])
                index += 1
            segments.append(group)
            continue
        segments.append([item])
        index += 1
    return segments


def _trim_history(payload: dict, key: str, limit: int) -> list:
    """Drop oldest history until the whole payload fits, keeping the newest segment."""
    segments = _history_segments(payload[key])
    while estimate_prompt_tokens(payload) > limit and len(segments) > 1:
        drop_at = next(
            (
                index
                for index, segment in enumerate(segments[:-1])
                if not all(_is_system_item(item) for item in segment)
            ),
            None,
        )
        if drop_at is None:
            drop_at = 0
        segments.pop(drop_at)
        payload[key] = [item for segment in segments for item in segment]
    return payload[key]


def trim_codex_prompt(payload: dict, channel: str, model: str) -> dict:
    """Trim old Codex history so the current turn can continue."""
    limit, window = codex_prompt_limit(channel, model)
    if estimate_prompt_tokens(payload) <= limit:
        return payload
    trimmed = copy.deepcopy(payload)
    if isinstance(trimmed.get("input"), list):
        _trim_history(trimmed, "input", limit)
    elif isinstance(trimmed.get("messages"), list):
        _trim_history(trimmed, "messages", limit)
    if estimate_prompt_tokens(trimmed) <= limit:
        return trimmed
    raise HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": (
                    f"Codex prompt is too long for {model}: the newest request still exceeds "
                    f"the {limit}-token Codex budget ({window}-token window) after trimming old history."
                ),
                "type": "invalid_request_error",
                "code": "prompt_too_long",
                "param": "input",
            }
        },
    )


def dispatch_payload(payload: dict, inner: str) -> dict:
    body = copy.copy(payload)
    body["model"] = inner
    return body


def _rewrite_json_model(obj, original: str):
    if isinstance(obj, dict):
        rewritten = None
        if "model" in obj and isinstance(obj["model"], str):
            rewritten = dict(obj)
            rewritten["model"] = original
        if isinstance(obj.get("response"), dict):
            if rewritten is None:
                rewritten = dict(obj)
            rewritten["response"] = _rewrite_json_model(obj["response"], original)
        if rewritten is not None:
            obj = rewritten
        return obj
    return obj


async def _rewrite_stream_model(stream, original: str):
    async for chunk in stream:
        if not isinstance(chunk, (bytes, bytearray, str)):
            yield chunk
            continue
        text = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else chunk
        rewritten = []
        for line in text.splitlines(keepends=True):
            raw = line[:-1] if line.endswith("\n") else line
            ended = line.endswith("\n")
            if raw.startswith("data:") and raw[5:].strip() not in {"", "[DONE]"}:
                data = raw[5:]
                if data.startswith(" "):
                    data = data[1:]
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    rewritten.append(line)
                    continue
                parsed = _rewrite_json_model(parsed, original)
                new_line = "data: " + json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                rewritten.append(new_line + ("\n" if ended else ""))
            else:
                rewritten.append(line)
        out = "".join(rewritten)
        yield out.encode("utf-8") if isinstance(chunk, (bytes, bytearray)) else out


async def echo_original(result: tuple, original: str) -> tuple:
    kind = result[0]
    if kind == "json" and isinstance(result[1], dict):
        body = dict(result[1])
        if "model" in body:
            body["model"] = original
        return ("json", body)
    if kind == "stream":
        return ("stream", _rewrite_stream_model(result[1], original))
    return result


async def chat_after_bind(
    bound: BindResult, payload: dict, api_key_info: dict | None
) -> tuple:
    result = await _chat_after_bind_no_echo(bound, payload, api_key_info)
    return await echo_original(result, bound.original)


async def _chat_after_bind_no_echo(
    bound: BindResult, payload: dict, api_key_info: dict | None
) -> tuple:
    provider = providers.get_provider(bound.channel)
    if provider is None:
        raise UnknownChannel(bound.channel)
    inner = provider.translate_model(bound.inner)
    dispatch = normalize_chat_reasoning(dispatch_payload(payload, inner))
    model = next((item for item in provider.list_models()
                  if isinstance(item, dict) and item.get("id") == inner), None)
    dispatch = clamp_output_tokens(dispatch, model)
    info = dict(api_key_info or {})
    info["_log_model"] = bound.original
    info["_bind_channel"] = bound.channel
    return await provider.chat_completions(dispatch, info)


async def responses_after_bind(
    bound: BindResult, payload: dict, api_key_info: dict | None
) -> tuple:
    """Bridge Responses through the provider selected by the existing bind."""
    dispatch = dispatch_payload(payload, bound.inner)

    async def chat_handler(chat_payload: dict, _api_key_info: dict | None) -> tuple:
        return await _chat_after_bind_no_echo(bound, chat_payload, api_key_info)

    result = await responses.proxy_responses(
        dispatch,
        api_key_info,
        chat_handler=chat_handler,
    )
    return await echo_original(result, bound.original)
