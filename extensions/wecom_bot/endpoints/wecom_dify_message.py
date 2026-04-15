from collections.abc import Mapping
import hashlib
import ipaddress
import json
import logging
import socket
from typing import Any, Iterator
from urllib import error, parse, request

from werkzeug import Request, Response

from dify_plugin import Endpoint
from dify_plugin.config.logger_format import plugin_logger_handler

from utils import crypto as wecom_crypto
from utils.crypto import WeComCryptor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(plugin_logger_handler)


class WeComDifyMessageEndpoint(Endpoint):
    _DIFY_TIMEOUT = 30
    _RECENT_IDS_KEY = "wecom_dify_recent_msgs"

    def _state_key(self, message_id: str) -> str:
        return f"wecom_dify_msg_state_{message_id}"

    def _content_key(self, message_id: str) -> str:
        return f"wecom_dify_msg_content_{message_id}"

    def _is_forbidden_ip(self, address: str) -> bool:
        ip = ipaddress.ip_address(address)
        return any(
            [
                ip.is_private,
                ip.is_loopback,
                ip.is_link_local,
                ip.is_multicast,
                ip.is_reserved,
                ip.is_unspecified,
            ]
        )

    def _validate_public_hostname(self, hostname: str) -> None:
        if not hostname:
            raise ValueError("base url must include a host")
        if hostname.lower() == "localhost":
            raise ValueError("base url must not target localhost")

        try:
            direct_ip = ipaddress.ip_address(hostname)
        except ValueError:
            direct_ip = None

        if direct_ip is not None:
            if self._is_forbidden_ip(str(direct_ip)):
                raise ValueError(
                    "base url must not target private or reserved addresses"
                )
            return

        try:
            addr_infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            raise ValueError(f"base url host is not resolvable: {hostname}") from exc

        resolved_addresses = {str(info[4][0]) for info in addr_infos}
        if not resolved_addresses:
            raise ValueError(f"base url host is not resolvable: {hostname}")

        for address in resolved_addresses:
            if self._is_forbidden_ip(address):
                raise ValueError(
                    "base url must not target private or reserved addresses"
                )

    def _user_safe_error_message(self) -> str:
        return "Request failed, please retry later."

    def _build_wecom_res(
        self,
        message_id: str,
        content: str,
        finish: bool,
        timestamp: str,
        nonce: str,
        cryptor: WeComCryptor,
    ) -> str:
        body = {
            "msgtype": "stream",
            "stream": {
                "id": message_id,
                "finish": finish,
                "content": content,
            },
        }

        encrypted = cryptor.encrypt_response(
            plain=json.dumps(body, ensure_ascii=False),
            timestamp=timestamp,
            nonce=nonce,
        )
        return json.dumps(encrypted, ensure_ascii=False)

    def _normalize_base_url(self, base_url: str) -> str:
        normalized = base_url.strip().rstrip("/")
        if not normalized:
            raise ValueError("missing base url")

        parsed = parse.urlsplit(normalized)
        if parsed.scheme != "https":
            raise ValueError("base url must start with https://")
        if not parsed.netloc:
            raise ValueError("base url must include a host")
        if parsed.query or parsed.fragment:
            raise ValueError("base url must not include query or fragment")

        hostname = parsed.hostname
        if hostname is None:
            raise ValueError("base url must include a valid host")
        self._validate_public_hostname(hostname)

        path = parsed.path.rstrip("/")
        if not path:
            path = "/v1"

        return parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    def _build_dify_endpoint(self, base_url: str) -> str:
        return f"{base_url}/chat-messages"

    def _build_dify_upload_endpoint(self, base_url: str) -> str:
        return f"{base_url}/files/upload"

    def _parse_json_bytes(self, raw: bytes) -> Any:
        if not raw:
            return {}

        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return raw.decode("utf-8", errors="replace")

    def _format_dify_error(self, status_code: int, payload: Any) -> str:
        if isinstance(payload, Mapping):
            code = payload.get("code")
            message = payload.get("message") or payload.get("msg")
            if code and message:
                return f"dify_http_{status_code}:{code}:{message}"
            if message:
                return f"dify_http_{status_code}:{message}"
        elif isinstance(payload, str) and payload:
            return f"dify_http_{status_code}:{payload}"

        return f"dify_http_{status_code}"

    def _download_wecom_media(self, url: str) -> bytes:
        parsed = parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("invalid wecom media url")

        req = request.Request(
            url=url,
            method="GET",
            headers={
                "User-Agent": "dify-wecom-bot/1.0",
            },
        )
        try:
            with request.urlopen(req, timeout=self._DIFY_TIMEOUT) as resp:
                return resp.read()
        except error.HTTPError as exc:
            raise RuntimeError(f"wecom_media_http_{exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"wecom_media_request_failed:{exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"wecom_media_request_failed:{exc}") from exc

    def _decrypt_wecom_media(
        self, encrypted_bytes: bytes, cryptor: WeComCryptor
    ) -> bytes:
        if not encrypted_bytes:
            return b""
        if len(encrypted_bytes) % 16 != 0:
            raise ValueError("invalid encrypted media length")

        cipher = wecom_crypto.AES.new(
            cryptor.key, wecom_crypto.AES.MODE_CBC, cryptor.iv
        )
        plain = cipher.decrypt(encrypted_bytes)
        pad = plain[-1]
        if pad < 1 or pad > 32 or plain[-pad:] != bytes([pad]) * pad:
            raise ValueError("invalid encrypted media padding")
        return plain[:-pad]

    def _detect_file_extension_and_mime(self, content: bytes) -> tuple[str, str]:
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return ("png", "image/png")
        if content.startswith(b"\xff\xd8\xff"):
            return ("jpg", "image/jpeg")
        if content.startswith((b"GIF87a", b"GIF89a")):
            return ("gif", "image/gif")
        if content.startswith(b"BM"):
            return ("bmp", "image/bmp")
        if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
            return ("webp", "image/webp")
        if content.startswith(b"%PDF"):
            return ("pdf", "application/pdf")
        if content.startswith(b"PK\x03\x04"):
            return ("zip", "application/zip")
        if content.startswith(b"ID3") or content[:2] in {
            b"\xff\xfb",
            b"\xff\xf3",
            b"\xff\xf2",
        }:
            return ("mp3", "audio/mpeg")
        if content.startswith(b"RIFF") and content[8:12] == b"WAVE":
            return ("wav", "audio/wav")
        if len(content) > 12 and content[4:8] == b"ftyp":
            return ("mp4", "video/mp4")
        return ("bin", "application/octet-stream")

    def _build_media_filename(
        self,
        *,
        message_id: str,
        index: int,
        msgtype: str,
        extension: str,
    ) -> str:
        suffix = extension.lstrip(".") or "bin"
        return f"{message_id}_{msgtype}_{index}.{suffix}"

    def _encode_multipart_form_data(
        self,
        *,
        field_name: str,
        filename: str,
        content: bytes,
        mime_type: str,
        user: str,
    ) -> tuple[str, bytes]:
        boundary = hashlib.sha1(
            f"{filename}|{len(content)}|{user}".encode("utf-8")
        ).hexdigest()
        body = bytearray()
        crlf = b"\r\n"

        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="user"\r\n\r\n{user}\r\n'.encode(
                "utf-8"
            )
        )
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(crlf)
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        return (f"multipart/form-data; boundary={boundary}", bytes(body))

    def _upload_dify_file(
        self,
        *,
        base_url: str,
        api_key: str,
        user: str,
        filename: str,
        content: bytes,
        mime_type: str,
    ) -> str:
        content_type, body = self._encode_multipart_form_data(
            field_name="file",
            filename=filename,
            content=content,
            mime_type=mime_type,
            user=user,
        )
        req = request.Request(
            url=self._build_dify_upload_endpoint(base_url),
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": content_type,
            },
        )

        try:
            with request.urlopen(req, timeout=self._DIFY_TIMEOUT) as resp:
                payload = self._parse_json_bytes(resp.read())
        except error.HTTPError as exc:
            raise RuntimeError(
                self._format_dify_error(exc.code, self._parse_json_bytes(exc.read()))
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"dify_file_upload_failed:{exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"dify_file_upload_failed:{exc}") from exc

        if not isinstance(payload, Mapping) or not payload.get("id"):
            raise RuntimeError("dify_file_upload_failed:missing_file_id")
        return str(payload["id"])

    def _upload_wecom_media_to_dify(
        self,
        *,
        media_url: str,
        msgtype: str,
        index: int,
        message_id: str,
        user: str,
        base_url: str,
        api_key: str,
        cryptor: WeComCryptor,
    ) -> Mapping[str, str]:
        encrypted_bytes = self._download_wecom_media(media_url)
        plain_bytes = self._decrypt_wecom_media(encrypted_bytes, cryptor)
        extension, mime_type = self._detect_file_extension_and_mime(plain_bytes)
        filename = self._build_media_filename(
            message_id=message_id,
            index=index,
            msgtype=msgtype,
            extension=extension,
        )
        upload_file_id = self._upload_dify_file(
            base_url=base_url,
            api_key=api_key,
            user=user,
            filename=filename,
            content=plain_bytes,
            mime_type=mime_type,
        )

        dify_type = "document"
        if msgtype == "image":
            dify_type = "image"
        elif msgtype == "video":
            dify_type = "video"
        elif msgtype == "voice":
            dify_type = "audio"

        return {
            "type": dify_type,
            "transfer_method": "local_file",
            "upload_file_id": upload_file_id,
        }

    def _default_query_for_msgtype(self, msgtype: str) -> str:
        if msgtype == "image":
            return "Please analyze the attached image."
        if msgtype == "video":
            return "Please analyze the attached video."
        if msgtype == "file":
            return "Please analyze the attached file."
        if msgtype == "mixed":
            return "Please analyze the attached content."
        return "Please help with the attached content."

    def _append_query_part(
        self, query_parts: list[str], text: str, *, prefix: str | None = None
    ) -> None:
        normalized = text.strip()
        if not normalized:
            return
        if prefix:
            query_parts.append(f"{prefix}{normalized}")
        else:
            query_parts.append(normalized)

    def _collect_dify_message_content(
        self,
        *,
        container: Mapping[str, Any],
        message_id: str,
        cryptor: WeComCryptor,
        normalized_base_url: str,
        api_key: str,
        user: str,
        query_parts: list[str],
        files: list[Mapping[str, str]],
        next_file_index: list[int],
        text_prefix: str | None = None,
    ) -> None:
        msgtype = str(container.get("msgtype") or "")

        if msgtype == "text":
            self._append_query_part(
                query_parts,
                str(container.get("text", {}).get("content", "")),
                prefix=text_prefix,
            )
            return

        if msgtype == "voice":
            self._append_query_part(
                query_parts,
                str(container.get("voice", {}).get("content", "")),
                prefix=text_prefix,
            )
            return

        if msgtype in {"image", "file", "video"}:
            media_url = str(container.get(msgtype, {}).get("url", "")).strip()
            if not media_url:
                return
            files.append(
                self._upload_wecom_media_to_dify(
                    media_url=media_url,
                    msgtype=msgtype,
                    index=next_file_index[0],
                    message_id=message_id,
                    user=user,
                    base_url=normalized_base_url,
                    api_key=api_key,
                    cryptor=cryptor,
                )
            )
            next_file_index[0] += 1
            return

        if msgtype != "mixed":
            return

        msg_items = container.get("mixed", {}).get("msg_item", [])
        if not isinstance(msg_items, list):
            return

        for item in msg_items:
            if not isinstance(item, Mapping):
                continue
            self._collect_dify_message_content(
                container=item,
                message_id=message_id,
                cryptor=cryptor,
                normalized_base_url=normalized_base_url,
                api_key=api_key,
                user=user,
                query_parts=query_parts,
                files=files,
                next_file_index=next_file_index,
                text_prefix=text_prefix,
            )

    def _build_dify_query_and_files(
        self,
        *,
        payload: Mapping[str, Any],
        message_id: str,
        cryptor: WeComCryptor,
        normalized_base_url: str,
        api_key: str,
    ) -> tuple[str, list[Mapping[str, str]]]:
        msgtype = str(payload.get("msgtype") or "")
        user = self._build_dify_user(payload)
        query_parts: list[str] = []
        files: list[Mapping[str, str]] = []
        next_file_index = [0]

        self._collect_dify_message_content(
            container=payload,
            message_id=message_id,
            cryptor=cryptor,
            normalized_base_url=normalized_base_url,
            api_key=api_key,
            user=user,
            query_parts=query_parts,
            files=files,
            next_file_index=next_file_index,
        )

        quote = payload.get("quote")
        if isinstance(quote, Mapping):
            self._collect_dify_message_content(
                container=quote,
                message_id=message_id,
                cryptor=cryptor,
                normalized_base_url=normalized_base_url,
                api_key=api_key,
                user=user,
                query_parts=query_parts,
                files=files,
                next_file_index=next_file_index,
                text_prefix="Quoted content: ",
            )

        query = "\n".join(part for part in query_parts if part).strip()
        if not query and files:
            fallback_msgtype = msgtype
            quote = payload.get("quote")
            if fallback_msgtype == "text" and isinstance(quote, Mapping):
                quote_msgtype = str(quote.get("msgtype") or "")
                if quote_msgtype in {"image", "file", "video", "mixed", "voice"}:
                    fallback_msgtype = quote_msgtype
            query = self._default_query_for_msgtype(fallback_msgtype)

        logger.info(
            "WeCom Dify request summary: msgid=%r msgtype=%r query_preview=%r files_count=%d",
            message_id,
            msgtype,
            query[:200],
            len(files),
        )
        logger.info(
            "WeCom Dify final query: msgid=%r query=%r",
            message_id,
            query,
        )

        return (query, files)

    def _stream_dify_chat_events(
        self,
        *,
        base_url: str,
        api_key: str,
        query: str,
        user: str,
        conversation_id: str | None,
        files: list[Mapping[str, str]] | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        payload: dict[str, Any] = {
            "inputs": {},
            "query": query,
            "response_mode": "streaming",
            "user": user,
            "auto_generate_name": False,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if files:
            payload["files"] = files

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url=self._build_dify_endpoint(base_url),
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        def iter_events() -> Iterator[Mapping[str, Any]]:
            try:
                with request.urlopen(req, timeout=self._DIFY_TIMEOUT) as resp:
                    event_lines: list[str] = []
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace")
                        stripped_line = line.strip()

                        if not stripped_line:
                            if not event_lines:
                                continue

                            raw_payload = "\n".join(event_lines)
                            event_lines = []
                            parsed = self._parse_json_bytes(raw_payload.encode("utf-8"))
                            if isinstance(parsed, Mapping):
                                yield parsed
                            continue

                        if stripped_line.startswith(":"):
                            continue

                        if stripped_line.startswith("data:"):
                            event_lines.append(stripped_line[5:].strip())

                    if event_lines:
                        raw_payload = "\n".join(event_lines)
                        parsed = self._parse_json_bytes(raw_payload.encode("utf-8"))
                        if isinstance(parsed, Mapping):
                            yield parsed
            except error.HTTPError as exc:
                raise RuntimeError(
                    self._format_dify_error(
                        exc.code, self._parse_json_bytes(exc.read())
                    )
                ) from exc
            except error.URLError as exc:
                raise RuntimeError(f"dify_request_failed:{exc.reason}") from exc
            except Exception as exc:
                raise RuntimeError(f"dify_request_failed:{exc}") from exc

        return iter_events()

    def _build_integration_key(self, base_url: str, api_key: str) -> str:
        raw = f"{base_url}|{api_key}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    def _build_conversation_key(
        self, integration_key: str, payload: Mapping[str, Any]
    ) -> str | None:
        raw_from = payload.get("from")
        raw_chatid = payload.get("chatid")
        raw_aibotid = payload.get("aibotid")
        raw_chattype = payload.get("chattype")
        raw_userid = raw_from.get("userid") if isinstance(raw_from, Mapping) else None
        bot_id = str(raw_aibotid or "default")

        if raw_chattype == "group" and raw_chatid:
            return f"wecom_conv_{integration_key}_{bot_id}_{raw_chatid}"
        if raw_userid:
            return f"wecom_conv_{integration_key}_{bot_id}_{raw_userid}"
        if raw_chatid:
            return f"wecom_conv_{integration_key}_{bot_id}_{raw_chatid}"

        return None

    def _build_dify_user(self, payload: Mapping[str, Any]) -> str:
        raw_from = payload.get("from")
        raw_userid = raw_from.get("userid") if isinstance(raw_from, Mapping) else None
        return str(raw_userid or "")

    def _extract_dify_event_conversation_id(
        self, data: Mapping[str, Any]
    ) -> str | None:
        conversation_id = data.get("conversation_id")
        if conversation_id:
            return str(conversation_id)

        metadata = data.get("metadata")
        if isinstance(metadata, Mapping):
            metadata_conversation_id = metadata.get("conversation_id")
            if metadata_conversation_id:
                return str(metadata_conversation_id)

        return None

    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        token = settings.get("token")
        encoding_key = settings.get("encoding_aes_key")
        api_key = settings.get("api_key")
        base_url = settings.get("base_url")
        if not token or not encoding_key or not api_key or not base_url:
            return Response(
                status=400, response="missing token, encoding key, api key or base url"
            )

        try:
            normalized_base_url = self._normalize_base_url(str(base_url))
        except ValueError as exc:
            return Response(status=400, response=str(exc))

        signature = r.args.get("msg_signature")
        timestamp = r.args.get("timestamp")
        nonce = r.args.get("nonce")
        if not signature or not timestamp or not nonce:
            return Response(status=400, response="missing signature params")

        request_signature = signature
        request_timestamp = timestamp
        request_nonce = nonce

        try:
            body = r.get_json(force=True)
        except Exception as exc:
            return Response(status=400, response=f"invalid json: {exc}")

        encrypt = body.get("encrypt") if isinstance(body, Mapping) else None
        if not encrypt:
            return Response(status=400, response="missing encrypt")

        cryptor = WeComCryptor(token=token, encoding_aes_key=encoding_key)
        try:
            payload = cryptor.decrypt(
                signature=request_signature,
                timestamp=request_timestamp,
                nonce=request_nonce,
                ciphertext=encrypt,
            )
        except Exception as exc:
            return Response(status=400, response=f"decrypt_failed:{exc}")

        logger.info("WeCom Dify endpoint payload: %s", payload)
        quote = payload.get("quote") if isinstance(payload, Mapping) else None
        logger.info(
            "WeCom Dify endpoint quote summary: has_quote=%r quote_msgtype=%r",
            isinstance(quote, Mapping),
            quote.get("msgtype") if isinstance(quote, Mapping) else None,
        )
        msgtype = payload.get("msgtype")
        logger.info(
            "WeCom Dify endpoint received message: msgtype=%r msgid=%r",
            msgtype,
            payload.get("msgid"),
        )

        def handle_poll(target_id: str) -> Response:
            try:
                state = self.session.storage.get(self._state_key(target_id))
            except Exception:
                state = None
            state_str = state.decode("utf-8") if state else "done"

            try:
                content_bytes = self.session.storage.get(self._content_key(target_id))
            except Exception:
                content_bytes = b""
            content_str = content_bytes.decode("utf-8") if content_bytes else ""

            if state_str == "processing":
                res = self._build_wecom_res(
                    message_id=target_id,
                    content=content_str,
                    finish=False,
                    timestamp=request_timestamp,
                    nonce=request_nonce,
                    cryptor=cryptor,
                )
            else:
                res = self._build_wecom_res(
                    message_id=target_id,
                    content=content_str,
                    finish=True,
                    timestamp=request_timestamp,
                    nonce=request_nonce,
                    cryptor=cryptor,
                )

                try:
                    self.session.storage.delete(self._state_key(target_id))
                    self.session.storage.delete(self._content_key(target_id))
                except Exception:
                    pass

                try:
                    recent_bytes = self.session.storage.get(self._RECENT_IDS_KEY)
                    recent_ids = (
                        json.loads(recent_bytes.decode("utf-8")) if recent_bytes else []
                    )
                    if target_id in recent_ids:
                        recent_ids.remove(target_id)
                        self.session.storage.set(
                            self._RECENT_IDS_KEY,
                            json.dumps(recent_ids).encode("utf-8"),
                        )
                except Exception:
                    pass

            return Response(status=200, response=res, mimetype="application/json")

        def safe_set(key: str, val: bytes):
            try:
                self.session.storage.set(key, val)
            except Exception as exc:
                logger.warning("Storage safe_set error for %s: %s", key, exc)

        if msgtype == "stream":
            stream_id = payload.get("stream", {}).get("id")
            if not stream_id:
                return Response(status=200, response="success")
            logger.info("Processing Dify API stream poll: %s", stream_id)
            return handle_poll(stream_id)

        message_id = payload.get("msgid")
        if not message_id:
            return Response(status=200, response="success")

        if msgtype not in {"text", "voice", "image", "file", "video", "mixed"}:
            return Response(status=200, response="success")

        if self.session.storage.exist(self._state_key(message_id)):
            logger.info("Duplicate Dify API message detected/poll: %s", message_id)
            return handle_poll(message_id)

        logger.info("Processing new Dify API message: %s", message_id)

        try:
            recent_bytes = self.session.storage.get(self._RECENT_IDS_KEY)
            recent_ids = (
                json.loads(recent_bytes.decode("utf-8")) if recent_bytes else []
            )
        except Exception:
            recent_ids = []

        if message_id not in recent_ids:
            recent_ids.append(message_id)
            while len(recent_ids) > 20:
                old_id = recent_ids.pop(0)
                try:
                    self.session.storage.delete(self._state_key(old_id))
                    self.session.storage.delete(self._content_key(old_id))
                except Exception:
                    pass
            try:
                safe_set(self._RECENT_IDS_KEY, json.dumps(recent_ids).encode("utf-8"))
            except Exception as exc:
                logger.warning("Failed to update recent_ids to storage: %s", exc)

        safe_set(self._state_key(message_id), b"processing")
        safe_set(self._content_key(message_id), b"")

        integration_key = self._build_integration_key(normalized_base_url, str(api_key))
        conv_key = self._build_conversation_key(integration_key, payload)
        conversation_id = None
        if conv_key and self.session.storage.exist(conv_key):
            conversation_id = self.session.storage.get(conv_key).decode("utf-8")

        logger.info(
            "WeCom Dify endpoint conversation state: msgid=%r has_conversation=%r",
            message_id,
            bool(conversation_id),
        )

        def consume_response():
            full_answer = ""
            try:
                query, dify_files = self._build_dify_query_and_files(
                    payload=payload,
                    message_id=message_id,
                    cryptor=cryptor,
                    normalized_base_url=normalized_base_url,
                    api_key=str(api_key),
                )
                if not query and not dify_files:
                    safe_set(
                        self._content_key(message_id),
                        self._user_safe_error_message().encode("utf-8"),
                    )
                    return

                response_generator = self._stream_dify_chat_events(
                    base_url=normalized_base_url,
                    api_key=str(api_key),
                    query=query,
                    user=self._build_dify_user(payload),
                    conversation_id=conversation_id,
                    files=dify_files,
                )

                for data in response_generator:
                    event = data.get("event")

                    if event in {"agent_message", "message"}:
                        answer = str(data.get("answer", ""))
                        if answer:
                            full_answer += answer
                            safe_set(
                                self._content_key(message_id),
                                full_answer.encode("utf-8"),
                            )

                        conv_id = self._extract_dify_event_conversation_id(data)
                        if conv_id and conv_key:
                            safe_set(conv_key, conv_id.encode("utf-8"))

                    elif event == "message_replace":
                        full_answer = str(data.get("answer", ""))
                        safe_set(
                            self._content_key(message_id), full_answer.encode("utf-8")
                        )

                        conv_id = self._extract_dify_event_conversation_id(data)
                        if conv_id and conv_key:
                            safe_set(conv_key, conv_id.encode("utf-8"))

                    elif event == "agent_thought":
                        conv_id = self._extract_dify_event_conversation_id(data)
                        if conv_id and conv_key:
                            safe_set(conv_key, conv_id.encode("utf-8"))

                    elif event == "message_end":
                        conv_id = self._extract_dify_event_conversation_id(data)
                        if conv_id and conv_key:
                            safe_set(conv_key, conv_id.encode("utf-8"))
                        break

                    elif event == "error":
                        logger.error(
                            "Dify streaming event error for msgid=%s: %s",
                            message_id,
                            data,
                        )
                        if not full_answer:
                            safe_set(
                                self._content_key(message_id),
                                self._user_safe_error_message().encode("utf-8"),
                            )
                        break

                    elif event == "ping":
                        continue
            except Exception as exc:
                logger.error(
                    "Dify chat request failed for msgid=%s: %s",
                    message_id,
                    exc,
                )
                safe_set(
                    self._content_key(message_id),
                    self._user_safe_error_message().encode("utf-8"),
                )
            finally:
                safe_set(self._state_key(message_id), b"done")

        consume_response()

        try:
            final_content_bytes = self.session.storage.get(
                self._content_key(message_id)
            )
        except Exception:
            final_content_bytes = b""
        final_content = (
            final_content_bytes.decode("utf-8") if final_content_bytes else ""
        )

        res = self._build_wecom_res(
            message_id=message_id,
            content=final_content,
            finish=True,
            timestamp=request_timestamp,
            nonce=request_nonce,
            cryptor=cryptor,
        )
        return Response(status=200, response=res, mimetype="application/json")
