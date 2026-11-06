"""Who a request is from. Identity comes from a verified token, never from anything a client says.

Tokens are ``base64url(claims).base64url(hmac-sha256)``. There is no algorithm field to negotiate,
so the classic JWT confusions do not apply. A browser cannot set headers on a WebSocket and a query
string ends up in access logs, so sockets carry the token in a subprotocol instead.
"""

import base64
import hashlib
import hmac
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from pydantic import SecretStr, TypeAdapter, ValidationError

from realtime_agentic_site_builder.core.clock import Clock, system_now_ms
from realtime_agentic_site_builder.domain.base import Identifier
from realtime_agentic_site_builder.domain.errors import AuthenticationRequiredError

SUBPROTOCOL: Final = "rasb.v1"
TOKEN_SUBPROTOCOL_PREFIX: Final = "rasb.token."
AGENT_ID_PREFIX: Final = "user:"
_MAX_SUBJECT_LENGTH: Final = 120
_SUBJECT: Final = TypeAdapter(Identifier)


@dataclass(frozen=True, slots=True)
class Identity:
    user_id: str
    expires_at_s: int

    @property
    def agent_id(self) -> str:
        """Name this person in locks and deltas. The prefix keeps them from ever being an agent."""
        return f"{AGENT_ID_PREFIX}{self.user_id}"


def _encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class TokenAuthenticator:
    def __init__(self, secret: SecretStr, *, clock: Clock = system_now_ms) -> None:
        self._key = secret.get_secret_value().encode()
        self._clock = clock

    def _signature(self, payload: str) -> str:
        return _encode(hmac.new(self._key, payload.encode(), hashlib.sha256).digest())

    def issue(self, user_id: str, ttl_seconds: int = 3600) -> str:
        _SUBJECT.validate_python(user_id)
        if len(user_id) > _MAX_SUBJECT_LENGTH:
            raise ValueError(f"user ids are limited to {_MAX_SUBJECT_LENGTH} characters")
        claims = {"sub": user_id, "exp": self._clock() // 1000 + ttl_seconds}
        payload = _encode(json.dumps(claims, separators=(",", ":")).encode())
        return f"{payload}.{self._signature(payload)}"

    def verify(self, token: str) -> Identity:
        """Check the signature before reading a single claim, and only then look at the claims."""
        payload, _, signature = token.partition(".")
        if not hmac.compare_digest(self._signature(payload).encode(), signature.encode()):
            raise AuthenticationRequiredError("The token is not valid.")
        try:
            claims = json.loads(_decode(payload))
            subject = _SUBJECT.validate_python(claims["sub"])
            expires_at = claims["exp"]
        except (ValueError, KeyError, TypeError, ValidationError) as error:
            raise AuthenticationRequiredError("The token is not valid.") from error
        if not isinstance(expires_at, int) or expires_at <= self._clock() // 1000:
            raise AuthenticationRequiredError("The token has expired.")
        return Identity(user_id=subject, expires_at_s=expires_at)


def token_from_header(authorization: str | None) -> str | None:
    scheme, _, token = (authorization or "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def token_from_subprotocols(subprotocols: Iterable[str]) -> str | None:
    for subprotocol in subprotocols:
        if subprotocol.startswith(TOKEN_SUBPROTOCOL_PREFIX):
            return subprotocol.removeprefix(TOKEN_SUBPROTOCOL_PREFIX)
    return None
