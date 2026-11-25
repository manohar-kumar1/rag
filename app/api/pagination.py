import base64
import binascii

from app.exceptions.errors import InvalidCursor

CURSOR_PREFIX = "id:"
MAX_CURSOR_LENGTH = 128
MAX_ID = 2**63 - 1


def encode_cursor(last_id: int) -> str:
    raw = f"{CURSOR_PREFIX}{last_id}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> int | None:
    if cursor is None:
        return None
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        text = base64.urlsafe_b64decode(padded.encode("ascii")).decode()
    except (binascii.Error, UnicodeError):
        raise InvalidCursor from None
    value = text.removeprefix(CURSOR_PREFIX)
    valid = text.startswith(CURSOR_PREFIX) and value.isascii() and value.isdigit()
    if not valid or not 1 <= int(value) <= MAX_ID:
        raise InvalidCursor
    return int(value)
