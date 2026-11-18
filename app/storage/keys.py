from uuid import uuid4


def new_storage_key() -> str:
    return uuid4().hex
