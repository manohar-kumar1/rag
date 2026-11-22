from urllib.parse import quote

import httpx

from app.exceptions.exception import StorageError, StoredFileMissing


class SupabaseFileStore:
    def __init__(self, client: httpx.Client, bucket: str) -> None:
        self._client = client
        self._bucket = quote(bucket, safe="")

    def put(self, key: str, data: bytes, content_type: str) -> None:
        response = self._send(
            "POST",
            f"/object/{self._bucket}/{self._path(key)}",
            content=data,
            headers={"Content-Type": content_type},
        )
        self._require_success(response, "upload")

    def get(self, key: str) -> bytes:
        response = self._send(
            "GET", f"/object/authenticated/{self._bucket}/{self._path(key)}"
        )
        if self._is_not_found(response):
            raise StoredFileMissing
        self._require_success(response, "download")
        return response.content

    def delete(self, key: str) -> None:
        response = self._send("DELETE", f"/object/{self._bucket}/{self._path(key)}")
        if self._is_not_found(response):
            return
        self._require_success(response, "delete")

    def check(self) -> None:
        response = self._send("GET", f"/bucket/{self._bucket}")
        self._require_success(response, "bucket check")

    @staticmethod
    def _is_not_found(response: httpx.Response) -> bool:
        if response.status_code == 404:
            return True
        try:
            body = response.json()
        except ValueError:
            return False
        return isinstance(body, dict) and str(body.get("statusCode")) == "404"

    @staticmethod
    def _path(key: str) -> str:
        return quote(key, safe="/")

    def _send(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            return self._client.request(method, path, **kwargs)
        except httpx.TransportError as error:
            raise StorageError(
                f"storage request failed: {type(error).__name__}"
            ) from error

    @staticmethod
    def _require_success(response: httpx.Response, action: str) -> None:
        if not response.is_success:
            raise StorageError(
                f"storage {action} failed with status {response.status_code}"
            )


def ensure_private_bucket(
    client: httpx.Client, bucket: str, max_file_bytes: int
) -> bool:
    name = quote(bucket, safe="")
    try:
        existing = client.get(f"/bucket/{name}")
        if existing.status_code == 200:
            current = existing.json()
            if current.get("public"):
                raise StorageError(f"bucket {bucket} is public and must be private")
            if current.get("file_size_limit") != max_file_bytes:
                updated = client.put(
                    f"/bucket/{name}",
                    json={"public": False, "file_size_limit": max_file_bytes},
                )
                if not updated.is_success:
                    raise StorageError(
                        f"bucket update failed with status {updated.status_code}"
                    )
            return False
        created = client.post(
            "/bucket",
            json={
                "name": bucket,
                "id": bucket,
                "public": False,
                "file_size_limit": max_file_bytes,
            },
        )
    except httpx.TransportError as error:
        raise StorageError(f"storage request failed: {type(error).__name__}") from error
    if not created.is_success:
        raise StorageError(f"bucket creation failed with status {created.status_code}")
    return True
