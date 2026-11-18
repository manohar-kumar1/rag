import httpx

from app.core.config import Settings
from app.storage.protocol import FileStore
from app.storage.supabase import SupabaseFileStore


def create_storage_client(settings: Settings) -> httpx.Client:
    return httpx.Client(
        base_url=f"{settings.supabase_url.rstrip('/')}/storage/v1",
        headers={
            "apikey": settings.supabase_secret_key,
            "Authorization": f"Bearer {settings.supabase_secret_key}",
        },
        timeout=settings.storage_timeout_seconds,
    )


def create_file_store(settings: Settings) -> FileStore:
    return SupabaseFileStore(create_storage_client(settings), settings.storage_bucket)
