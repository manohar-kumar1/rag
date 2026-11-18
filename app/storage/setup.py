from app.core.config import get_settings
from app.exceptions.exception import AppError
from app.exceptions.handlers import report_cli_error
from app.storage.factory import create_storage_client
from app.storage.supabase import ensure_private_bucket


def main() -> int:
    settings = get_settings()
    with create_storage_client(settings) as client:
        try:
            created = ensure_private_bucket(
                client, settings.storage_bucket, settings.max_upload_bytes
            )
        except AppError as error:
            return report_cli_error(error)
    outcome = "created" if created else "already exists"
    print(f"private bucket {settings.storage_bucket} {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
