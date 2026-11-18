from enum import StrEnum

from app.exceptions.exception import UnsupportedFileType

PDF_SIGNATURE = b"%PDF-"
PDF_SIGNATURE_WINDOW = 1024
ZIP_SIGNATURE = b"PK\x03\x04"


class FileType(StrEnum):
    PDF = "pdf"
    TEXT = "text"


def detect_file_type(data: bytes) -> FileType:
    if data[:PDF_SIGNATURE_WINDOW].lstrip().startswith(PDF_SIGNATURE):
        return FileType.PDF
    if data.startswith(ZIP_SIGNATURE):
        raise UnsupportedFileType("archive-based files are not supported")
    if b"\x00" in data:
        raise UnsupportedFileType("binary content is not supported")
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise UnsupportedFileType("text must be UTF-8 encoded") from error
    return FileType.TEXT
