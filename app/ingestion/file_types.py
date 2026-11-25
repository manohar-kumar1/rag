import zipfile
from enum import StrEnum
from io import BytesIO

from app.exceptions.errors import UnsupportedFileType

PDF_SIGNATURE = b"%PDF-"
PDF_SIGNATURE_WINDOW = 1024
ZIP_SIGNATURE = b"PK\x03\x04"
OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
DOCX_REQUIRED_PARTS = frozenset({"[Content_Types].xml", "word/document.xml"})


class FileType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TEXT = "text"


MIME_TYPES = {
    FileType.PDF: "application/pdf",
    FileType.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    FileType.TEXT: "text/plain",
}


def detect_file_type(data: bytes) -> FileType:
    if data[:PDF_SIGNATURE_WINDOW].lstrip().startswith(PDF_SIGNATURE):
        return FileType.PDF
    if data.startswith(ZIP_SIGNATURE):
        return detect_archive_type(data)
    if data.startswith(OLE_SIGNATURE):
        raise UnsupportedFileType("legacy or encrypted Office files are not supported")
    if b"\x00" in data:
        raise UnsupportedFileType("binary content is not supported")
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise UnsupportedFileType("text must be UTF-8 encoded") from error
    return FileType.TEXT


def detect_archive_type(data: bytes) -> FileType:
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile as error:
        raise UnsupportedFileType("archive is corrupt") from error
    if DOCX_REQUIRED_PARTS <= names:
        return FileType.DOCX
    raise UnsupportedFileType("archive-based files are not supported")
