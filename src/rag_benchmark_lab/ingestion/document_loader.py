from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import mimetypes

from .text_cleaner import clean_text

try:  # optional dependencies are loaded lazily so unit tests can run without them
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:
        import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None

try:
    from docx import Document as DocxDocument  # type: ignore
except Exception:  # pragma: no cover
    DocxDocument = None


@dataclass(slots=True)
class LoadedPage:
    page_number: int
    text: str


@dataclass(slots=True)
class LoadedDocument:
    document_name: str
    source_path: str | None
    mime_type: str
    raw_bytes: bytes
    pages: list[LoadedPage]

    @property
    def text(self) -> str:
        return clean_text("\n\n".join(page.text for page in self.pages))

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def load_bytes(document_name: str, raw_bytes: bytes, source_path: str | None = None) -> LoadedDocument:
    suffix = Path(document_name).suffix.lower()
    mime_type = _guess_mime_type(Path(document_name))
    if suffix == ".pdf":
        if fitz is None:
            raise RuntimeError("PyMuPDF is required to parse PDF documents")
        pages: list[LoadedPage] = []
        with fitz.open(stream=raw_bytes, filetype="pdf") as pdf:
            for index, page in enumerate(pdf, start=1):
                pages.append(LoadedPage(page_number=index, text=page.get_text("text")))
        return LoadedDocument(document_name=document_name, source_path=source_path, mime_type=mime_type, raw_bytes=raw_bytes, pages=pages)
    if suffix == ".docx":
        if DocxDocument is None:
            raise RuntimeError("python-docx is required to parse DOCX documents")
        document = DocxDocument(BytesIO(raw_bytes))
        pages = [LoadedPage(page_number=1, text="\n".join(paragraph.text for paragraph in document.paragraphs))]
        return LoadedDocument(document_name=document_name, source_path=source_path, mime_type=mime_type, raw_bytes=raw_bytes, pages=pages)

    text = raw_bytes.decode("utf-8", errors="replace")
    pages = [LoadedPage(page_number=1, text=text)]
    return LoadedDocument(document_name=document_name, source_path=source_path, mime_type=mime_type, raw_bytes=raw_bytes, pages=pages)


def load_path(path: str | Path) -> LoadedDocument:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    raw_bytes = resolved.read_bytes()
    return load_bytes(resolved.name, raw_bytes, source_path=str(resolved))


def validate_upload_size(raw_bytes: bytes, max_size_mb: int) -> None:
    if len(raw_bytes) > max_size_mb * 1024 * 1024:
        raise ValueError(f"File exceeds the configured upload limit of {max_size_mb} MB")

