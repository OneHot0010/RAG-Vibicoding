"""PDF loader with a dependency-light fallback parser."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.types import Document, ImageRef, image_placeholder
from libs.loader.base_loader import BaseLoader, LoaderError


PdfImageExtractor = Callable[[Path], list[bytes]]


class PdfLoader(BaseLoader):
    """Load PDF files as markdown-like text plus normalized metadata."""

    def __init__(
        self,
        images_root: str | Path = "data/images",
        image_extractor: PdfImageExtractor | None = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.image_extractor = image_extractor or _extract_image_streams

    def load(self, path: str | Path) -> Document:
        pdf_path = Path(path)
        if not pdf_path.is_file():
            raise LoaderError(f"PDF file not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise LoaderError(f"PdfLoader only supports .pdf files: {pdf_path}")

        raw_bytes = pdf_path.read_bytes()
        doc_hash = hashlib.sha256(raw_bytes).hexdigest()
        text = _extract_text(pdf_path, raw_bytes).strip()
        if not text:
            text = f"# {pdf_path.stem}"

        metadata: dict[str, Any] = {
            "source_path": str(pdf_path),
            "doc_type": "pdf",
            "file_hash": doc_hash,
            "title": pdf_path.stem,
            "images": [],
        }
        warnings: list[str] = []

        try:
            image_refs, text = self._extract_images(pdf_path, doc_hash, text)
            metadata["images"] = [image_ref.to_dict() for image_ref in image_refs]
        except Exception as exc:
            warnings.append(f"image extraction failed: {exc}")

        if warnings:
            metadata["warnings"] = warnings
        return Document(id=doc_hash, text=text, metadata=metadata)

    def _extract_images(self, pdf_path: Path, doc_hash: str, text: str) -> tuple[list[ImageRef], str]:
        image_bytes_items = self.image_extractor(pdf_path)
        if not image_bytes_items:
            return [], text

        image_dir = self.images_root / doc_hash
        image_dir.mkdir(parents=True, exist_ok=True)

        image_refs: list[ImageRef] = []
        output_text = text
        for index, image_bytes in enumerate(image_bytes_items):
            if not image_bytes:
                continue
            image_id = f"{doc_hash}_1_{index}"
            relative_path = image_dir / f"{image_id}.png"
            relative_path.write_bytes(image_bytes)
            placeholder = image_placeholder(image_id)
            if output_text and not output_text.endswith("\n"):
                output_text += "\n"
            text_offset = len(output_text)
            output_text += placeholder
            image_refs.append(
                ImageRef(
                    id=image_id,
                    path=str(relative_path),
                    page=1,
                    text_offset=text_offset,
                    text_length=len(placeholder),
                    position={},
                )
            )
        return image_refs, output_text


def _extract_text(pdf_path: Path, raw_bytes: bytes) -> str:
    pypdf_text = _try_extract_with_pypdf(pdf_path)
    if pypdf_text:
        return pypdf_text
    return _extract_text_from_streams(raw_bytes)


def _try_extract_with_pypdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found,no-redef]
        except ImportError:
            return ""

    try:
        reader = PdfReader(str(pdf_path))
        return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    except Exception as exc:
        raise LoaderError(f"Failed to parse PDF text with installed PDF backend: {exc}") from exc


def _extract_text_from_streams(raw_bytes: bytes) -> str:
    decoded = raw_bytes.decode("latin-1", errors="ignore")
    fragments: list[str] = []
    for stream in re.findall(r"stream\s*(.*?)\s*endstream", decoded, flags=re.DOTALL):
        if "/Subtype /Image" in stream:
            continue
        fragments.extend(_extract_literal_text(stream))
    return "\n".join(fragment for fragment in fragments if fragment).strip()


def _extract_literal_text(stream: str) -> list[str]:
    fragments: list[str] = []
    for raw in re.findall(r"\((.*?)\)\s*Tj", stream, flags=re.DOTALL):
        fragments.append(_unescape_pdf_text(raw))
    for raw_array in re.findall(r"\[(.*?)\]\s*TJ", stream, flags=re.DOTALL):
        pieces = re.findall(r"\((.*?)\)", raw_array, flags=re.DOTALL)
        if pieces:
            fragments.append("".join(_unescape_pdf_text(piece) for piece in pieces))
    return fragments


def _unescape_pdf_text(value: str) -> str:
    return (
        value.replace(r"\(", "(")
        .replace(r"\)", ")")
        .replace(r"\\", "\\")
        .replace(r"\n", "\n")
        .replace(r"\r", "\r")
        .replace(r"\t", "\t")
    )


def _extract_image_streams(pdf_path: Path) -> list[bytes]:
    decoded = pdf_path.read_bytes().decode("latin-1", errors="ignore")
    images: list[bytes] = []
    for match in re.finditer(r"<<(?P<dict>.*?)>>\s*stream\s*(?P<data>.*?)\s*endstream", decoded, flags=re.DOTALL):
        if "/Subtype /Image" not in match.group("dict"):
            continue
        images.append(match.group("data").encode("latin-1", errors="ignore"))
    return images
