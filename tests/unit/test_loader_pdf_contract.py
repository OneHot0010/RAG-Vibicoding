"""Tests for the PDF loader contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.types import Document, extract_image_placeholders
from libs.loader import BaseLoader, LoaderError, PdfLoader


def write_minimal_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
    path.write_text(
        "\n".join(
            [
                "%PDF-1.4",
                "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
                "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
                "3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R >> endobj",
                f"4 0 obj << /Length {len(escaped) + 40} >>",
                "stream",
                "BT",
                "/F1 12 Tf",
                f"72 720 Td ({escaped}) Tj",
                "ET",
                "endstream",
                "endobj",
                "%%EOF",
            ]
        ),
        encoding="latin-1",
    )


def test_base_loader_contract_can_be_implemented(tmp_path: Path) -> None:
    class FakeLoader(BaseLoader):
        def load(self, path: str | Path) -> Document:
            return Document(id="fake", text="content", metadata={"source_path": str(path)})

    document = FakeLoader().load(tmp_path / "sample.pdf")

    assert document.id == "fake"
    assert document.metadata["source_path"].endswith("sample.pdf")


def test_pdf_loader_loads_text_pdf_as_document(tmp_path: Path) -> None:
    pdf_path = tmp_path / "simple.pdf"
    write_minimal_pdf(pdf_path, "Hello PDF")

    document = PdfLoader(images_root=tmp_path / "images").load(pdf_path)

    assert isinstance(document, Document)
    assert document.text == "Hello PDF"
    assert len(document.id) == 64
    assert document.metadata["source_path"] == str(pdf_path)
    assert document.metadata["doc_type"] == "pdf"
    assert document.metadata["title"] == "simple"
    assert document.metadata["images"] == []


def test_pdf_loader_extracts_images_and_inserts_placeholders(tmp_path: Path) -> None:
    pdf_path = tmp_path / "with_images.pdf"
    write_minimal_pdf(pdf_path, "Text before image")

    def fake_image_extractor(path: Path) -> list[bytes]:
        assert path == pdf_path
        return [b"png-bytes"]

    document = PdfLoader(
        images_root=tmp_path / "images",
        image_extractor=fake_image_extractor,
    ).load(pdf_path)

    images = document.metadata["images"]
    assert len(images) == 1
    assert images[0]["id"] in extract_image_placeholders(document.text)
    assert document.text.endswith(f"[IMAGE: {images[0]['id']}]")
    assert images[0]["text_offset"] == document.text.index("[IMAGE:")
    assert images[0]["text_length"] == len(f"[IMAGE: {images[0]['id']}]")
    image_path = Path(images[0]["path"])
    assert image_path.is_file()
    assert image_path.read_bytes() == b"png-bytes"


def test_pdf_loader_image_extraction_failure_does_not_block_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "image_failure.pdf"
    write_minimal_pdf(pdf_path, "Still parse text")

    def failing_image_extractor(path: Path) -> list[bytes]:
        raise RuntimeError("boom")

    document = PdfLoader(
        images_root=tmp_path / "images",
        image_extractor=failing_image_extractor,
    ).load(pdf_path)

    assert document.text == "Still parse text"
    assert document.metadata["images"] == []
    assert document.metadata["warnings"] == ["image extraction failed: boom"]


def test_pdf_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LoaderError, match="PDF file not found"):
        PdfLoader().load(tmp_path / "missing.pdf")


def test_pdf_loader_rejects_non_pdf_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("not pdf", encoding="utf-8")

    with pytest.raises(LoaderError, match="only supports .pdf"):
        PdfLoader().load(path)


def test_pdf_loader_falls_back_to_title_when_text_is_empty(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_text("%PDF-1.4\n%%EOF", encoding="latin-1")

    document = PdfLoader(images_root=tmp_path / "images").load(pdf_path)

    assert document.text == "# empty"


def test_default_image_stream_extractor_handles_basic_pdf_image_stream(tmp_path: Path) -> None:
    pdf_path = tmp_path / "raw_image.pdf"
    pdf_path.write_text(
        "%PDF-1.4\n"
        "<< /Type /XObject /Subtype /Image /Width 1 /Height 1 >>\n"
        "stream\nRAWIMAGE\nendstream\n"
        "%%EOF",
        encoding="latin-1",
    )

    document = PdfLoader(images_root=tmp_path / "images").load(pdf_path)

    assert len(document.metadata["images"]) == 1
    assert Path(document.metadata["images"][0]["path"]).read_bytes() == b"RAWIMAGE"
