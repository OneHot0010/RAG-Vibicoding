"""Tests for the dashboard data browser page model."""

from __future__ import annotations

from pathlib import Path

from observability.dashboard.pages.data_browser import ALL_COLLECTIONS, data_browser_model, render


class FakeDataService:
    def __init__(self) -> None:
        self.documents = [
            {
                "source_path": "docs/a.pdf",
                "collection": "docs",
                "doc_hash": "hash-a",
                "chunk_count": 2,
                "image_count": 1,
                "processed_at": "2026-05-08 10:00:00",
            },
            {
                "source_path": "notes/b.md",
                "collection": "notes",
                "doc_hash": "hash-b",
                "chunk_count": 1,
                "image_count": 0,
                "processed_at": "2026-05-08 11:00:00",
            },
        ]

    def list_documents(self, collection: str | None = None) -> list[dict[str, object]]:
        if collection is None:
            return list(self.documents)
        return [document for document in self.documents if document["collection"] == collection]

    def get_document_detail(self, doc_id: str) -> dict[str, object]:
        if doc_id != "docs/a.pdf":
            return {
                "document": self.documents[1],
                "chunks": [
                    {
                        "id": "vec-c",
                        "text": "note chunk",
                        "metadata": {"chunk_index": 0, "source_path": "notes/b.md"},
                    }
                ],
                "images": [],
            }
        return {
            "document": self.documents[0],
            "chunks": [
                {
                    "id": "vec-a",
                    "text": "alpha chunk",
                    "metadata": {"chunk_index": 0, "source_path": "docs/a.pdf"},
                },
                {
                    "id": "vec-b",
                    "text": "beta chunk",
                    "metadata": {"chunk_index": 1, "source_path": "docs/a.pdf"},
                },
            ],
            "images": [
                {
                    "image_id": "img-a",
                    "file_path": str(Path("data/images/docs/img-a.png")),
                    "page_num": 1,
                    "created_at": "2026-05-08 10:01:00",
                }
            ],
        }


def test_data_browser_model_selects_first_document_and_formats_rows() -> None:
    model = data_browser_model(data_service=FakeDataService())

    assert model["collections"] == ["docs", "notes"]
    assert model["selected_collection"] is None
    assert [row["source_path"] for row in model["document_rows"]] == ["docs/a.pdf", "notes/b.md"]
    assert model["selected_document"]["source_path"] == "docs/a.pdf"
    assert [row["id"] for row in model["chunk_rows"]] == ["vec-a", "vec-b"]
    assert model["image_rows"] == [
        {
            "image_id": "img-a",
            "file_path": str(Path("data/images/docs/img-a.png")),
            "page_num": 1,
            "created_at": "2026-05-08 10:01:00",
        }
    ]


def test_data_browser_model_filters_collection_and_search() -> None:
    model = data_browser_model(
        data_service=FakeDataService(),
        collection="notes",
        search="b.md",
    )

    assert [document["source_path"] for document in model["documents"]] == ["notes/b.md"]
    assert model["selected_collection"] == "notes"
    assert model["chunk_rows"][0]["id"] == "vec-c"


def test_data_browser_model_accepts_all_collections_label_and_selected_hash() -> None:
    model = data_browser_model(
        data_service=FakeDataService(),
        collection=ALL_COLLECTIONS,
        selected_doc_id="hash-b",
    )

    assert model["selected_collection"] is None
    assert model["selected_document"]["source_path"] == "notes/b.md"
    assert model["detail"]["document"]["doc_hash"] == "hash-b"


def test_data_browser_render_without_streamlit_returns_model() -> None:
    rendered = render(data_service=FakeDataService())

    assert rendered["documents"][0]["source_path"] == "docs/a.pdf"
