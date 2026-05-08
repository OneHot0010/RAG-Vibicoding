"""Roundtrip tests for the default ChromaStore backend."""

from __future__ import annotations

import json

import pytest

from core.settings import VectorStoreSettings
from libs.vector_store import ChromaStore, ChromaStoreError, VectorRecord, VectorStoreFactory


@pytest.fixture(autouse=True)
def register_chroma_store() -> None:
    VectorStoreFactory.clear()
    VectorStoreFactory.register("chroma", ChromaStore)


def test_factory_creates_chroma_store(tmp_path) -> None:
    store = VectorStoreFactory.create(
        VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma"))
    )

    assert isinstance(store, ChromaStore)


def test_upsert_query_roundtrip_is_deterministic(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma")))
    store.upsert(
        [
            VectorRecord(id="chunk-a", vector=[1.0, 0.0], text="alpha", metadata={"source": "a.md"}),
            VectorRecord(id="chunk-b", vector=[0.0, 1.0], text="beta", metadata={"source": "b.md"}),
            VectorRecord(id="chunk-c", vector=[0.8, 0.2], text="gamma", metadata={"source": "c.md"}),
        ]
    )

    results = store.query([1.0, 0.0], top_k=2)

    assert [result.id for result in results] == ["chunk-a", "chunk-c"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].text == "alpha"
    assert results[0].metadata == {"source": "a.md"}


def test_top_k_limits_results(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma")))
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0, 0.0], text="a"),
            VectorRecord(id="b", vector=[0.9, 0.1], text="b"),
        ]
    )

    results = store.query([1.0, 0.0], top_k=1)

    assert [result.id for result in results] == ["a"]


def test_metadata_filters_are_applied(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma")))
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0, 0.0], text="a", metadata={"collection": "docs"}),
            VectorRecord(id="b", vector=[1.0, 0.0], text="b", metadata={"collection": "notes"}),
        ]
    )

    results = store.query([1.0, 0.0], top_k=10, filters={"collection": "notes"})

    assert [result.id for result in results] == ["b"]


def test_upsert_replaces_existing_record_by_id(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma")))
    store.upsert([VectorRecord(id="a", vector=[1.0], text="old")])
    store.upsert([VectorRecord(id="a", vector=[2.0], text="new")])

    results = store.query([2.0], top_k=10)

    assert [result.text for result in results] == ["new"]


def test_records_persist_across_store_instances(tmp_path) -> None:
    persist_path = tmp_path / "chroma"
    first = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(persist_path)))
    first.upsert([VectorRecord(id="a", vector=[1.0, 0.0], text="persisted")])

    second = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(persist_path)))
    results = second.query([1.0, 0.0], top_k=1)

    assert [result.text for result in results] == ["persisted"]
    raw = json.loads((persist_path / "records.json").read_text(encoding="utf-8"))
    assert raw[0]["id"] == "a"


def test_get_by_ids_preserves_order_and_returns_full_records(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma")))
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0], text="alpha", metadata={"source": "a.md"}),
            VectorRecord(id="b", vector=[2.0], text="beta", metadata={"source": "b.md"}),
        ]
    )

    records = store.get_by_ids(["b", "missing", "a"])

    assert [record.id for record in records] == ["b", "a"]
    assert records[0].text == "beta"
    assert records[0].metadata == {"source": "b.md"}


def test_get_by_ids_can_match_original_chunk_id_metadata(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma")))
    store.upsert(
        [
            VectorRecord(
                id="stable-vector-id",
                vector=[1.0],
                text="alpha",
                metadata={"source": "a.md", "original_chunk_id": "chunk-a"},
            )
        ]
    )

    records = store.get_by_ids(["chunk-a"])

    assert [record.id for record in records] == ["stable-vector-id"]


def test_query_dimension_mismatch_has_clear_error(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma")))
    store.upsert([VectorRecord(id="a", vector=[1.0, 0.0], text="alpha")])

    with pytest.raises(ChromaStoreError, match="dimension"):
        store.query([1.0], top_k=1)


def test_invalid_top_k_has_clear_error(tmp_path) -> None:
    store = ChromaStore(VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma")))

    with pytest.raises(ChromaStoreError, match="top_k"):
        store.query([1.0], top_k=0)
