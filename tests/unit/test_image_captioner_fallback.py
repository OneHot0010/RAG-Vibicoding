"""Tests for image captioning transform fallback behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
    VisionLLMSettings,
)
from core.trace.trace_context import TraceContext
from core.types import Chunk, ImageRef, image_placeholder
from ingestion.transform import ImageCaptioner
from libs.llm import BaseVisionLLM, ChatResponse


class FakeVisionLLM(BaseVisionLLM):
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, str | bytes, Any | None]] = []

    def chat_with_image(self, text: str, image_path: str | bytes, trace: Any | None = None) -> ChatResponse:
        self.calls.append((text, image_path, trace))
        if isinstance(self.response, Exception):
            raise self.response
        return ChatResponse(content=self.response)


def make_settings(enabled: bool = False) -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=0),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings("bm25", "rrf", 20, 20, 10),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
        raw={"ingestion": {"image_captioner": {"enabled": enabled}}},
    )


def make_chunk(image_count: int = 1) -> Chunk:
    images = []
    image_refs = []
    text = "Architecture diagram"
    for index in range(image_count):
        image_id = f"img-{index}"
        images.append(
            ImageRef(
                id=image_id,
                path=f"data/images/doc/{image_id}.png",
                page=1,
                text_offset=len(text),
                text_length=len(image_placeholder(image_id)),
            ).to_dict()
        )
        image_refs.append(image_id)
        text += "\n" + image_placeholder(image_id)
    return Chunk(
        id="chunk-1",
        text=text,
        metadata={
            "source_path": "docs/a.pdf",
            "chunk_index": 0,
            "images": images,
            "image_refs": image_refs,
        },
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_no_image_refs_returns_chunk_unchanged() -> None:
    chunk = Chunk(
        id="chunk-no-image",
        text="plain text",
        metadata={"source_path": "docs/a.md", "chunk_index": 0},
        start_offset=0,
        end_offset=10,
    )

    result = ImageCaptioner(make_settings(enabled=True), vision_llm=FakeVisionLLM("caption")).transform([chunk])[0]

    assert result == chunk


def test_disabled_captioner_preserves_refs_and_marks_unprocessed() -> None:
    chunk = make_chunk()
    vision = FakeVisionLLM("caption")

    result = ImageCaptioner(make_settings(enabled=False), vision_llm=vision).transform([chunk])[0]

    assert result.metadata["image_refs"] == ["img-0"]
    assert result.metadata["has_unprocessed_images"] is True
    assert result.metadata["image_caption_fallback_reason"] == "image_captioner_disabled"
    assert "image_captions" not in result.metadata
    assert vision.calls == []


def test_enabled_captioner_generates_captions() -> None:
    chunk = make_chunk()
    vision = FakeVisionLLM("A layered system architecture diagram.")

    result = ImageCaptioner(make_settings(enabled=True), vision_llm=vision).transform([chunk])[0]

    assert result.metadata["image_captions"] == {"img-0": "A layered system architecture diagram."}
    assert result.metadata["captioned_by"] == "vision_llm"
    assert "has_unprocessed_images" not in result.metadata
    assert vision.calls[0][1] == "data/images/doc/img-0.png"
    assert "Architecture diagram" in vision.calls[0][0]
    assert "img-0" in vision.calls[0][0]


def test_enabled_captioner_handles_multiple_images_in_order() -> None:
    chunk = make_chunk(image_count=2)
    vision = FakeVisionLLM("caption")

    result = ImageCaptioner(make_settings(enabled=True), vision_llm=vision).transform([chunk])[0]

    assert result.metadata["image_captions"] == {"img-0": "caption", "img-1": "caption"}
    assert [call[1] for call in vision.calls] == [
        "data/images/doc/img-0.png",
        "data/images/doc/img-1.png",
    ]


def test_vision_failure_falls_back_without_blocking() -> None:
    chunk = make_chunk()

    result = ImageCaptioner(
        make_settings(enabled=True),
        vision_llm=FakeVisionLLM(RuntimeError("vision unavailable")),
    ).transform([chunk])[0]

    assert result.metadata["has_unprocessed_images"] is True
    assert result.metadata["image_caption_fallback_reason"] == "vision unavailable"
    assert "image_captions" not in result.metadata


def test_missing_image_metadata_falls_back() -> None:
    chunk = make_chunk()
    metadata = dict(chunk.metadata)
    metadata["images"] = []
    chunk = Chunk(chunk.id, chunk.text, metadata, chunk.start_offset, chunk.end_offset, chunk.source_ref)

    result = ImageCaptioner(make_settings(enabled=True), vision_llm=FakeVisionLLM("caption")).transform([chunk])[0]

    assert result.metadata["has_unprocessed_images"] is True
    assert "missing image metadata" in result.metadata["image_caption_fallback_reason"]


def test_empty_caption_falls_back() -> None:
    result = ImageCaptioner(make_settings(enabled=True), vision_llm=FakeVisionLLM(" ")).transform([make_chunk()])[0]

    assert result.metadata["has_unprocessed_images"] is True
    assert "empty caption" in result.metadata["image_caption_fallback_reason"]


def test_prompt_can_be_loaded_from_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "caption.txt"
    prompt_path.write_text("Caption {image_id}: {text}", encoding="utf-8")
    vision = FakeVisionLLM("caption")

    ImageCaptioner(make_settings(enabled=True), vision_llm=vision, prompt_path=prompt_path).transform([make_chunk()])

    assert vision.calls[0][0].startswith("Caption img-0:")


def test_trace_records_caption_and_fallback_paths() -> None:
    caption_trace = TraceContext()
    fallback_trace = TraceContext()

    ImageCaptioner(make_settings(enabled=True), vision_llm=FakeVisionLLM("caption")).transform(
        [make_chunk()],
        trace=caption_trace,
    )
    ImageCaptioner(make_settings(enabled=False), vision_llm=FakeVisionLLM("caption")).transform(
        [make_chunk()],
        trace=fallback_trace,
    )

    assert caption_trace.stages == [
        {"name": "image_captioner.captioned", "data": {"chunk_id": "chunk-1", "image_count": 1}}
    ]
    assert fallback_trace.stages == [
        {
            "name": "image_captioner.fallback",
            "data": {"chunk_id": "chunk-1", "reason": "image_captioner_disabled"},
        }
    ]
