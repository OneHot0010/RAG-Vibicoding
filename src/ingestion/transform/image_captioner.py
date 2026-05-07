"""Image captioning transform backed by optional Vision LLM calls."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.types import Chunk, normalize_image_refs
from ingestion.transform.base_transform import BaseTransform
from libs.llm import BaseVisionLLM, LLMFactory


DEFAULT_IMAGE_CAPTIONING_PROMPT = Path("config/prompts/image_captioning.txt")


class ImageCaptioner(BaseTransform):
    """Generate captions for chunk-local image references without blocking ingestion."""

    def __init__(
        self,
        settings: Settings,
        vision_llm: BaseVisionLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.vision_llm = vision_llm
        self.prompt_template = self._load_prompt(prompt_path)

    def transform(self, chunks: list[Chunk], trace: Any | None = None) -> list[Chunk]:
        """Caption images referenced by each chunk when enabled."""
        captioned: list[Chunk] = []
        for chunk in chunks:
            try:
                captioned.append(self._transform_one(chunk, trace))
            except Exception as exc:
                captioned.append(self._fallback_chunk(chunk, str(exc), trace))
        return captioned

    def _transform_one(self, chunk: Chunk, trace: Any | None = None) -> Chunk:
        image_refs = list(chunk.metadata.get("image_refs") or [])
        if not image_refs:
            return chunk

        if not self._enabled():
            return self._fallback_chunk(chunk, "image_captioner_disabled", trace)

        images = _images_by_id(chunk.metadata.get("images"))
        missing = [image_id for image_id in image_refs if image_id not in images]
        if missing:
            return self._fallback_chunk(chunk, f"missing image metadata: {missing}", trace)

        captions: dict[str, str] = {}
        try:
            vision_llm = self.vision_llm or LLMFactory.create_vision_llm(self.settings)
            for image_id in image_refs:
                image = images[image_id]
                prompt = self._build_prompt(chunk, image)
                response = vision_llm.chat_with_image(prompt, image["path"], trace=trace)
                if not response.content.strip():
                    raise ValueError(f"empty caption for image {image_id}")
                captions[image_id] = response.content.strip()
        except Exception as exc:
            return self._fallback_chunk(chunk, str(exc), trace)

        metadata = copy.deepcopy(chunk.metadata)
        metadata["image_captions"] = captions
        metadata["captioned_by"] = "vision_llm"
        metadata.pop("has_unprocessed_images", None)
        metadata.pop("image_caption_fallback_reason", None)
        _record_trace(trace, "image_captioner.captioned", {"chunk_id": chunk.id, "image_count": len(captions)})
        return _copy_chunk(chunk, metadata)

    def _build_prompt(self, chunk: Chunk, image: dict[str, Any]) -> str:
        context = chunk.text
        image_id = image["id"]
        try:
            return self.prompt_template.format(text=context, image_id=image_id, metadata=image)
        except KeyError:
            return f"{self.prompt_template}\n\nChunk context:\n{context}\n\nImage id: {image_id}"

    def _fallback_chunk(self, chunk: Chunk, reason: str, trace: Any | None = None) -> Chunk:
        metadata = copy.deepcopy(chunk.metadata)
        if metadata.get("image_refs"):
            metadata["has_unprocessed_images"] = True
            metadata["image_caption_fallback_reason"] = reason
            metadata.pop("image_captions", None)
        _record_trace(trace, "image_captioner.fallback", {"chunk_id": chunk.id, "reason": reason})
        return _copy_chunk(chunk, metadata)

    def _enabled(self) -> bool:
        raw = self.settings.raw or {}
        ingestion = raw.get("ingestion") if isinstance(raw, dict) else None
        image_captioner = ingestion.get("image_captioner") if isinstance(ingestion, dict) else None
        if isinstance(image_captioner, dict) and "enabled" in image_captioner:
            return bool(image_captioner["enabled"])
        return False

    def _load_prompt(self, prompt_path: str | Path | None = None) -> str:
        path = Path(prompt_path or DEFAULT_IMAGE_CAPTIONING_PROMPT)
        if not path.exists():
            return _default_prompt()
        prompt = path.read_text(encoding="utf-8").strip()
        return prompt or _default_prompt()


def _images_by_id(images: Any) -> dict[str, dict[str, Any]]:
    normalized = normalize_image_refs(images if isinstance(images, list) else [])
    return {image["id"]: image for image in normalized}


def _copy_chunk(chunk: Chunk, metadata: dict[str, Any]) -> Chunk:
    return Chunk(
        id=chunk.id,
        text=chunk.text,
        metadata=metadata,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        source_ref=chunk.source_ref,
    )


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)


def _default_prompt() -> str:
    return (
        "Describe the image for retrieval. Preserve concrete entities, labels, relationships, "
        "visible text, and how the image relates to this chunk.\n\n"
        "Chunk context:\n{text}\n\nImage id: {image_id}"
    )
