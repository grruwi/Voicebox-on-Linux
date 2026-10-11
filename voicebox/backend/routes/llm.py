"""LLM inference endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from .. import models
from ..services import llm

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/llm/generate", response_model=models.LLMGenerateResponse)
async def llm_generate(request: models.LLMGenerateRequest):
    """A single-turn completion through the LLM backend (LM Studio — whatever model it
    has loaded). Voicebox downloads no LLM weights."""
    backend = llm.get_llm_model()
    model_size = backend.model_size

    examples: list[tuple[str, str]] | None = None
    if request.examples:
        for pair in request.examples:
            if len(pair) != 2:
                raise HTTPException(
                    status_code=400,
                    detail="Each example must be a [user, assistant] pair",
                )
        examples = [(pair[0], pair[1]) for pair in request.examples]

    try:
        text = await backend.generate(
            prompt=request.prompt,
            system=request.system,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            model_size=model_size,
            examples=examples,
        )
        return models.LLMGenerateResponse(text=text, model_size=model_size)
    except Exception as e:
        # The backend exception text can include filesystem paths and stack
        # frames — log it server-side and hand the client a generic message.
        logger.exception("LLM generate failed")
        raise HTTPException(status_code=500, detail="LLM generation failed") from e
