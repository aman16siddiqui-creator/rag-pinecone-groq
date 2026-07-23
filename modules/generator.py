from __future__ import annotations

from dataclasses import dataclass
from typing import List

from groq import Groq

from config import settings
from modules.retriever import RetrievedChunk

FALLBACK_ANSWER = "The answer is not available in the provided document."

SYSTEM_PROMPT = """You are a careful, helpful document question-answering assistant.
You must answer the user's question using ONLY the CONTEXT excerpts provided below.

Rules (follow strictly):
- Do NOT use any outside knowledge, prior training data, or assumptions about
  facts that aren't in the context.
- If the question is broad ("what is this document about", "summarize this",
  "explain the PDF"), synthesize a real answer from everything in the
  CONTEXT excerpts — describe the topics, key points, and structure they
  cover. This still counts as "using only the context", just at a higher
  level. Do NOT refuse just because no single excerpt directly restates
  the question.
- Only use the exact fallback sentence "The answer is not available in the
  provided document." when the CONTEXT excerpts are genuinely irrelevant to
  what was asked. Do not use it just because the answer requires combining
  or summarizing multiple excerpts.
- When you do answer, be concise and factual, and where useful mention which
  page(s) the information came from (the context excerpts are labelled with
  their page numbers).
- Never fabricate a page number, document name, or fact not present in the context.
"""


@dataclass
class GeneratedAnswer:
    answer: str
    used_context: bool
    grounded: bool  # False if we short-circuited to the fallback


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i} | Document: {c.doc_name} | Page: {c.page_number} | "
            f"Similarity: {c.score:.2f}]\n{c.text}"
        )
    return "\n\n".join(parts)


class GroqGenerator:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing.")
        self.client = Groq(api_key=self.api_key)

    def generate(self, query: str, chunks: List[RetrievedChunk], temperature: float = 0.1) -> GeneratedAnswer:
        # Layer 2: nothing survived retrieval -> don't even call the LLM.
        if not chunks:
            return GeneratedAnswer(answer=FALLBACK_ANSWER, used_context=False, grounded=False)

        context_block = _build_context_block(chunks)
        user_prompt = (
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {query}\n\n"
            f"Answer strictly based on the CONTEXT above."
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            answer_text = completion.choices[0].message.content.strip()
        except Exception as exc:
            raise RuntimeError(f"Groq API call failed: {exc}") from exc

        is_fallback = FALLBACK_ANSWER.lower() in answer_text.lower()
        return GeneratedAnswer(
            answer=answer_text,
            used_context=True,
            grounded=not is_fallback,
        )


def estimate_confidence(chunks: List[RetrievedChunk], generated: GeneratedAnswer) -> float:
    """A simple, explainable confidence score (0-100) combining:
      - the average similarity score of the chunks actually used, and
      - whether the model itself declared the context insufficient.
    This is intentionally simple/transparent rather than a black-box
    number, so it can be defended in a viva."""
    if not chunks or not generated.grounded:
        return 0.0
    avg_score = sum(c.score for c in chunks) / len(chunks)
    # scores are cosine similarities in [0,1] (post-threshold), scale to 0-100
    return round(min(avg_score, 1.0) * 100, 1)
