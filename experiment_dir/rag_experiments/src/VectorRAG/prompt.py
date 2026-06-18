from __future__ import annotations

FINAL_PROMPT_TEMPLATE = """You are the answer engine in a Japanese RAG system.
Follow these constraints strictly:
- Use only facts included in the provided context.
- If the context does not contain the needed information, state: "本文に該当記述がありません。"
- Answer in natural Japanese within two or three sentences.
- When an important supporting fact exists, include a concise summary of that evidence.

Context:
{context}

Question:
{question}

Answer:
"""


def build_final_prompt(context: str, question: str) -> str:
    return FINAL_PROMPT_TEMPLATE.format(context=context, question=question)
