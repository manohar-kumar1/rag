from collections.abc import Sequence

from app.retrieval.protocol import RetrievedChunk


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        f'<passage number="{number}" source="{chunk.source}">\n{chunk.prompt_text}\n</passage>'
        for number, chunk in enumerate(chunks, start=1)
    )
