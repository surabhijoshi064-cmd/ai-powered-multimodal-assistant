from typing import List, Dict, Any, Optional, Tuple
from app.config import settings


class TextChunker:
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 4)

    def chunk_pages(
        self,
        pages: List[Tuple[Optional[int], str]],
    ) -> List[Dict[str, Any]]:
        """
        Splits text extracted from document pages into overlapping chunks,
        preserving page numbers for citations.
        """
        chunks: List[Dict[str, Any]] = []
        global_index = 0

        for page_num, text in pages:
            cleaned_text = " ".join(text.split())
            if not cleaned_text:
                continue

            start = 0
            text_len = len(cleaned_text)

            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                
                # Try to break on a sentence or word boundary if possible
                if end < text_len:
                    last_period = cleaned_text.rfind(". ", start, end)
                    last_space = cleaned_text.rfind(" ", start, end)
                    if last_period != -1 and last_period > start + (self.chunk_size // 2):
                        end = last_period + 1
                    elif last_space != -1 and last_space > start + (self.chunk_size // 2):
                        end = last_space

                chunk_str = cleaned_text[start:end].strip()
                if chunk_str:
                    chunks.append({
                        "chunk_index": global_index,
                        "chunk_text": chunk_str,
                        "page_number": page_num,
                    })
                    global_index += 1

                if end >= text_len:
                    break

                start = max(start + 1, end - self.chunk_overlap)

        return chunks
