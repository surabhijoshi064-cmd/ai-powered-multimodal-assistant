from typing import Tuple, List, Optional
from app.retrieval.vector_store import vector_store
from app.providers.embedding_provider import get_embedding_provider
from app.schemas.chat import Citation
from app.config import settings
from app.utils.logging_config import setup_logging

logger = setup_logging(settings.LOG_LEVEL)


class RAGService:
    def __init__(self):
        self.embedding_provider = get_embedding_provider()

    def retrieve_context(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], List[Citation]]:
        """
        Embeds query, retrieves top-k chunks from vector store, sanitizes untrusted context,
        and constructs citation references.
        """
        k = top_k or settings.RAG_TOP_K
        sim_threshold = threshold or settings.RAG_SIMILARITY_THRESHOLD

        # Embed query
        query_vec = self.embedding_provider.embed_query(query)

        # Retrieve relevant chunks
        results = vector_store.similarity_search(
            query_embedding=query_vec,
            top_k=k,
            threshold=sim_threshold,
            conversation_id=conversation_id,
        )

        if not results:
            logger.info("No matching document chunks found above similarity threshold.")
            return None, []

        context_blocks = []
        citations: List[Citation] = []

        for idx, item in enumerate(results, start=1):
            doc_name = item["document_name"]
            page = item.get("page_number")
            page_info = f", Page {page}" if page else ""
            
            # Untrusted context framing to neutralize prompt injections in uploaded docs
            safe_block = (
                f"[Source {idx}: {doc_name}{page_info}]\n"
                f"<untrusted_retrieved_context>\n"
                f"{item['chunk_text']}\n"
                f"</untrusted_retrieved_context>"
            )
            context_blocks.append(safe_block)

            # Build Citation object
            citations.append(
                Citation(
                    document_name=doc_name,
                    page_number=page,
                    chunk_index=item["chunk_index"],
                    snippet=item["chunk_text"][:250],
                    score=item.get("score"),
                )
            )

        rag_context_str = "\n\n".join(context_blocks)
        logger.info(f"Retrieved {len(citations)} relevant chunk(s) for query.")
        return rag_context_str, citations
