"""
RAG query engine — uses Voyage AI (voyage-law-2) for embeddings.
Called directly by streamlit_app.py (no HTTP layer needed).
"""

import os
import anthropic
import voyageai
from supabase import create_client

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_KEY"]
EMBED_MODEL   = "voyage-law-2"   # Voyage's legal-domain model (1024 dims)
TOP_K         = 5

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
vo       = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
claude   = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are LexAssist, a professional legal information assistant for a law firm.

You have access to the firm's own document library — contracts, case files, templates, and SOPs —
provided to you as CONTEXT below. Always prefer information from the context over general knowledge.
When citing from context, mention the source document name.

CRITICAL RULES:
1. Never give definitive legal advice — provide legal information only
2. Always recommend attorney review for specific situations
3. If the context does not contain relevant information, say so clearly — do not fabricate
4. Flag any urgent deadlines or limitation periods prominently
5. End substantive responses with: "This is legal information only. Please consult a qualified attorney."
"""

# ── Core functions ────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    """Embed a single query string using Voyage AI."""
    result = vo.embed([text], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]


def retrieve(query: str, doc_type: str = None) -> list[dict]:
    """Embed the query and retrieve the top-K most similar chunks from Supabase."""
    query_vector = embed_query(query)

    params = {
        "query_embedding": query_vector,
        "match_count":     TOP_K,
    }
    if doc_type:
        params["filter_doc_type"] = doc_type

    result = supabase.rpc("match_documents", params).execute()
    return result.data or []


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for the prompt."""
    if not chunks:
        return "No relevant documents found in the knowledge base."

    lines = ["--- RETRIEVED CONTEXT FROM FIRM DOCUMENT LIBRARY ---\n"]
    for i, chunk in enumerate(chunks, 1):
        source   = chunk.get("metadata", {}).get("source", "unknown")
        doc_type = chunk.get("metadata", {}).get("doc_type", "")
        content  = chunk.get("content", "")
        score    = chunk.get("similarity", 0)
        lines.append(
            f"[{i}] Source: {source} (type: {doc_type}, relevance: {score:.2f})\n{content}\n"
        )
    lines.append("--- END OF CONTEXT ---")
    return "\n".join(lines)


def ask(
    question: str,
    conversation_history: list[dict] = None,
    doc_type_filter: str = None,
) -> str:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks via Voyage AI embeddings
    2. Build augmented system prompt with context
    3. Call Claude with full conversation history
    4. Return the response text
    """
    chunks  = retrieve(question, doc_type=doc_type_filter)
    context = build_context_block(chunks)

    augmented_system = f"{SYSTEM_PROMPT}\n\n{context}"

    messages = list(conversation_history or [])
    messages.append({"role": "user", "content": question})

    response = claude.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 1024,
        system     = augmented_system,
        messages   = messages,
    )

    return response.content[0].text


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What does our standard NDA say about non-solicitation?"
    print(f"\nQuestion: {q}\n")
    print(f"Answer:\n{ask(q)}")
