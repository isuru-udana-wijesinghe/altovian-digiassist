"""
Document ingestion — embeds PDFs/text files and stores chunks in Supabase.
Uses Voyage AI (voyage-law-2) instead of sentence-transformers.

CLI usage:
    python ingest.py --dir ./legal_docs
    python ingest.py --file contract.pdf

Called directly by streamlit_app.py for in-app uploads.
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path

import voyageai
import PyPDF2
from supabase import create_client

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_KEY"]
EMBED_MODEL   = "voyage-law-2"   # 1024-dim legal embedding model
CHUNK_SIZE    = 500              # approx words per chunk
CHUNK_OVERLAP = 80               # overlap between adjacent chunks
BATCH_SIZE    = 50               # rows per Supabase insert
VOYAGE_BATCH  = 128              # max texts per Voyage embed call

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
vo       = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

# ── Text helpers ──────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    pages = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    step  = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def file_hash(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def doc_type_from_filename(filename: str) -> str:
    name = filename.lower()
    if any(k in name for k in ["nda", "non-disclosure"]):            return "nda"
    if any(k in name for k in ["employment", "contract_of_service"]): return "employment"
    if any(k in name for k in ["lease", "tenancy", "rental"]):        return "property"
    if any(k in name for k in ["case", "judgment", "ruling"]):        return "case_law"
    if any(k in name for k in ["sop", "procedure", "policy"]):        return "sop"
    return "general"


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed all chunks via Voyage AI in batches, with input_type='document'."""
    vectors = []
    for i in range(0, len(chunks), VOYAGE_BATCH):
        batch  = chunks[i : i + VOYAGE_BATCH]
        result = vo.embed(batch, model=EMBED_MODEL, input_type="document")
        vectors.extend(result.embeddings)
    return vectors

# ── Main ingest function ──────────────────────────────────────────────────────

def ingest_file(path: str, original_name: str = None):
    """
    Ingest a single file into Supabase.
    `original_name` is used when the file is a temp upload (the tmp path
    has no meaningful name, so the UI passes the original filename).
    """
    p    = Path(path)
    name = original_name or p.name
    print(f"\n→ Ingesting: {name}")

    # Extract text
    if p.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(str(p))
    elif p.suffix.lower() in [".txt", ".md"]:
        text = p.read_text(encoding="utf-8")
    else:
        print(f"  Skipping unsupported type: {p.suffix}")
        return

    if not text.strip():
        print("  Warning: no text extracted, skipping.")
        return

    # Dedup by file hash
    fhash    = file_hash(str(p))
    existing = supabase.table("documents").select("id").eq("file_hash", fhash).execute()
    if existing.data:
        print(f"  Already ingested (hash match), skipping.")
        return

    # Chunk
    chunks = chunk_text(text)
    print(f"  {len(text):,} chars → {len(chunks)} chunks")

    # Embed via Voyage AI
    print(f"  Embedding with {EMBED_MODEL}…")
    vectors = embed_chunks(chunks)

    # Build rows
    doc_type = doc_type_from_filename(name)
    rows = [
        {
            "content":   chunk,
            "embedding": vector,
            "metadata": {
                "source":    name,
                "doc_type":  doc_type,
                "chunk_idx": idx,
                "file_hash": fhash,
            },
            "file_hash": fhash,
        }
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    # Upsert in batches
    total_batches = (len(rows) - 1) // BATCH_SIZE + 1
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        supabase.table("documents").insert(batch).execute()
        print(f"  Stored batch {i // BATCH_SIZE + 1}/{total_batches}")

    print(f"  ✓ Done — {len(chunks)} chunks stored for {name}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ingest legal documents into LexAssist")
    parser.add_argument("--dir",  help="Directory of documents to ingest")
    parser.add_argument("--file", help="Single file to ingest")
    args = parser.parse_args()

    if args.dir:
        for ext in ["*.pdf", "*.txt", "*.md"]:
            for fpath in Path(args.dir).glob(ext):
                ingest_file(str(fpath))
    elif args.file:
        ingest_file(args.file)
    else:
        print("Usage: python ingest.py --dir ./docs  OR  --file contract.pdf")
        sys.exit(1)

    print("\n✓ Ingestion complete.")
