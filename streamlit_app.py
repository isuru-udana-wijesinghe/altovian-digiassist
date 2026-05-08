"""
LexAssist RAG — Streamlit frontend
Replaces FastAPI + HTML with a single Streamlit app.
Deploy to Streamlit Community Cloud (share.streamlit.io).
"""

import streamlit as st
import tempfile, os
from pathlib import Path

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexAssist",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Secrets → env vars (Streamlit Cloud injects via st.secrets) ──────────────
for key in ["ANTHROPIC_API_KEY", "SUPABASE_URL", "SUPABASE_KEY", "VOYAGE_API_KEY"]:
    if key in st.secrets and key not in os.environ:
        os.environ[key] = st.secrets[key]

# ── Validate secrets are present ─────────────────────────────────────────────
missing = [k for k in ["ANTHROPIC_API_KEY", "SUPABASE_URL", "SUPABASE_KEY", "VOYAGE_API_KEY"]
           if not os.environ.get(k)]
if missing:
    st.error(f"⚠️ Missing secrets: {', '.join(missing)}. Add them in your Streamlit Cloud app settings → Secrets.")
    st.stop()

# ── Lazy imports (after env vars are set) ────────────────────────────────────
from rag import ask
from ingest import ingest_file

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content}]
if "doc_type" not in st.session_state:
    st.session_state.doc_type = None

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main header */
.lex-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #1a3a5c;
    margin-bottom: 16px;
}
.lex-title { font-size: 1.6rem; font-weight: 700; color: #1a3a5c; margin: 0; }
.lex-subtitle { font-size: 0.85rem; color: #666; margin: 0; }

/* Disclaimer banner */
.disclaimer {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    padding: 8px 14px;
    border-radius: 4px;
    font-size: 0.8rem;
    color: #5d4037;
    margin-bottom: 12px;
}

/* Chat messages */
.stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ LexAssist")
    st.caption("Legal Knowledge Base · RAG-powered")
    st.divider()

    # Document type filter
    st.markdown("### 🗂️ Filter by Document Type")
    DOC_TYPES = {
        "All documents": None,
        "NDAs": "nda",
        "Employment contracts": "employment",
        "Property / Tenancy": "property",
        "Case law": "case_law",
        "SOPs & Policies": "sop",
        "General": "general",
    }
    selected_label = st.selectbox(
        "Retrieve from:",
        list(DOC_TYPES.keys()),
        index=0,
    )
    st.session_state.doc_type = DOC_TYPES[selected_label]

    st.divider()

    # Document ingestion
    st.markdown("### 📄 Add Documents")
    st.caption("Upload PDFs or text files to expand the knowledge base.")
    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("⬆️ Ingest selected files", use_container_width=True, type="primary"):
            progress = st.progress(0, text="Starting ingestion…")
            results = []
            for i, uf in enumerate(uploaded):
                suffix = Path(uf.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                try:
                    progress.progress((i / len(uploaded)), text=f"Ingesting {uf.name}…")
                    ingest_file(tmp_path, original_name=uf.name)
                    results.append(f"✅ {uf.name}")
                except Exception as e:
                    results.append(f"❌ {uf.name}: {e}")
                finally:
                    os.unlink(tmp_path)
            progress.empty()
            for r in results:
                st.write(r)

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("LexAssist provides legal *information* only, not legal advice.")

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lex-header">
  <div>
    <p class="lex-title">⚖️ LexAssist Legal Assistant</p>
    <p class="lex-subtitle">Retrieval-augmented answers from your firm's document library</p>
  </div>
</div>
<div class="disclaimer">
  ⚠️ <strong>Legal information only.</strong>
  LexAssist draws from your firm's uploaded documents. Always have a qualified attorney
  review advice specific to your matter.
</div>
""", unsafe_allow_html=True)

# Show active filter badge
if st.session_state.doc_type:
    st.info(f"🔍 Searching only **{selected_label}** — change in sidebar to search all documents.")

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="⚖️" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

# ── Suggested starter questions (shown when chat is empty) ────────────────────
if not st.session_state.messages:
    st.markdown("#### 💬 Try asking:")
    cols = st.columns(2)
    starters = [
        "What does our standard NDA say about non-solicitation?",
        "What are the notice periods in our employment contracts?",
        "Summarise the key obligations in our tenancy agreements.",
        "What are the penalties for breach of contract?",
    ]
    for i, q in enumerate(starters):
        if cols[i % 2].button(q, use_container_width=True, key=f"starter_{i}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a legal question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("Searching documents and drafting answer…"):
            # Build conversation history in Anthropic format (exclude last user msg)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            try:
                answer = ask(
                    question=prompt,
                    conversation_history=history,
                    doc_type_filter=st.session_state.doc_type,
                )
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
