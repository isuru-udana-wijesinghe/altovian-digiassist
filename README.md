# LexAssist RAG — Legal Knowledge Base

A retrieval-augmented generation (RAG) assistant for law firms.
Powered by **Claude** (LLM) + **Voyage AI** (`voyage-law-2` embeddings) + **Supabase** (pgvector).

---

## Project structure

```
streamlit_app.py      ← Streamlit UI (replaces FastAPI + HTML)
rag.py                ← RAG query engine
ingest.py             ← Document ingestion (CLI + called by UI)
requirements.txt      ← Python dependencies
supabase_setup.sql    ← Run once in Supabase SQL editor
.streamlit/
  secrets.toml        ← API keys (do NOT commit to Git)
```

---

## Deployment — Streamlit Community Cloud

### 1. Get your API keys

| Key | Where to get it |
|-----|----------------|
| `ANTHROPIC_API_KEY` | console.anthropic.com |
| `SUPABASE_URL` + `SUPABASE_KEY` | Supabase project → Settings → API |
| `VOYAGE_API_KEY` | dash.voyageai.com |

> **Voyage AI free tier** includes 50M tokens/month — more than enough to get started.
> `voyage-law-2` is their purpose-built legal embedding model.

### 2. Set up Supabase

1. Create a free project at [supabase.com](https://supabase.com)
2. Open **SQL Editor → New Query**
3. Paste the contents of `supabase_setup.sql` and click **Run**

⚠️ If you previously ran the old setup (all-MiniLM, 384 dims), you must reset first:
```sql
DROP TABLE IF EXISTS documents;
```
Then re-run `supabase_setup.sql`. Re-ingest your documents afterward.

### 3. Push to GitHub

```bash
git init
git add streamlit_app.py rag.py ingest.py requirements.txt supabase_setup.sql
# Do NOT add .streamlit/secrets.toml
echo ".streamlit/secrets.toml" >> .gitignore
git commit -m "Initial LexAssist Streamlit app"
git remote add origin https://github.com/your-org/lexassist.git
git push -u origin main
```

### 4. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Connect your GitHub repo, set **Main file path** to `streamlit_app.py`
3. Click **Advanced settings → Secrets** and paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
SUPABASE_URL      = "https://your-project.supabase.co"
SUPABASE_KEY      = "your-supabase-key"
VOYAGE_API_KEY    = "pa-..."
```

4. Click **Deploy** — done! 🎉

### 5. Ingest documents

**Option A — from the app:** Use the sidebar file uploader after deploying.

**Option B — CLI (faster for bulk):**
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml .env   # or set env vars manually
python ingest.py --dir ./legal_docs
```

---

## Document naming conventions

The ingester auto-tags document types from filenames:

| Prefix | Type |
|--------|------|
| `nda_*`, `non-disclosure_*` | `nda` |
| `employment_*`, `contract_of_service_*` | `employment` |
| `lease_*`, `tenancy_*`, `rental_*` | `property` |
| `case_*`, `judgment_*`, `ruling_*` | `case_law` |
| `sop_*`, `procedure_*`, `policy_*` | `sop` |
| anything else | `general` |

---

## Why Voyage AI instead of sentence-transformers?

- **No heavy ML runtime** — Voyage is an API call; `sentence-transformers` requires PyTorch (~1 GB), which would exhaust Streamlit Cloud's free-tier memory.
- **Legal-domain model** — `voyage-law-2` is specifically trained on legal text, giving meaningfully better retrieval on contracts, case law, and SOPs.
- **Generous free tier** — 50M tokens/month at no cost.
