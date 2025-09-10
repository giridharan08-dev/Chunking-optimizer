# app.py (updated -- sequential step UI)
import streamlit as st
import pandas as pd
import time

# Try to import backend (must be in same folder)
try:
    import backend
except Exception as e:
    backend = None
    backend_import_err = e

from sentence_transformers import SentenceTransformer
import chromadb

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Chunking Optimizer — Sequential Flow", layout="wide")
st.title("📦 Chunking Optimizer — Step-by-step")

# -------------------------
# Session-state initialization
# -------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "upload" # upload -> dtype -> layer1 -> layer2 -> quality -> chunk -> embed -> store -> retrieve

# store objects
if "df" not in st.session_state:
    st.session_state.df = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "model_name" not in st.session_state:
    st.session_state.model_name = None
if "collection" not in st.session_state:
    st.session_state.collection = None
if "model_obj" not in st.session_state:
    st.session_state.model_obj = None

# helper to update stage
def goto(stage_name):
    st.session_state.stage = stage_name

# show helpful import error if backend not found
if backend is None:
    st.error(f"backend.py import failed: {backend_import_err}. Put backend.py in same folder and restart Streamlit.")
    st.stop()

# -------------------------
# STAGE: Upload
# -------------------------
if st.session_state.stage == "upload":
    st.header("Step 1 — Upload CSV")
    uploaded = st.file_uploader("Upload a CSV file (only .csv). After upload click 'Confirm upload' to proceed.", type=["csv"], key="file_upload")

    if uploaded is not None:
        # dynamically load using backend.load_csv which handles bytes/file-like
        try:
            df = backend.load_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            st.stop()
        st.session_state.df = df
        st.success("✅ File loaded into memory.")
        st.subheader("Preview: random 5 rows")
        st.dataframe(backend.preview_data(df, 5))
        st.subheader("Column datatypes")
        dtypes_df = pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]})
        st.table(dtypes_df)

        if st.button("Confirm upload & proceed to Data Type changes", key="confirm_upload"):
            goto("dtype")

# -------------------------
# STAGE: DataType changes
# -------------------------
elif st.session_state.stage == "dtype":
    st.header("Step 2 — Column datatypes (change if needed)")
    df = st.session_state.df
    st.write("Current datatypes:")
    st.table(pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}))

    st.markdown("Select columns (multi-select) to change datatypes, then apply.")
    cols = df.columns.tolist()
    cols_to_change = st.multiselect("Columns to change", options=cols, key="dtype_multiselect")

    dtype_choices = {}
    for c in cols_to_change:
        dtype_choices[c] = st.selectbox(f"Target dtype for `{c}`", ["Keep", "str", "int", "float", "datetime"], key=f"dtype_choice_{c}")

    if st.button("Apply dtype changes", key="apply_dtype"):
        df2 = df.copy()
        applied_cols = []
        for c, tgt in dtype_choices.items():
            if tgt == "Keep":
                continue
            df2, err = backend.change_dtype(df2, c, tgt)
            if err:
                st.warning(f"Could not convert {c} to {tgt}: {err}")
            else:
                applied_cols.append(c)
        st.session_state.df = df2
        if applied_cols:
            st.success(f"Applied dtype changes: {applied_cols}")
        else:
            st.info("No dtype changes applied.")
        st.dataframe(backend.preview_data(st.session_state.df, 5))

    # proceed only when user explicitly clicks next
    if st.button("Proceed to Preprocessing Layer 1", key="to_layer1"):
        goto("layer1")

# -------------------------
# STAGE: Preprocessing Layer 1
# -------------------------
elif st.session_state.stage == "layer1":
    st.header("Step 3 — Preprocessing Layer 1")
    st.write("Options shown only if relevant.")
    df = st.session_state.df

    # detect html-like columns
    html_cols = [c for c in df.columns if df[c].astype(str).str.contains(r"<.*?>", na=False).any()]
    if html_cols:
        st.info(f"HTML-like content detected in: {html_cols}")
        remove_html = st.checkbox("Remove HTML tags from detected columns", key="layer1_remove_html")
    else:
        st.info("No HTML-like content detected in dataset.")
        remove_html = False

    # lowercase option (always available)
    lowercase = st.checkbox("Convert text columns to lowercase", key="layer1_lowercase")

    # Apply button (only once per run)
    if st.button("Apply Layer 1 changes", key="apply_layer1"):
        df2 = df.copy()
        if remove_html and html_cols:
            df2 = backend.remove_html(df2)
        if lowercase:
            df2 = backend.to_lowercase(df2)
        st.session_state.df = df2
        st.success("✅ Layer 1 applied.")
        st.dataframe(backend.preview_data(df2, 5))
        # show explicit proceed button now (user must click)
    if st.button("Proceed to Preprocessing Layer 2", key="to_layer2"):
        goto("layer2")

# -------------------------
# STAGE: Preprocessing Layer 2
# -------------------------
elif st.session_state.stage == "layer2":
    st.header("Step 4 — Preprocessing Layer 2 (duplicates, missing, text normalization)")
    df = st.session_state.df

    # show detected issues
    total_missing = int(df.isnull().sum().sum())
    dup_count = int(df.duplicated().sum())
    st.write(f"- Total missing values: **{total_missing}**")
    st.write(f"- Duplicate rows: **{dup_count}**")

    # Duplicate handling - shown only if duplicates present
    remove_dup = False
    if dup_count > 0:
        remove_dup = st.checkbox("Remove duplicate rows", key="layer2_remove_dup")
    else:
        st.info("No duplicate rows detected.")

    # Missing value handling - shown only if missing present
    missing_choice = "none"
    fill_val = None
    if total_missing > 0:
        missing_choice = st.selectbox("Missing values handling", ["none", "drop", "fill"], key="layer2_missing_choice")
        if missing_choice == "fill":
            fill_val = st.text_input("Fill value", value="Unknown", key="layer2_fill_val")
    else:
        st.info("No missing values detected.")

    # text normalization options - only if text columns exist
    text_cols = [c for c in df.columns if df[c].dtype == object]
    norm_remove_stop = False
    norm_stem = False
    norm_lemma = False
    if len(text_cols) > 0:
        st.write(f"Text columns detected: {text_cols}")
        norm_remove_stop = st.checkbox("Remove stopwords (e.g. the, and)", key="layer2_stop")
        norm_stem = st.checkbox("Apply stemming", key="layer2_stem")
        norm_lemma = st.checkbox("Apply lemmatization", key="layer2_lemma")
    else:
        st.info("No text columns to normalize.")

    # Apply Layer 2 changes
    if st.button("Apply Layer 2 changes", key="apply_layer2"):
        df2 = df.copy()
        # missing
        if missing_choice == "drop":
            df2 = backend.handle_missing(df2, "drop")
        elif missing_choice == "fill":
            df2 = backend.handle_missing(df2, "fill", fill_val)
        # duplicates
        if remove_dup:
            df2 = backend.drop_duplicates(df2)
        # text normalization: implement simple normalize locally
        if any([norm_remove_stop, norm_stem, norm_lemma]) and len(text_cols) > 0:
            # import nltk tooling here, but keep it safe: backend previously had nltks; if not present do simple ops
            try:
                import nltk
                from nltk.corpus import stopwords
                from nltk.stem import PorterStemmer, WordNetLemmatizer
                from nltk.tokenize import word_tokenize
                # ensure resources (silently)
                try:
                    nltk.data.find("tokenizers/punkt")
                except LookupError:
                    nltk.download("punkt", quiet=True)
                try:
                    nltk.data.find("corpora/stopwords")
                except LookupError:
                    nltk.download("stopwords", quiet=True)
                try:
                    nltk.data.find("corpora/wordnet")
                except LookupError:
                    nltk.download("wordnet", quiet=True)

                stop_words = set(stopwords.words("english"))
                ps = PorterStemmer()
                wnl = WordNetLemmatizer()

                def _normalize_text(s):
                    tokens = word_tokenize(str(s))
                    out = []
                    for t in tokens:
                        if norm_remove_stop and t.lower() in stop_words:
                            continue
                        if norm_lemma:
                            out.append(wnl.lemmatize(t))
                        elif norm_stem:
                            out.append(ps.stem(t))
                        else:
                            out.append(t)
                    return " ".join(out)

                for c in text_cols:
                    df2[c] = df2[c].astype(str).apply(_normalize_text)
            except Exception as ex_norm:
                st.warning(f"Text normalization failed (nltk issue): {ex_norm}. Skipping normalization.")

        st.session_state.df = df2
        st.success("✅ Layer 2 applied.")
        st.dataframe(backend.preview_data(df2, 5))

    # show proceed only after user clicks apply (we cannot know they applied, so require explicit button)
    if st.button("Proceed to Quality Gate", key="to_quality"):
        goto("quality")

# -------------------------
# STAGE: Quality Gate
# -------------------------
elif st.session_state.stage == "quality":
    st.header("Step 5 — Quality Gate")
    df = st.session_state.df
    num_rows = len(df)
    num_cols = len(df.columns)
    total_missing = int(df.isnull().sum().sum())
    dup_count = int(df.duplicated().sum())
    # simple token avg
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if len(text_cols) > 0:
        avg_tokens = df[text_cols].astype(str).apply(lambda row: len(" ".join(row).split()), axis=1).mean()
    else:
        avg_tokens = 0
    st.metric("Rows", num_rows)
    st.metric("Columns", num_cols)
    st.metric("Missing values", total_missing)
    st.metric("Duplicate rows", dup_count)
    st.metric("Avg tokens per row (text cols)", f"{avg_tokens:.1f}")

    # Basic pass criteria (you can tune)
    missing_frac = total_missing / max(1, num_rows * num_cols)
    dup_frac = dup_count / max(1, num_rows)
    passed = (missing_frac <= 0.2) and (dup_frac <= 0.05)

    if passed:
        st.success("Quality gate PASSED")
        if st.button("Proceed to Chunking (Recursive)", key="quality_to_chunk"):
            goto("chunk")
    else:
        st.warning("Quality gate FAILED. Please go back to Layer 2 and fix issues.")
        if st.button("Back to Layer 2", key="quality_back"):
            goto("layer2")

# -------------------------
# STAGE: Chunking (recursive only)
# -------------------------
elif st.session_state.stage == "chunk":
    st.header("Step 6 — Recursive Chunking")
    df = st.session_state.df
    st.write("Recursive chunking will split each row intelligently into chunks.")
    chunk_size = st.number_input("Chunk size (characters)", min_value=100, max_value=2000, value=400, key="chunk_size")
    overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=chunk_size-1, value=50, key="chunk_overlap")
    if st.button("Apply recursive chunking", key="apply_chunking"):
        with st.spinner("Creating chunks..."):
            try:
                chunks = backend.recursive_chunk(df, chunk_size=int(chunk_size), overlap=int(overlap))
                st.session_state.chunks = chunks
                st.success(f"✅ Chunking created {len(chunks)} chunks.")
                st.write("Sample chunks (first 3):")
                for c in chunks[:3]:
                    st.code(c[:1000])
            except Exception as e:
                st.error(f"Chunking failed: {e}")
    # proceed only if chunks exist
    if st.session_state.chunks:
        if st.button("Proceed to Embedding", key="chunk_to_embed"):
            goto("embed")

# -------------------------
# STAGE: Embedding
# -------------------------
elif st.session_state.stage == "embed":
    st.header("Step 7 — Embedding")
    chunks = st.session_state.chunks or []
    if not chunks:
        st.warning("No chunks found. Go back to chunking.")
    else:
        st.write(f"Chunks available: {len(chunks)}")
        model_choice = st.selectbox("Choose SentenceTransformer model", ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"], key="embed_model_select")
        batch_size = st.number_input("Embedding batch size", min_value=1, max_value=1024, value=64, key="embed_batch")
        if st.button("Generate embeddings (in-memory)", key="gen_emb"):
            with st.spinner("Computing embeddings (this may take time)..."):
                try:
                    model = SentenceTransformer(model_choice)
                    embeddings = model.encode(chunks, batch_size=int(batch_size), show_progress_bar=True)
                    st.session_state.embeddings = embeddings
                    st.session_state.model_name = model_choice
                    st.session_state.model_obj = model
                    st.success("✅ Embeddings generated and stored in session.")
                    st.write("First embedding vector length:", len(embeddings[0]) if len(embeddings) else 0)
                except Exception as e:
                    st.error(f"Failed to compute embeddings: {e}")

        if st.session_state.embeddings is not None:
            if st.button("Proceed to Storage (ChromaDB)", key="embed_to_store"):
                goto("store")

# -------------------------
# STAGE: Storage (ChromaDB)
# -------------------------
elif st.session_state.stage == "store":
    st.header("Step 8 — Store in ChromaDB (optional)")
    chunks = st.session_state.chunks
    embeddings = st.session_state.embeddings
    model_choice = st.session_state.model_name
    if embeddings is None or chunks is None:
        st.warning("Missing chunks or embeddings. Generate embeddings first.")
    else:
        store_choice = st.radio("Store embeddings in ChromaDB?", ["No", "Yes"], index=1, key="store_choice")
        coll_prefix = st.text_input("Collection prefix (name)", value="my_collection", key="coll_prefix")
        if store_choice == "Yes":
            if st.button("Store in ChromaDB now", key="do_store"):
                with st.spinner("Storing embeddings in ChromaDB..."):
                    try:
                        # Use backend helper which computes and stores OR we store using chromadb directly
                        # backend.embed_and_store recomputes embeddings; since we already computed, we'll do direct store
                        client = chromadb.PersistentClient(path="chromadb_store")
                        coll_name = f"{coll_prefix}_{model_choice.replace('/', '_')}"
                        try:
                            collection = client.get_collection(coll_name)
                        except Exception:
                            collection = client.create_collection(coll_name)
                        # clear
                        try:
                            existing = collection.get()
                            if "ids" in existing and existing["ids"]:
                                collection.delete(ids=existing["ids"])
                        except Exception:
                            pass
                        ids = [str(i) for i in range(len(chunks))]
                        # ensure embeddings are lists of floats
                        emb_lists = [list(map(float, e)) for e in embeddings]
                        collection.add(ids=ids, documents=chunks, embeddings=emb_lists)
                        st.session_state.collection = collection
                        st.success(f"✅ Stored {len(chunks)} chunks in collection: {coll_name}")
                    except Exception as e:
                        st.error(f"Failed to store to ChromaDB: {e}")

        # proceed to retrieval only if collection present in session
        if st.session_state.collection:
            if st.button("Proceed to Retrieval", key="store_to_retrieve"):
                goto("retrieve")

# -------------------------
# STAGE: Retrieval
# -------------------------
elif st.session_state.stage == "retrieve":
    st.header("Step 9 — Retrieval (semantic search)")
    if not st.session_state.collection or not st.session_state.model_obj:
        st.error("No stored collection or model available. Store embeddings first.")
    else:
        query = st.text_input("Enter search query", key="search_query")
        topk = st.slider("Top-k results", 1, 20, 5, key="topk")
        threshold = st.slider("Optional distance threshold (min distance to accept result)", 0.0, 5.0, 0.9, key="thresh")
        if st.button("Search", key="do_search"):
            with st.spinner("Searching..."):
                try:
                    q_emb = st.session_state.model_obj.encode([query])
                    res = st.session_state.collection.query(query_embeddings=q_emb, n_results=int(topk), include=["documents", "metadatas", "distances"])
                    docs = res.get("documents", [[]])[0]
                    dists = res.get("distances", [[]])[0]
                    metas = res.get("metadatas", [[]])[0]
                    if not docs:
                        st.info("❌ Your object not found in this table.")
                    else:
                        # filter by threshold (if distances exist)
                        filtered = []
                        for doc, meta, d in zip(docs, metas, dists):
                            filtered.append((doc, meta, float(d)))
                        # If threshold used, filter out items with distance > threshold
                        filtered = [f for f in filtered if f[2] <= threshold] if threshold is not None else filtered
                        if not filtered:
                            st.info("❌ No results within distance threshold.")
                        else:
                            st.success(f"Found {len(filtered)} results (ranked by distance):")
                            for i, (doc, meta, dist) in enumerate(filtered):
                                st.markdown(f"**Rank {i+1} — distance {dist:.4f}**")
                                st.write(doc)
                                st.json(meta)
                except Exception as e:
                    st.error(f"Search failed: {e}")

# -------------------------
# End of stages
# -------------------------