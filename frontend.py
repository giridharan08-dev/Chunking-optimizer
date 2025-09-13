# app.py
import streamlit as st
import pandas as pd
import time

# Import backend (must be in same folder)
try:
    import backend
except Exception as e:
    st.error(f"Cannot import backend.py: {e}")
    st.stop()

from sentence_transformers import SentenceTransformer
import chromadb

# -------------------------
# Page + styles
# -------------------------
st.set_page_config(page_title="Chunking Optimizer", layout="wide")

st.markdown("""
    <style>
        /* Orange headings */
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #FF8000 !important;
        }
        /* Grey helper text */
        .grey-text {
            color: #808080 !important;
            font-size: 14px;
        }
        .stButton > button {
            background-color: #FF8000 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📦 Chunking Optimizer — Sequential Flow")

# -------------------------
# Sidebar progress UI helper
# -------------------------
STAGES = ["upload", "dtype", "layer1", "layer2", "quality", "chunk", "embed", "store", "retrieve"]
LABELS = {
    "upload": "1. Upload",
    "dtype": "2. Data Types",
    "layer1": "3. Preprocessing Layer 1",
    "layer2": "4. Preprocessing Layer 2",
    "quality": "5. Quality Gate",
    "chunk": "6. Chunking",
    "embed": "7. Embedding",
    "store": "8. Store in ChromaDB",
    "retrieve": "9. Retrieval"
}

def render_sidebar_progress(current_stage):
    st.sidebar.title("🚦 Progress Tracker")
    for stage in STAGES:
        label = LABELS[stage]
        if STAGES.index(stage) < STAGES.index(current_stage):
            st.sidebar.markdown(f"✅ <span style='color:green'>{label}</span>", unsafe_allow_html=True)
        elif stage == current_stage:
            st.sidebar.markdown(f"🟠 <span style='color:orange;font-weight:bold'>{label}</span>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"⚪ <span style='color:grey'>{label}</span>", unsafe_allow_html=True)

# -------------------------
# Session-state initialization
# -------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "upload"

# data containers
for k in ["df", "chunks", "embeddings", "model_name", "collection", "model_obj", "metadatas"]:
    if k not in st.session_state:
        st.session_state[k] = None

def goto(stage_name):
    st.session_state.stage = stage_name

render_sidebar_progress(st.session_state.stage)

# -------------------------
# STAGE: Upload
# -------------------------
if st.session_state.stage == "upload":
    st.header("Step 1 — Upload CSV")
    uploaded = st.file_uploader("Upload a CSV file (only .csv). After upload click 'Confirm upload' to proceed.", type=["csv"], key="file_upload")
    if uploaded is not None:
        try:
            df = backend.load_csv(uploaded)
            # normalize column headers
            df.columns = [str(c).strip() for c in df.columns]
            st.session_state.df = df
            st.success("✅ File loaded into memory.")
            st.subheader("Preview: random 5 rows")
            st.dataframe(backend.preview_data(df, 5))
            st.subheader("Column datatypes")
            dtypes_df = pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]})
            st.table(dtypes_df)
            st.markdown('<p class="grey-text">Confirm upload before proceeding.</p>', unsafe_allow_html=True)
            if st.button("Confirm upload & proceed to Data Type changes", key="confirm_upload"):
                goto("dtype")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

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

    if st.button("Proceed to Preprocessing Layer 1", key="to_layer1"):
        goto("layer1")

# -------------------------
# STAGE: Preprocessing Layer 1
# -------------------------
elif st.session_state.stage == "layer1":
    st.header("Step 3 — Preprocessing Layer 1")
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

    if st.button("Apply Layer 1 changes", key="apply_layer1"):
        df2 = df.copy()
        if remove_html and html_cols:
            df2 = backend.remove_html(df2)
        if lowercase:
            df2 = backend.to_lowercase(df2)
        st.session_state.df = df2
        st.success("✅ Layer 1 applied.")
        st.dataframe(backend.preview_data(df2, 5))

    if st.button("Proceed to Preprocessing Layer 2", key="to_layer2"):
        goto("layer2")

# -------------------------
# STAGE: Preprocessing Layer 2
# -------------------------
elif st.session_state.stage == "layer2":
    st.header("Step 4 — Preprocessing Layer 2 (duplicates, missing, text normalization)")
    df = st.session_state.df

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
        cols = text_cols  # for preview normalization we use these
        norm_remove_stop = st.checkbox("Remove stopwords (e.g. the, and)", key="layer2_stop")
        norm_stem = st.checkbox("Apply stemming", key="layer2_stem")
        norm_lemma = st.checkbox("Apply lemmatization", key="layer2_lemma")
    else:
        st.info("No text columns to normalize.")

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
        # text normalization using backend.text_normalize
        if len(text_cols) > 0 and any([norm_remove_stop, norm_stem, norm_lemma]):
            try:
                df2 = backend.text_normalize(df2, text_cols, stop=norm_remove_stop, stem=norm_stem, lemma=norm_lemma)
            except Exception as ex_norm:
                st.warning(f"Text normalization failed (nltk issue): {ex_norm}. Skipping normalization.")
        st.session_state.df = df2
        st.success("✅ Layer 2 applied.")
        st.dataframe(backend.preview_data(df2, 5))

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
    else:
        st.warning("Quality gate FAILED. Please go back to Layer 2 and fix issues.")

    if st.button("Proceed to Chunking (you may proceed even if failed)", key="quality_to_chunk"):
        goto("chunk")

# -------------------------
# STAGE: Chunking
# -------------------------
elif st.session_state.stage == "chunk":
    st.header("Step 6 — Chunking Options")
    df = st.session_state.df

    chunking_type = st.selectbox(
        "Chunking strategy",
        [
            "Fixed Size",
            "Recursive (langchain)",
            "Semantic Compression + Recursive",
            "Semantic (Cosine similarity grouping)"
        ],
        key="chunking_type"
    )

    chunk_size = st.number_input("Chunk size (characters)", min_value=50, max_value=5000, value=400, key="chunk_size")
    overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=chunk_size-1, value=50, key="chunk_overlap")

    semantic_threshold = None
    if chunking_type == "Semantic (Cosine similarity grouping)":
        semantic_threshold = st.slider("Similarity threshold (0-1)", 0.0, 1.0, 0.7, 0.01, key="semantic_threshold")

    if st.button("Apply selected chunking", key="apply_chunking"):
        with st.spinner("Creating chunks..."):
            try:
                if chunking_type == "Fixed Size":
                    chunks, metadatas = backend.fixed_size_chunking_from_df(df, chunk_size=int(chunk_size), overlap=int(overlap))
                elif chunking_type == "Recursive (langchain)":
                    chunks, metadatas = backend.recursive_chunk(df, chunk_size=int(chunk_size), overlap=int(overlap))
                elif chunking_type == "Semantic Compression + Recursive":
                    chunks, metadatas = backend.semantic_recursive_chunk(df, chunk_size=int(chunk_size), overlap=int(overlap))
                else:
                    chunks, metadatas = backend.semantic_chunking(df, model_name="all-MiniLM-L6-v2", threshold=float(semantic_threshold))

                st.session_state.chunks = chunks
                st.session_state.metadatas = metadatas
                st.success(f"✅ Created {len(chunks)} chunks using {chunking_type}.")
                st.write("Sample chunks (first 3):")
                for c in chunks[:3]:
                    st.code(c[:1000])
            except Exception as e:
                st.error(f"Chunking failed: {e}")

    # show proceed only if chunks exist
    if st.session_state.chunks and len(st.session_state.chunks) > 0:
        if st.button("Proceed to Embedding", key="chunk_to_embed"):
            goto("embed")

    # show space usage comparison
    try:
        stats = backend.compare_space_usage_all(df, chunk_size=int(chunk_size), overlap=int(overlap))
        st.subheader("📊 Space Usage Comparison")
        st.write(stats)
    except Exception:
        pass

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
    st.markdown("<h2 style='color: orange;'>Step 8 — Store in ChromaDB (optional)</h2>", unsafe_allow_html=True)
    chunks = st.session_state.chunks
    embeddings = st.session_state.embeddings
    model_choice = st.session_state.model_name
    metadatas = st.session_state.metadatas

    if embeddings is None or chunks is None:
        st.warning("Missing chunks or embeddings. Generate embeddings first.")
    else:
        store_choice = st.radio("Store embeddings in ChromaDB?", ["No", "Yes"], index=1, key="store_choice")
        coll_prefix = st.text_input("Collection name", value="my_collection", key="coll_prefix")
        if store_choice == "Yes":
            if st.button("Store in ChromaDB now", key="do_store"):
                with st.spinner("Storing embeddings in ChromaDB..."):
                    try:
                        client = chromadb.PersistentClient(path="chromadb_store")
                        coll_name = coll_prefix
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
                        emb_lists = [list(map(float, e)) for e in embeddings]

                        if (metadatas is not None) and (len(metadatas) == len(chunks)):
                            # sanitize inside backend.embed_and_store OR here
                            collection.add(ids=ids, documents=chunks, embeddings=emb_lists, metadatas=metadatas)
                        else:
                            collection.add(ids=ids, documents=chunks, embeddings=emb_lists)

                        st.session_state.collection = collection
                        st.success(f"✅ Stored {len(chunks)} chunks in collection: {coll_name}")
                    except Exception as e:
                        st.error(f"Failed to store to ChromaDB: {e}")

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

        # for metadata numeric filter, list numeric columns from original df
        numeric_columns = [c for c in st.session_state.df.columns if pd.api.types.is_numeric_dtype(st.session_state.df[c])]
        selected_num_col = st.selectbox("Optional numeric filter column (metadata)", ["None"] + numeric_columns, key="meta_col")

        r_min = r_max = None
        if selected_num_col != "None":
            col_series = st.session_state.df[selected_num_col].dropna().astype(float)
            if not col_series.empty:
                r_min, r_max = st.slider(f"Filter range for {selected_num_col}", float(col_series.min()), float(col_series.max()), (float(col_series.min()), float(col_series.max())), key="meta_range")

        if st.button("Search", key="do_search"):
            with st.spinner("Searching..."):
                try:
                    model = st.session_state.model_obj
                    res = st.session_state.collection.query(query_embeddings=model.encode([query]), n_results=int(topk), include=["documents", "metadatas", "distances"])
                    docs = res.get("documents", [[]])[0]
                    dists = res.get("distances", [[]])[0]
                    metas = res.get("metadatas", [[]])[0]

                    results = []
                    for doc, meta, d in zip(docs, metas, dists):
                        results.append((doc, meta, float(d)))

                    # Apply numeric filter if requested
                    if selected_num_col != "None" and r_min is not None:
                        filtered = []
                        for doc, meta, dist in results:
                            if not meta:
                                continue
                            # check if meta has direct value or aggregated min_/max_
                            if selected_num_col in meta:
                                try:
                                    v = float(meta[selected_num_col])
                                    if r_min <= v <= r_max:
                                        filtered.append((doc, meta, dist))
                                except Exception:
                                    continue
                            else:
                                # look for aggregated keys min_<col> and max_<col>
                                mink = f"min_{selected_num_col}"
                                maxk = f"max_{selected_num_col}"
                                if mink in meta and maxk in meta:
                                    try:
                                        minv = meta[mink]
                                        maxv = meta[maxk]
                                        if minv is None or maxv is None:
                                            continue
                                        # check overlap: chunk contains some value in requested range
                                        if (float(minv) <= float(r_max)) and (float(maxv) >= float(r_min)):
                                            filtered.append((doc, meta, dist))
                                    except Exception:
                                        continue
                                else:
                                    # not present, skip
                                    continue
                        results = filtered

                    if not results:
                        st.info("❌ No results found (after metadata filtering if applied).")
                    else:
                        # sort by ascending distance (lower = more similar)
                        results = sorted(results, key=lambda x: x[2])
                        st.success(f"Found {len(results)} results (ranked by distance):")
                        for i, (doc, meta, dist) in enumerate(results):
                            st.markdown(f"**Rank {i+1} — distance {dist:.4f}**")
                            st.write(doc)
                            if meta and isinstance(meta, dict):
                                st.json(meta)
                            else:
                                st.caption("ℹ️ No metadata available for this result.")
                except Exception as e:
                    st.error(f"Search failed: {e}")

              