# app.py
import streamlit as st
import pandas as pd
import time

# Import backend (must be in same directory)
try:
    import backend
except Exception as e:
    st.error(f"Cannot import backend.py: {e}")
    st.stop()

from sentence_transformers import SentenceTransformer
import chromadb

# Page + theme suggestion (user can put theme in .streamlit/config.toml)
st.set_page_config(page_title="Chunking Optimizer", layout="wide")
st.title("📦 Chunking Optimizer — Sequential Flow")

# Sidebar progress UI helper
STAGES = ["upload", "dtype", "layer1", "layer2", "quality", "chunk", "embed", "store", "retrieve"]
LABELS = {
    "upload": "1. Upload",
    "dtype": "2. DTypes",
    "layer1": "3. Preprocess L1",
    "layer2": "4. Preprocess L2",
    "quality": "5. Quality",
    "chunk": "6. Chunking",
    "embed": "7. Embedding",
    "store": "8. Store",
    "retrieve": "9. Retrieve",
}

def render_progress(stage):
    st.sidebar.title("Progress")
    for s in STAGES:
        if STAGES.index(s) < STAGES.index(stage):
            st.sidebar.markdown(f"✅ <span style='color:green'>{LABELS[s]}</span>", unsafe_allow_html=True)
        elif s == stage:
            st.sidebar.markdown(f"🟠 <span style='color:orange'>{LABELS[s]}</span>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"⚪ <span style='color:grey'>{LABELS[s]}</span>", unsafe_allow_html=True)

# session init
if "stage" not in st.session_state:
    st.session_state.stage = "upload"
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
if "metadatas" not in st.session_state:
    st.session_state.metadatas = None

def goto(s): st.session_state.stage = s

render_progress(st.session_state.stage)

# -------------------------
# Upload
# -------------------------
if st.session_state.stage == "upload":
    st.header("Step 1 — Upload CSV")
    uploaded = st.file_uploader("Upload CSV (only .csv)", type=["csv"], key="upload_file")
    if uploaded:
        try:
            df = backend.load_csv(uploaded)
            # normalize column headers simple
            df.columns = [str(c).strip() for c in df.columns]
            st.session_state.df = df
            st.success("Loaded CSV")
            st.subheader("Preview (random 5 rows)")
            st.dataframe(backend.preview_data(df, 5))
            st.subheader("Column dtypes")
            st.table(pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}))
            if st.button("Confirm upload & Proceed", key="confirm_upload"):
                goto("dtype")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")

# -------------------------
# dtype step
# -------------------------
elif st.session_state.stage == "dtype":
    st.header("Step 2 — Change datatypes (if needed)")
    df = st.session_state.df
    st.table(pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}))
    cols = df.columns.tolist()
    cols_to_change = st.multiselect("Select columns to change dtype", options=cols, key="dtype_sel")
    dtype_map = {}
    for c in cols_to_change:
        dtype_map[c] = st.selectbox(f"Target dtype for {c}", ["Keep","str","int","float","datetime"], key=f"dtype_{c}")
    if st.button("Apply dtype changes", key="apply_dtype"):
        df2 = df.copy()
        applied=[]
        for c,t in dtype_map.items():
            if t=="Keep": continue
            df2,err = backend.change_dtype(df2, c, t)
            if err: st.warning(f"{c} -> {t} failed: {err}")
            else: applied.append(c)
        st.session_state.df = df2
        st.success(f"Applied: {applied}" if applied else "No dtype changes")
        st.dataframe(backend.preview_data(st.session_state.df,5))
    if st.button("Proceed to Preprocessing L1", key="to_l1"):
        goto("layer1")

# -------------------------
# Preprocess Layer 1
# -------------------------
elif st.session_state.stage == "layer1":
    st.header("Step 3 — Preprocessing Layer 1")
    df = st.session_state.df
    # detect HTML
    html_cols = [c for c in df.columns if df[c].astype(str).str.contains(r"<.*?>", na=False).any()]
    if html_cols:
        st.info(f"HTML-like content found in: {html_cols}")
        opt_remove_html = st.checkbox("Remove HTML tags (detected columns)", key="l1_html")
    else:
        st.info("No HTML detected")
        opt_remove_html = False
    opt_lower = st.checkbox("Convert text columns to lowercase", key="l1_lower")
    if st.button("Apply Layer1", key="apply_l1"):
        df2 = df.copy()
        if opt_remove_html and html_cols:
            df2 = backend.remove_html(df2)
        if opt_lower:
            df2 = backend.to_lowercase(df2)
        st.session_state.df = df2
        st.success("Layer1 applied")
        st.dataframe(backend.preview_data(df2,5))
    if st.button("Proceed to Preprocessing L2", key="to_l2"):
        goto("layer2")

# -------------------------
# Preprocess Layer 2
# -------------------------
elif st.session_state.stage == "layer2":
    st.header("Step 4 — Preprocessing Layer 2")
    df = st.session_state.df
    total_missing = int(df.isnull().sum().sum())
    dup_count = int(df.duplicated().sum())
    st.write(f"Missing values total: **{total_missing}** — Duplicate rows: **{dup_count}**")
    remove_dup = False
    if dup_count > 0:
        remove_dup = st.checkbox("Remove duplicate rows", key="l2_dup")
    if total_missing > 0:
        missing_choice = st.selectbox("Missing handling", ["none","drop","fill"], key="l2_missing")
        fill_value = None
        if missing_choice == "fill":
            fill_value = st.text_input("Fill value", value="Unknown", key="l2_fill")
    else:
        st.info("No missing values detected.")
        missing_choice = "none"
        fill_value = None

    # text normalization
    text_cols = [c for c in df.columns if df[c].dtype == object]
    norm_stop = st.checkbox("Remove stopwords (text cols)", key="l2_stop")
    norm_stem = st.checkbox("Apply stemming", key="l2_stem")
    norm_lemma = st.checkbox("Apply lemmatization", key="l2_lemma")

    if st.button("Apply Layer2", key="apply_l2"):
        df2 = df.copy()
        if missing_choice == "drop":
            df2 = backend.handle_missing(df2, "drop")
        elif missing_choice == "fill":
            df2 = backend.handle_missing(df2, "fill", fill_value)
        if remove_dup:
            df2 = backend.drop_duplicates(df2)
        # rudimentary normalization: lowercase already done; advanced (nltk) left as optional
        st.session_state.df = df2
        st.success("Layer2 applied")
        st.dataframe(backend.preview_data(df2,5))

    if st.button("Proceed to Quality Gate", key="to_quality"):
        goto("quality")

# -------------------------
# Quality Gate
# -------------------------
elif st.session_state.stage == "quality":
    st.header("Step 5 — Quality Gate")
    df = st.session_state.df
    st.metric("Rows", len(df))
    st.metric("Columns", len(df.columns))
    st.metric("Missing (total)", int(df.isnull().sum().sum()))
    st.metric("Duplicate rows", int(df.duplicated().sum()))
    # simple pass criteria
    missing_frac = df.isnull().sum().sum() / max(1, (len(df)*len(df.columns)))
    dup_frac = int(df.duplicated().sum()) / max(1, len(df))
    passed = (missing_frac <= 0.2) and (dup_frac <= 0.05)
    if passed:
        st.success("Quality gate PASSED")
    else:
        st.warning("Quality gate FAILED — please fix before chunking")
    if st.button("Proceed to Chunking", key="to_chunk"):
        if not passed:
            st.warning("Proceeding despite quality gate failing (you chose to continue).")
        goto("chunk")

# -------------------------
# Chunking
# -------------------------
elif st.session_state.stage == "chunk":
    st.header("Step 6 — Chunking")
    df = st.session_state.df
    st.write("Chunking strategies available: Fixed, Recursive, Semantic+Recursive")
    choice = st.selectbox("Chunking strategy", ["Fixed Size", "Recursive (langchain)", "Semantic+Recursive"], key="chunk_choice")
    chunk_size = st.number_input("Chunk size (chars)", min_value=50, max_value=5000, value=400, key="chunk_size")
    overlap = st.number_input("Chunk overlap", min_value=0, max_value=chunk_size-1, value=50, key="chunk_overlap")
    if st.button("Run Chunking", key="run_chunk"):
        with st.spinner("Creating chunks..."):
            try:
                if choice == "Fixed Size":
                    chunks = backend.fixed_size_chunking_from_df(df, chunk_size=int(chunk_size), overlap=int(overlap))
                    metadatas = None
                elif choice == "Recursive (langchain)":
                    chunks = backend.recursive_chunk(df, chunk_size=int(chunk_size), overlap=int(overlap))
                    # build per-row metadata but note: recursive chunking splits across many chunks,
                    # this simple approach will not map chunks->row perfectly; we will provide row-level metadata list
                    metadatas = backend.build_row_metadatas(df)
                else:
                    chunks = backend.semantic_recursive_chunk(df, chunk_size=int(chunk_size), overlap=int(overlap))
                    metadatas = backend.build_row_metadatas(df)

                st.session_state.chunks = chunks
                st.session_state.metadatas = metadatas
                st.success(f"Created {len(chunks)} chunks")
                st.write("Sample chunks (first 3):")
                for c in chunks[:3]:
                    st.code(c[:400])
                # show space usage comparison
                stats = backend.compare_space_usage_all(df, chunk_size=int(chunk_size), overlap=int(overlap))
                st.subheader("Space usage (chars)")
                st.write(stats)
            except Exception as e:
                st.error(f"Chunking failed: {e}")

    if st.session_state.chunks:
        if st.button("Proceed to Embedding", key="to_embed"):
            goto("embed")

# -------------------------
# Embedding
# -------------------------
elif st.session_state.stage == "embed":
    st.header("Step 7 — Embedding")
    chunks = st.session_state.chunks or []
    if not chunks:
        st.warning("No chunks to embed")
    else:
        model_choice = st.selectbox("Model for embeddings", ["all-MiniLM-L6-v2","paraphrase-MiniLM-L6-v2","all-mpnet-base-v2"], key="model_select")
        batch_size = st.number_input("Batch size", 1, 1024, 64, key="emb_batch")
        if st.button("Generate embeddings (in session)", key="gen_emb"):
            with st.spinner("Computing embeddings..."):
                try:
                    model = SentenceTransformer(model_choice)
                    embeddings = model.encode(chunks, batch_size=int(batch_size), show_progress_bar=True)
                    st.session_state.embeddings = embeddings
                    st.session_state.model_name = model_choice
                    st.session_state.model_obj = model
                    st.success("Embeddings created and stored in session")
                    st.write("Vector length:", len(embeddings[0]) if len(embeddings)>0 else 0)
                except Exception as e:
                    st.error(f"Failed to embed: {e}")
        if st.session_state.embeddings is not None:
            if st.button("Proceed to Storage", key="to_store"):
                goto("store")

# -------------------------
# Store (Chroma)
# -------------------------
elif st.session_state.stage == "store":
    st.header("Step 8 — Store in ChromaDB (optional)")
    chunks = st.session_state.chunks
    embeddings = st.session_state.embeddings
    model_choice = st.session_state.model_name
    metadatas = st.session_state.metadatas
    if chunks is None or embeddings is None:
        st.warning("Chunks/embeddings missing")
    else:
        store = st.radio("Store in ChromaDB?", ["No","Yes"], key="store_radio")
        coll_name = st.text_input("Collection name", value="my_collection", key="coll_name")
        if store=="Yes" and st.button("Store now", key="do_store"):
            with st.spinner("Storing in Chroma..."):
                try:
                    client = chromadb.PersistentClient(path="chromadb_store")
                    # create/get collection
                    try:
                        collection = client.get_collection(coll_name)
                    except Exception:
                        collection = client.create_collection(coll_name)
                    # clear existing
                    try:
                        ex = collection.get()
                        if "ids" in ex and ex["ids"]:
                            collection.delete(ids=ex["ids"])
                    except Exception:
                        pass
                    ids = [str(i) for i in range(len(chunks))]
                    emb_lists = [list(map(float,e)) for e in embeddings]
                    # if metadatas present and same length, add
                    if (metadatas is not None) and (len(metadatas) == len(chunks)):
                        collection.add(ids=ids, documents=chunks, embeddings=emb_lists, metadatas=metadatas)
                    else:
                        # If metadatas None or mismatch -> add without metadatas
                        collection.add(ids=ids, documents=chunks, embeddings=emb_lists)
                    st.session_state.collection = collection
                    st.success(f"Stored {len(chunks)} in collection {coll_name}")
                except Exception as e:
                    st.error(f"Store failed: {e}")
        if st.session_state.collection:
            if st.button("Proceed to Retrieval", key="to_retrieve"):
                goto("retrieve")

# -------------------------
# Retrieve / Search
# -------------------------
elif st.session_state.stage == "retrieve":
    st.header("Step 9 — Semantic Retrieval")
    if not st.session_state.collection or not st.session_state.model_obj:
        st.error("No collection or model available (store embeddings first).")
    else:
        q = st.text_input("Query (semantic):", key="query_text")
        k = st.slider("Top-k", 1, 20, 5, key="topk")
        # metadata filter UI (simple numeric min-max)
        col_opts = [c for c in st.session_state.df.columns if pd.api.types.is_numeric_dtype(st.session_state.df[c])]
        selected_num_col = st.selectbox("Optional numeric filter column (metadata)", ["None"] + col_opts, key="meta_col")
        num_min, num_max = None, None
        if selected_num_col != "None":
            col_series = st.session_state.df[selected_num_col].dropna().astype(float)
            if not col_series.empty:
                num_min = float(col_series.min())
                num_max = float(col_series.max())
                r_min, r_max = st.slider("Range", float(num_min), float(num_max), (float(num_min), float(num_max)), key="meta_range")
            else:
                r_min, r_max = None, None
        else:
            r_min, r_max = None, None

        if st.button("Search now", key="search_btn"):
            with st.spinner("Searching..."):
                try:
                    model = st.session_state.model_obj
                    coll = st.session_state.collection
                    q_emb = model.encode([q])
                    res = coll.query(query_embeddings=q_emb, n_results=k, include=["documents","metadatas","distances"])
                    docs = res.get("documents", [[]])[0]
                    dists = res.get("distances", [[]])[0]
                    metas = res.get("metadatas", [[]])[0]
                    # combine
                    combined = []
                    for doc, meta, dist in zip(docs, metas, dists):
                        combined.append((doc, meta, float(dist)))
                    # If user provided numeric filter, apply client-side filter using metadata dict
                    if selected_num_col != "None" and r_min is not None:
                        filtered = []
                        for doc, meta, dist in combined:
                            if not meta:
                                continue
                            val = meta.get(selected_num_col)
                            try:
                                if val is None:
                                    continue
                                valf = float(val)
                                if r_min <= valf <= r_max:
                                    filtered.append((doc, meta, dist))
                            except Exception:
                                # non-numeric metadata -> skip
                                continue
                        combined = filtered
                    if not combined:
                        st.info("No results found (after metadata filtering if applied).")
                    else:
                        # sort by ascending distance (lower = more similar)
                        combined = sorted(combined, key=lambda x: x[2])
                        st.success(f"Found {len(combined)} results (top {k} requested):")
                        for i, (doc, meta, dist) in enumerate(combined):
                            st.markdown(f"**Rank {i+1} (dist {dist:.4f})**")
                            st.write(doc)
                            if meta:
                                st.json(meta)
                            else:
                                st.caption("No metadata attached.")
                except Exception as e:
                    st.error(f"Search failed: {e}")

# end