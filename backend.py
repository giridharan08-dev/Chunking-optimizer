# backend.py
import io, csv, typing, re, math
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# Text splitter import (langchain_text_splitters). Try both names.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# optional: encoding detection
try:
    import chardet
except Exception:
    chardet = None

# ----------------------------
# CSV Loading / Utilities
# ----------------------------
def _detect_sep(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text[:4096])
        return dialect.delimiter
    except Exception:
        return ","

def load_csv(file_or_path: typing.Union[str, io.BytesIO, io.StringIO]):
    """Load CSV from path or stream (Streamlit uploaded file)"""
    if isinstance(file_or_path, str):
        return pd.read_csv(file_or_path)

    if hasattr(file_or_path, "read"):
        file_or_path.seek(0)
        raw = file_or_path.read()

        if isinstance(raw, (bytes, bytearray)):
            encoding = "utf-8"
            if chardet:
                try:
                    res = chardet.detect(raw)
                    encoding = res.get("encoding") or "utf-8"
                except Exception:
                    pass
            text = raw.decode(encoding, errors="replace")
            sep = _detect_sep(text)
            return pd.read_csv(io.StringIO(text), sep=sep)

        if isinstance(raw, str):
            sep = _detect_sep(raw)
            return pd.read_csv(io.StringIO(raw), sep=sep)

    raise ValueError("Unsupported input for load_csv")

def preview_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    n = min(len(df), n)
    if n <= 0:
        return df.head(0)
    return df.sample(n=n, random_state=42).reset_index(drop=True)

def change_dtype(df: pd.DataFrame, col: str, dtype: str):
    """Try converting dtype; returns (df, error_string_or_None)"""
    try:
        if dtype == "str":
            df[col] = df[col].astype(str)
        elif dtype == "int":
            df[col] = pd.to_numeric(df[col], errors="raise").astype("Int64")
        elif dtype == "float":
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        elif dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors="raise")
        else:
            return df, f"Unsupported dtype: {dtype}"
        return df, None
    except Exception as e:
        return df, str(e)

# ----------------------------
# Simple Preprocessing helpers
# ----------------------------
def remove_html(df: pd.DataFrame) -> pd.DataFrame:
    clean_re = re.compile(r"<.*?>")
    df2 = df.copy()
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).apply(lambda s: re.sub(clean_re, "", s))
    return df2

def to_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).str.lower()
    return df2

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)

def handle_missing(df: pd.DataFrame, strategy: str, fill_value: typing.Any = "unknown") -> pd.DataFrame:
    if strategy == "drop":
        return df.dropna().reset_index(drop=True)
    elif strategy == "fill":
        return df.fillna(fill_value)
    else:
        return df

# ----------------------------
# Text Normalization (stopwords / stemming / lemmatization)
# ----------------------------
def text_normalize(df: pd.DataFrame, text_cols: typing.List[str], stop=False, stem=False, lemma=False):
    """
    Normalize text columns with optional stopword removal / stemming / lemmatization.
    Returns a new dataframe.
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        from nltk.tokenize import word_tokenize
    except Exception:
        raise RuntimeError("nltk is required for text_normalize: install nltk and required corpora")

    # ensure resources
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

    stop_words = set(stopwords.words("english")) if stop else set()
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()

    def _norm(s: str):
        tokens = word_tokenize(str(s))
        out = []
        for t in tokens:
            if stop and t.lower() in stop_words:
                continue
            if lemma:
                out.append(wnl.lemmatize(t))
            elif stem:
                out.append(ps.stem(t))
            else:
                out.append(t)
        return " ".join(out)

    df2 = df.copy()
    for c in text_cols:
        df2[c] = df2[c].astype(str).apply(_norm)
    return df2

# ----------------------------
# Helper: make a per-row metadata dict (simple)
# ----------------------------
def _row_metadata(row: pd.Series, numeric_cols: typing.List[str], small_cat: typing.List[str]):
    md = {}
    for c in numeric_cols:
        v = row.get(c)
        if pd.isna(v):
            md[c] = None
        else:
            try:
                md[c] = float(v)
            except Exception:
                try:
                    md[c] = float(str(v))
                except Exception:
                    md[c] = None
    for c in small_cat:
        v = row.get(c)
        md[c] = None if pd.isna(v) else str(v)
    md["_row_index"] = int(row.name) if hasattr(row, "name") else None
    return md

# ----------------------------
# Chunking functions (return chunks + per-chunk metadata)
# ----------------------------
def fixed_size_chunking_from_text(text: str, chunk_size: int = 400, overlap: int = 50):
    chunks = []
    if chunk_size <= 0:
        return [text]
    step = chunk_size - overlap if chunk_size > overlap else chunk_size
    for i in range(0, len(text), step):
        chunk = text[i:i+chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks

def fixed_size_chunking_from_df(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50):
    """
    Returns: (chunks:list[str], metadatas:list[dict])
    Each chunk gets the metadata of the row it came from (row-level metadata).
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    small_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()[:2]
    chunks = []
    metadatas = []
    for idx, row in df.iterrows():
        row_text = " | ".join([f"{c}: {row[c]}" for c in df.columns])
        row_chunks = fixed_size_chunking_from_text(row_text, chunk_size=chunk_size, overlap=overlap)
        md = _row_metadata(row, numeric_cols, small_cat)
        for rc in row_chunks:
            chunks.append(rc)
            metadatas.append(md.copy())
    return chunks, metadatas

def recursive_chunk(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50):
    """
    Use RecursiveCharacterTextSplitter per-row, so metadata aligns with each chunk.
    Returns (chunks, metadatas).
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    small_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()[:2]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    metadatas = []
    for idx, row in df.iterrows():
        row_text = ", ".join([f"{c}: {row[c]}" for c in df.columns])
        try:
            row_chunks = splitter.split_text(row_text)
        except Exception:
            # fallback to simple fixed splitting if splitter fails
            row_chunks = fixed_size_chunking_from_text(row_text, chunk_size=chunk_size, overlap=overlap)
        md = _row_metadata(row, numeric_cols, small_cat)
        for rc in row_chunks:
            chunks.append(rc)
            metadatas.append(md.copy())
    return chunks, metadatas

# Semantic compression (row -> short descriptive sentence)
def semantic_compression(df: pd.DataFrame):
    compressed_rows = []
    for _, row in df.iterrows():
        try:
            if "Description" in row.index and "Country" in row.index:
                description = str(row.get("Description", "item"))
                country = str(row.get("Country", "Unknown"))
                date = str(row.get("InvoiceDate", ""))
                qty = str(row.get("Quantity", ""))
                price = str(row.get("UnitPrice", ""))
                compressed = f"{description} sold in {country} on {date}, qty {qty}, price ${price}"
            else:
                pieces = []
                for c, v in row.items():
                    val_str = "" if pd.isna(v) else str(v)
                    pieces.append(f"{c}:{val_str[:30]}")
                compressed = "; ".join(pieces)
        except Exception:
            compressed = "record"
        compressed_rows.append(compressed)
    return compressed_rows

def semantic_recursive_chunk(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50):
    """
    Compress each row semantically then split per-row. Returns (chunks, metadatas).
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    small_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()[:2]
    compressor = semantic_compression(df)  # list aligned with df rows
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    metadatas = []
    for idx, (row, compressed_text) in enumerate(zip(df.itertuples(index=False), compressor)):
        # build a metadata dict for the row
        row_series = df.iloc[idx]
        md = _row_metadata(row_series, numeric_cols, small_cat)
        try:
            row_chunks = splitter.split_text(compressed_text)
        except Exception:
            row_chunks = fixed_size_chunking_from_text(compressed_text, chunk_size=chunk_size, overlap=overlap)
        for rc in row_chunks:
            chunks.append(rc)
            metadatas.append(md.copy())
    return chunks, metadatas

# ----------------------------
# Semantic Chunking (group rows by semantic similarity into chunks)
# Returns (chunks, metadatas) where per-chunk metadata aggregates numeric columns.
# ----------------------------
from sklearn.metrics.pairwise import cosine_similarity

def semantic_chunking(df: pd.DataFrame,
                      model_name: str = "all-MiniLM-L6-v2",
                      threshold: float = 0.7):
    """
    Group rows into semantic chunks using cosine similarity of row embeddings.
    threshold: similarity threshold to add a new row to the current chunk.
    Returns: (chunks_list, metadatas_list)
    """
    docs = df.astype(str).apply(lambda row: " | ".join([f"{c}: {row[c]}" for c in df.columns]), axis=1).tolist()
    if not docs:
        return [], []

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    small_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()[:2]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=False)

    chunks = []
    metadatas = []
    current_rows = []
    current_embs = []

    def _emit_chunk(rows_idx_list):
        # rows_idx_list = list of indices included in this chunk
        texts = [docs[i] for i in rows_idx_list]
        chunk_text = "\n".join(texts)
        md = {"row_indices": rows_idx_list}
        # numeric aggregates
        for c in numeric_cols:
            vals = []
            for i in rows_idx_list:
                try:
                    v = df.iloc[i][c]
                    if not pd.isna(v):
                        vals.append(float(v))
                except Exception:
                    pass
            md[f"min_{c}"] = None if not vals else float(np.min(vals))
            md[f"max_{c}"] = None if not vals else float(np.max(vals))
            md[f"avg_{c}"] = None if not vals else float(np.mean(vals))
        # small categorical aggregate (unique)
        for c in small_cat:
            uniques = []
            for i in rows_idx_list:
                try:
                    v = df.iloc[i][c]
                    if pd.isna(v):
                        continue
                    uniques.append(str(v))
                except Exception:
                    continue
            md[f"unique_{c}"] = None if not uniques else ", ".join(sorted(set(uniques)))
        return chunk_text, md

    for idx, emb in enumerate(embeddings):
        if not current_rows:
            current_rows = [idx]
            current_embs = [emb]
            continue
        avg_emb = np.mean(current_embs, axis=0).reshape(1, -1)
        sim = cosine_similarity(avg_emb, emb.reshape(1, -1))[0][0]
        if sim >= threshold:
            current_rows.append(idx)
            current_embs.append(emb)
        else:
            chunk_text, md = _emit_chunk(current_rows)
            chunks.append(chunk_text)
            metadatas.append(md)
            current_rows = [idx]
            current_embs = [emb]

    if current_rows:
        chunk_text, md = _emit_chunk(current_rows)
        chunks.append(chunk_text)
        metadatas.append(md)

    return chunks, metadatas

# ----------------------------
# Space usage comparison
# ----------------------------
def compare_space_usage_all(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50):
    docs = df.astype(str).apply(lambda row: ", ".join([f"{c}: {row[c]}" for c in df.columns]), axis=1).tolist()
    rec_text = "\n".join(docs)
    rec_total = len(rec_text)

    sem_rows = semantic_compression(df)
    sem_text = "\n".join(sem_rows)
    sem_total = len(sem_text)

    fixed_text = "\n".join(df.astype(str).apply(lambda row: " | ".join([f"{c}: {row[c]}" for c in df.columns]), axis=1).tolist())
    fixed_total = len(fixed_text)

    return {
        "recursive_total_chars": rec_total,
        "recursive_avg_per_row": rec_total / max(1, len(df)),
        "semantic_total_chars": sem_total,
        "semantic_avg_per_row": sem_total / max(1, len(df)),
        "fixed_total_chars": fixed_total,
        "fixed_avg_per_row": fixed_total / max(1, len(df)),
        "rows": len(df)
    }

# ----------------------------
# Embedding + Chroma helpers
# ----------------------------
def _sanitize_metadata_for_chroma(m):
    out = {}
    for k, v in (m or {}).items():
        if v is None:
            out[k] = None
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple, set)):
            # convert lists to comma-separated string
            try:
                out[k] = ", ".join(map(str, v))
            except Exception:
                out[k] = str(v)
        else:
            out[k] = str(v)
    return out

def embed_and_store(chunks: typing.List[str],
                    embeddings: typing.Optional[typing.List[typing.List[float]]] = None,
                    model_name: str = "all-MiniLM-L6-v2",
                    chroma_path: str = "chromadb_store",
                    collection_name: str = "default",
                    metadatas: typing.Optional[typing.List[dict]] = None):
    """
    If embeddings None -> compute via model. Then store in Chroma.
    metadatas: optional list matching chunks length (dictionaries)
    Returns (collection, model_instance)
    """
    model = None
    if embeddings is None:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True)

    emb_lists = [list(map(float, e)) for e in embeddings]

    client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(collection_name)

    # Clear existing
    try:
        existing = collection.get()
        if "ids" in existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    ids = [str(i) for i in range(len(chunks))]

    if metadatas is not None and len(metadatas) == len(chunks):
        sanitized = [_sanitize_metadata_for_chroma(m) for m in metadatas]
        collection.add(ids=ids, documents=chunks, embeddings=emb_lists, metadatas=sanitized)
    else:
        collection.add(ids=ids, documents=chunks, embeddings=emb_lists)

    return collection, (model if model is not None else SentenceTransformer(model_name))

def search_query(collection, model, query: str, k: int = 5):
    q_emb = model.encode([query])
    res = collection.query(query_embeddings=q_emb, n_results=k, include=["documents", "metadatas", "distances"])
    return res

# ----------------------------
# Build row-level metadata (helper)
# ----------------------------
def build_row_metadatas(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    small_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()[:2]
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append(_row_metadata(row, numeric_cols, small_cat))
    return metadatas

