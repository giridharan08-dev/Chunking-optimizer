# backend.py
import io, csv, typing, re, math
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Text splitter import (langchain_text_splitters). Use whichever you have installed.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # fallback name for older langchain packages
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------
# CSV Loading / Utilities
# ----------------------------
try:
    import chardet
except Exception:
    chardet = None

def _detect_sep(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text[:4096])
        return dialect.delimiter
    except Exception:
        return ","

def load_csv(file_or_path: typing.Union[str, io.BytesIO, io.StringIO]):
    """Load CSV from path or stream file (Streamlit uploaded file)"""
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
# Chunking functions
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
    # flatten each row into a comparable string and chunk across full text
    docs = df.astype(str).apply(lambda row: " | ".join([f"{c}: {row[c]}" for c in df.columns]), axis=1).tolist()
    text = "\n".join(docs)
    return fixed_size_chunking_from_text(text, chunk_size=chunk_size, overlap=overlap)

def recursive_chunk(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50):
    docs = df.astype(str).apply(lambda row: ", ".join([f"{c}: {row[c]}" for c in df.columns]), axis=1).tolist()
    text = "\n".join(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    return chunks

# Semantic compression (row → short sentence)
def semantic_compression(df: pd.DataFrame):
    compressed_rows = []
    for _, row in df.iterrows():
        try:
            # try to cover common columns; if not present, compress generically
            if "Description" in row.index and "Country" in row.index:
                description = str(row.get("Description", "item"))
                country = str(row.get("Country", "Unknown"))
                date = str(row.get("InvoiceDate", ""))
                qty = str(row.get("Quantity", ""))
                price = str(row.get("UnitPrice", ""))
                compressed = f"{description} sold in {country} on {date}, qty {qty}, price ${price}"
            else:
                # generic numeric-heavy summary
                pieces = []
                for c, v in row.items():
                    if pd.api.types.is_numeric_dtype(type(v)) or isinstance(v, (int, float)):
                        try:
                            pieces.append(f"{c}:{float(v):.0f}")
                        except Exception:
                            pieces.append(f"{c}:{v}")
                    else:
                        pieces.append(f"{c}:{str(v)[:30]}")
                compressed = "; ".join(pieces)
        except Exception:
            compressed = "record"
        compressed_rows.append(compressed)
    return compressed_rows

def semantic_recursive_chunk(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50):
    compressed = semantic_compression(df)
    text = "\n".join(compressed)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    return chunks

# ----------------------------
# Space usage comparison
# ----------------------------
def compare_space_usage_all(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50):
    # recursive
    docs = df.astype(str).apply(lambda row: ", ".join([f"{c}: {row[c]}" for c in df.columns]), axis=1).tolist()
    rec_text = "\n".join(docs)
    rec_total = len(rec_text)

    # semantic
    sem_rows = semantic_compression(df)
    sem_text = "\n".join(sem_rows)
    sem_total = len(sem_text)

    # fixed (flatten whole text then chunk; total chars equals flattened)
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
def embed_and_store(chunks: typing.List[str],
                    embeddings: typing.Optional[typing.List[typing.List[float]]] = None,
                    model_name: str = "all-MiniLM-L6-v2",
                    chroma_path: str = "chromadb_store",
                    collection_name: str = "default",
                    metadatas: typing.Optional[typing.List[dict]] = None):
    """
    If embeddings None -> compute via model. Then store in Chroma.
    metadatas: optional list matching chunks length (dictionaries)
    """
    model = None
    if embeddings is None:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True)

    # ensure embeddings are lists of float
    emb_lists = [list(map(float, e)) for e in embeddings]

    client = chromadb.PersistentClient(path=chroma_path)
    # create or get collection
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(collection_name)

    # Clear previous content (safe)
    try:
        existing = collection.get()
        if "ids" in existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    ids = [str(i) for i in range(len(chunks))]

    # add with or without metadata
    if metadatas is not None:
        # sanitize metadata values (must be primitives)
        sanitized = []
        for m in metadatas:
            dd = {}
            for k, v in (m or {}).items():
                if v is None:
                    dd[k] = None
                elif isinstance(v, (str, int, float, bool)):
                    dd[k] = v
                else:
                    dd[k] = str(v)
            sanitized.append(dd)
        collection.add(ids=ids, documents=chunks, embeddings=emb_lists, metadatas=sanitized)
    else:
        collection.add(ids=ids, documents=chunks, embeddings=emb_lists)

    return collection, (model if model is not None else SentenceTransformer(model_name))

def search_query(collection, model, query: str, k: int = 5):
    q_emb = model.encode([query])
    res = collection.query(query_embeddings=q_emb, n_results=k, include=["documents", "metadatas", "distances"])
    return res

# ----------------------------
# Simple helper to build metadata list from dataframe (numeric columns preserved)
# ----------------------------
def build_row_metadatas(df: pd.DataFrame):
    metadatas = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # also include a set of small useful categorical for filtering
    small_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()[:2]
    for _, row in df.iterrows():
        md = {}
        for c in numeric_cols:
            try:
                val = row[c]
                # convert numpy types
                if pd.isna(val):
                    md[c] = None
                else:
                    md[c] = float(val) if not isinstance(val, (str,)) else float(str(val))
            except Exception:
                md[c] = None
        for c in small_cat:
            md[c] = (None if pd.isna(row.get(c)) else str(row.get(c)))
        metadatas.append(md)
    return metadatas