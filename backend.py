# backend.py
import pandas as pd
import io, csv, typing, re
from sentence_transformers import SentenceTransformer
import chromadb

# optional: encoding detection
try:
    import chardet
except ImportError:
    chardet = None


# ----------------------------
# CSV Loading
# ----------------------------
def _detect_sep(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text[:4096])
        return dialect.delimiter
    except Exception:
        return ","


def load_csv(file_or_path: typing.Union[str, io.BytesIO]):
    """Load CSV from path or streamlit uploaded file"""
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
    return df.sample(n=min(len(df), n), random_state=42).reset_index(drop=True)


def change_dtype(df: pd.DataFrame, col: str, dtype: str):
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
# Preprocessing
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


def handle_missing(df: pd.DataFrame, strategy: str, fill_value: str = "unknown") -> pd.DataFrame:
    if strategy == "drop":
        return df.dropna().reset_index(drop=True)
    elif strategy == "fill":
        return df.fillna(fill_value)
    else:
        return df


# ----------------------------
# Chunking (Recursive)
# ----------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_chunk(df: pd.DataFrame, chunk_size: int = 200, overlap: int = 20):
    """Flatten rows to text and apply recursive chunking"""
    docs = df.astype(str).apply(lambda row: ", ".join([f"{c}: {row[c]}" for c in df.columns]), axis=1).tolist()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text("\n".join(docs))
    return chunks

# Semantic Compression
# ----------------------------
def semantic_compression(df: pd.DataFrame):
    """
    Create a compressed semantic representation of each row.
    Example: 'Shoes sold in Germany on 2023-01-01, quantity 3, priced at $306.04.'
    """
    compressed_rows = []
    for _, row in df.iterrows():
        try:
            description = str(row.get("Description", "item"))
            country = str(row.get("Country", "Unknown"))
            date = str(row.get("InvoiceDate", ""))
            qty = str(row.get("Quantity", ""))
            price = str(row.get("UnitPrice", ""))
            compressed = f"{description} sold in {country} on {date}, quantity {qty}, priced at ${price}."
        except Exception:
            compressed = "Transaction record"
        compressed_rows.append(compressed)
    return compressed_rows


def semantic_recursive_chunk(df: pd.DataFrame, chunk_size: int = 200, overlap: int = 20):
    """
    Apply semantic compression first, then recursive chunking.
    """
    compressed = semantic_compression(df)
    text = "\n".join(compressed)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    return chunks

# ----------------------------
# Space Comparison
# ----------------------------
def compare_space_usage(df, chunk_size=200, overlap=20):
    """
    Compare total character size of Recursive vs Semantic Compression + Recursive chunking.
    Returns dict with stats.
    """
    # 1. Standard recursive
    docs = df.astype(str).apply(
        lambda row: ", ".join([f"{c}: {row[c]}" for c in df.columns]), axis=1
    ).tolist()
    rec_text = "\n".join(docs)
    rec_total_chars = len(rec_text)

    # 2. Semantic compression recursive
    compressed = semantic_compression(df)
    sem_text = "\n".join(compressed)
    sem_total_chars = len(sem_text)

    return {
        "recursive_total_chars": rec_total_chars,
        "recursive_avg_per_row": rec_total_chars / len(df),
        "semantic_total_chars": sem_total_chars,
        "semantic_avg_per_row": sem_total_chars / len(df),
        "rows": len(df)
    }



# ----------------------------
# Embeddings + ChromaDB
# ----------------------------
def embed_and_store(chunks, model_name="all-MiniLM-L6-v2", chroma_path="chroma_store", collection_name="default"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(collection_name)

    # clear old data
    if collection.count() > 0:
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)

    ids = [str(i) for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)

    return collection, model


def search_query(collection, model, query: str, k: int = 5):
    q_emb = model.encode([query])
    res = collection.query(query_embeddings=q_emb, n_results=k)
    return res
