
import streamlit as st
import pandas as pd
import pyarrow.csv as pacsv
import pyarrow as pa
from sentence_transformers import SentenceTransformer
import chromadb
from streamlit_lottie import st_lottie
import requests

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Retail Search App", page_icon="🛍️", layout="wide")
st.title("🛒 Retail Search Engine")
st.markdown("Search your retail dataset with semantic search + filters + evaluation")

# ----------------------------
# Load Lottie Animation
# ----------------------------
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Girl walking animation
girl_walk = load_lottie_url("https://lottie.host/42fd9639-1d2e-4de1-91db-d9e2b289ecfa/Q8uwbNfM7R.json")


# ----------------------------
# Paths
# ----------------------------
DATA_CSV = "synthetic_retail_data.csv"
CHROMA_PATH = "chromadb_store"
COLLECTION_NAME = "retail_chunks_arrow"

# ----------------------------
# Step 1: Load CSV with PyArrow
# ----------------------------
read_options = pacsv.ReadOptions(autogenerate_column_names=False)
parse_options = pacsv.ParseOptions(delimiter=",")
convert_options = pacsv.ConvertOptions()
table: pa.Table = pacsv.read_csv(DATA_CSV, read_options=read_options,
                                 parse_options=parse_options,
                                 convert_options=convert_options)

all_cols = table.schema.names
preferred_categorical = ["InvoiceNo", "StockCode", "Description", "InvoiceDate", "Country"]
preferred_numeric = ["Quantity", "UnitPrice", "CustomerID"]
categorical_cols = [c for c in preferred_categorical if c in all_cols]
numeric_cols = [c for c in preferred_numeric if c in all_cols]

# ----------------------------
# Step 2: Build Chunks + Metadata
# ----------------------------
def to_python_value(arr, i):
    return arr[i].as_py()

def build_row_dict(batch, row_idx_in_batch):
    return {col_name: to_python_value(batch.column(col_name), row_idx_in_batch)
            for col_name in batch.schema.names}

def make_chunk_text(row_dict, cat_cols):
    return ", ".join([f"{c}: {row_dict.get(c, '')}" for c in cat_cols])

chunks, metadatas, id_list = [], [], []
row_counter = 0

for batch in table.to_batches(max_chunksize=2048):
    batch = pa.Table.from_batches([batch])
    for i in range(batch.num_rows):
        row = build_row_dict(batch, i)
        text = make_chunk_text(row, categorical_cols)

        meta = {k: row[k] for k in numeric_cols}
        for keep in ["Country", "Description", "InvoiceDate", "InvoiceNo", "StockCode"]:
            if keep in row:
                val = row[keep]
                if hasattr(val, "isoformat"):
                    val = str(val)
                elif val is None:
                    val = None
                else:
                    val = str(val) if not isinstance(val, (int, float, bool)) else val
                meta[keep] = val

        chunks.append(text)
        metadatas.append(meta)
        id_list.append(str(row_counter))
        row_counter += 1

# ----------------------------
# Step 3: Embeddings + ChromaDB
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, batch_size=128, show_progress_bar=False)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

if collection.count() == 0:  # only populate if empty
    BATCH = 500
    for start in range(0, len(chunks), BATCH):
        end = start + BATCH
        collection.add(
            ids=id_list[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end]
        )

# ----------------------------
# Search function
# ----------------------------
def search(query_text, k=5):
    q_emb = model.encode([query_text])
    return collection.query(query_embeddings=q_emb, n_results=k,
                            include=["documents", "metadatas", "distances"])

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("🔎 Filters")
country_filter = st.sidebar.selectbox("Country", ["All", "India", "USA", "UK", "Germany", "Canada"])
price_range = st.sidebar.slider("Unit Price Range", 0, 500, (0, 500))
k_value = st.sidebar.slider("Recall@k", 1, 20, 5)

# ----------------------------
# Search box
# ----------------------------
query = st.text_input("Search your dataset (e.g. 'cheap laptop'):")

if query:
    # Show animation while searching
    with st.spinner("Searching..."):
        if girl_walk:
            anim_placeholder = st.empty()
            with anim_placeholder:
                st_lottie(girl_walk, height=250, key="girl")

        # (simulate a small delay so animation is visible)
        import time
        time.sleep(1.5)

        results = search(query, k=k_value)

    # Remove animation after search completes
    if girl_walk:
        anim_placeholder.empty()


    docs, metas = results["documents"][0], results["metadatas"][0]

    if docs:
        st.subheader("📊 Search Results")
        filtered_docs = []

        for doc, meta in zip(docs, metas):
            price = float(meta.get("UnitPrice", 0))
            country = meta.get("Country", "")

            if country_filter != "All" and country != country_filter:
                continue
            if not (price_range[0] <= price <= price_range[1]):
                continue

            filtered_docs.append(meta)

            st.markdown(f"""
            <div style="padding:12px; margin:10px; border-radius:12px; background-color:#f9f9f9; color:#000; box-shadow:0px 3px 8px rgba(0,0,0,0.15);">
                <b>🖥️ {meta.get('Description', 'N/A')}</b><br>
                🏷️ Price: {meta.get('UnitPrice', 'N/A')}  
                📦 Quantity: {meta.get('Quantity', 'N/A')}  
                🌍 Country: {meta.get('Country', 'N/A')}  
                🧾 Invoice: {meta.get('InvoiceNo', 'N/A')}  
                👤 Customer: {meta.get('CustomerID', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

        # Recall@K
        df_all = table.to_pandas()
        gt_ids = [str(i) for i, val in enumerate(df_all["Description"].astype(str))
                  if query.lower() in val.lower()]
        retrieved_ids = results["ids"][0] if "ids" in results else []
        relevant_retrieved = set(retrieved_ids) & set(gt_ids)
        recall = len(relevant_retrieved) / len(gt_ids) if gt_ids else 0.0

        st.info(f"📈 Recall@{k_value}: **{recall:.2f}** "
                f"(Relevant retrieved: {len(relevant_retrieved)} / {len(gt_ids)})")

        if filtered_docs:
            df = pd.DataFrame(filtered_docs)
            st.subheader("📊 Analytics")
            col1, col2, col3 = st.columns(3)

            avg_price = df["UnitPrice"].astype(float).mean()
            col1.metric("Average Price", f"${avg_price:.2f}")

            df["SalesValue"] = df["UnitPrice"].astype(float) * df["Quantity"].astype(int)
            total_sales = df["SalesValue"].sum()
            col2.metric("Total Sales Value", f"${total_sales:.2f}")

            top_product = df["Description"].value_counts().idxmax()
            top_count = df["Description"].value_counts().max()
            col3.metric("Top Product", f"{top_product} ({top_count})")

            st.subheader("🌍 Results by Country")
            st.bar_chart(df["Country"].value_counts())

            st.subheader("🏆 Top 5 Products")
            st.bar_chart(df["Description"].value_counts().head(5))

    else:
        st.warning("⚠️ No results found. Try another query.")
