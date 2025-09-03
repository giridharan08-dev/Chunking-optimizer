import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="Retail Search App", page_icon="🛍️", layout="wide")
st.title("🛒 Retail Search Engine")
st.markdown("Search through your retail dataset with semantic search + filters + evaluation")

# ----------------------------
# Connect to ChromaDB
# ----------------------------
client = chromadb.PersistentClient(path="chromadb_store")
collection = client.get_or_create_collection("retail_chunks")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("🔎 Filters")
country_filter = st.sidebar.selectbox("Country", ["All", "India", "USA", "UK", "Germany", "Canada"])
price_range = st.sidebar.slider("Unit Price Range", 0, 500, (0, 500))
k_value = st.sidebar.slider("Recall@k (choose k)", 1, 20, 5)

# ----------------------------
# Search box
# ----------------------------
query = st.text_input("Search your dataset (e.g. 'cheap laptop'):")

if query:
    query_emb = model.encode([query])
    results = collection.query(
        query_embeddings=query_emb,
        n_results=k_value
    )
    docs = results["documents"][0]
    ids = results["ids"][0]

    if docs:
        st.subheader("📊 Search Results")

        filtered_docs = []
        for doc, _id in zip(docs, ids):
            fields = dict(item.split(": ", 1) for item in doc.split(", "))
            price = float(fields.get("UnitPrice", 0))
            country = fields.get("Country", "")
            
            # Apply filters
            if country_filter != "All" and country != country_filter:
                continue
            if not (price_range[0] <= price <= price_range[1]):
                continue

            filtered_docs.append(fields)

            # Render card
            st.markdown(f"""
            <div style="padding:12px; margin:10px; border-radius:12px; background-color:#ffffff; color:#000000; box-shadow: 0px 3px 8px rgba(0,0,0,0.15);">
                <b>🖥️ {fields.get('Description', 'N/A')}</b><br>
                🏷️ Price: {fields.get('UnitPrice', 'N/A')}  
                📦 Quantity: {fields.get('Quantity', 'N/A')}  
                🌍 Country: {fields.get('Country', 'N/A')}  
                🧾 Invoice: {fields.get('InvoiceNo', 'N/A')}  
                👤 Customer: {fields.get('CustomerID', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

        # ----------------------------
        # Evaluation: Recall@k
        # ----------------------------
        df_all = pd.read_csv("synthetic_retail_data.csv")
        query_lower = query.lower()
        ground_truth_ids = [
            str(i) for i, desc in enumerate(df_all["Description"])
            if query_lower.split()[0] in str(desc).lower()
        ]
        relevant_retrieved = set(ids) & set(ground_truth_ids)
        recall = len(relevant_retrieved) / len(ground_truth_ids) if ground_truth_ids else 0.0

        st.info(f"📈 Recall@{k_value}: **{recall:.2f}** "
                f"(Relevant retrieved: {len(relevant_retrieved)} / {len(ground_truth_ids)})")

        # ----------------------------
        # Analytics visualization
        # ----------------------------
        if filtered_docs:
            df = pd.DataFrame(filtered_docs)

            st.subheader("📊 Analytics")

            col1, col2, col3 = st.columns(3)

            # 1. Average Price
            avg_price = df["UnitPrice"].astype(float).mean()
            col1.metric("Average Price", f"${avg_price:.2f}")

            # 2. Total Sales Value
            df["SalesValue"] = df["UnitPrice"].astype(float) * df["Quantity"].astype(int)
            total_sales = df["SalesValue"].sum()
            col2.metric("Total Sales Value", f"${total_sales:.2f}")

            # 3. Top Product
            top_product = df["Description"].value_counts().idxmax()
            top_count = df["Description"].value_counts().max()
            col3.metric("Top Product", f"{top_product} ({top_count})")

            # ----------------------------
            # Country Distribution
            # ----------------------------
            st.subheader("🌍 Results by Country")
            st.bar_chart(df["Country"].value_counts())

            # ----------------------------
            # Top 5 Products
            # ----------------------------
            st.subheader("🏆 Top 5 Products")
            st.bar_chart(df["Description"].value_counts().head(5))
        else:
            st.warning("⚠️ No results after applying filters.")

    else:
        st.warning("⚠️ No results found. Try another query.")


