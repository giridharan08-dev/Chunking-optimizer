import streamlit as st
import pandas as pd
import re
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- Utility Functions ----------------

def preview_random(df, n=5):
    return df.sample(n) if len(df) > n else df

def to_lowercase(df):
    return df.applymap(lambda s: s.lower() if isinstance(s, str) else s)

def remove_html_tags(df, columns=None):
    html_pattern = re.compile(r'<.*?>')
    if columns:
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: re.sub(html_pattern, '', x))
    return df

# ---------------- Streamlit App ----------------

st.set_page_config(page_title="Chunking Optimizer", layout="wide")
st.title("📊 Chunking Optimizer")

if "stage" not in st.session_state:
    st.session_state.stage = "upload"
if "df" not in st.session_state:
    st.session_state.df = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []

# ---------------- Stage 1: File Upload ----------------

if st.session_state.stage == "upload":
    st.header("Step 1: Upload CSV")
    file = st.file_uploader("Upload your CSV file", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file)
            st.session_state.df = df
            st.success("✅ File uploaded successfully!")

            st.write("### Data Preview")
            st.dataframe(preview_random(df, 5))

            st.write("### Column Data Types")
            st.write(df.dtypes)

            if st.button("Proceed to Data Type Change"):
                st.session_state.stage = "datatype_change"

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# ---------------- Stage 1.5: Data Type Change ----------------

elif st.session_state.stage == "datatype_change":
    st.header("Step 1.5: Change Column Data Types")

    df = st.session_state.df
    st.write("Current Data Types:")
    st.write(df.dtypes)

    column_to_change = st.selectbox("Select column to change type", df.columns)
    new_type = st.selectbox("Select new data type", ["int", "float", "str", "datetime"])

    if st.button("Apply Data Type Change"):
        try:
            if new_type == "datetime":
                df[column_to_change] = pd.to_datetime(df[column_to_change])
            else:
                df[column_to_change] = df[column_to_change].astype(new_type)

            st.session_state.df = df
            st.success(f"Data type of {column_to_change} changed to {new_type}.")
            st.dataframe(df.head())
            st.session_state.stage = "layer1"
        except Exception as e:
            st.error(f"❌ Failed to convert data type: {e}")

# ---------------- Stage 2: Preprocessing Layer 1 ----------------

elif st.session_state.stage == "layer1":
    st.header("Step 2: Preprocessing - Layer 1")
    df = st.session_state.df

    lower = st.checkbox("Convert all text to lowercase")
    html_cols = [c for c in df.columns if df[c].astype(str).str.contains("<.*?>", regex=True).any()]
    remove_html = False
    if html_cols:
        remove_html = st.checkbox(f"Remove HTML tags in columns: {html_cols}")

    if st.button("Apply Layer 1"):
        if lower:
            df = to_lowercase(df)
        if remove_html and html_cols:
            df = remove_html_tags(df, columns=html_cols)

        st.session_state.df = df
        st.success("✅ Layer 1 applied.")
        st.dataframe(preview_random(df, 5))
        st.session_state.stage = "layer2"

# ---------------- Stage 3: Preprocessing Layer 2 ----------------

elif st.session_state.stage == "layer2":
    st.header("Step 3: Preprocessing - Layer 2")
    df = st.session_state.df

    if df.isnull().sum().sum() > 0:
        st.warning("⚠️ Missing values detected.")
        mv_action = st.radio("How to handle missing values?",
                             ["Do nothing", "Drop rows", "Fill with 'Unknown'"])
    else:
        mv_action = None
        st.info("✅ No missing values found.")

    if df.duplicated().sum() > 0:
        drop_dupes = st.checkbox("Remove duplicate rows?")
    else:
        drop_dupes = False
        st.info("✅ No duplicate rows found.")

    stemming = st.checkbox("Apply stemming (basic demo)")
    lemmatize = st.checkbox("Apply lemmatization (basic demo)")
    stopwords = st.checkbox("Remove stopwords (basic demo)")

    if st.button("Apply Layer 2"):
        if mv_action == "Drop rows":
            df = df.dropna()
        elif mv_action == "Fill with 'Unknown'":
            df = df.fillna("Unknown")
        if drop_dupes:
            df = df.drop_duplicates()

        st.session_state.df = df
        st.success("✅ Layer 2 applied.")
        st.dataframe(preview_random(df, 5))
        st.session_state.stage = "quality"

# ---------------- Stage 4: Quality Gate ----------------

elif st.session_state.stage == "quality":
    st.header("Step 4: Quality Gate")
    df = st.session_state.df

    st.info("Performing quality checks...")
    passes = True
    if df.isnull().sum().sum() > 0:
        passes = False

    if passes:
        st.success("✅ Quality check passed!")
        if st.button("Proceed to Chunking"):
            st.session_state.stage = "chunking"
    else:
        st.error("❌ Quality check failed. Please reconfigure preprocessing.")

# ---------------- Stage 5: Chunking ----------------

elif st.session_state.stage == "chunking":
    st.header("Step 5: Chunking Strategy")
    df = st.session_state.df

    strategy = st.radio("Choose a chunking strategy:",
                        ["Fixed-size", "Recursive", "Document-based"])

    if strategy == "Fixed-size":
        chunk_size = st.number_input("Chunk size (characters)", min_value=50, value=500)

    if strategy == "Recursive":
        chunk_size = st.number_input("Chunk size (characters)", min_value=50, value=300)
        overlap = st.number_input("Chunk overlap (characters)", min_value=0, value=50)

    if st.button("Apply Chunking"):
        chunks = []
        if strategy == "Fixed-size":
            text_data = df.to_string(index=False)
            chunks = [text_data[i:i + chunk_size] for i in range(0, len(text_data), chunk_size)]

        elif strategy == "Recursive":
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            text_data = df.to_string(index=False)
            chunks = splitter.split_text(text_data)

        elif strategy == "Document-based":
            chunks = df.apply(lambda row: row.to_dict(), axis=1).tolist()

        st.session_state.chunks = chunks
        st.success(f"✅ {strategy} chunking applied. Created {len(chunks)} chunks.")
        st.session_state.stage = "embedding"

# ---------------- Stage 6: Embedding ----------------

elif st.session_state.stage == "embedding":
    st.header("Step 6: Embedding")
    chunks = st.session_state.chunks

    model_choice = st.selectbox("Choose embedding model:",
                                ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "multi-qa-MiniLM-L6-v1"])

    if st.button("Generate Embeddings"):
        model = SentenceTransformer(model_choice)
        embeddings = model.encode([str(c) for c in chunks])
        st.session_state.embeddings = embeddings
        st.success("✅ Embeddings generated.")
        st.session_state.stage = "store"

# ---------------- Stage 7: Store in ChromaDB ----------------

elif st.session_state.stage == "store":
    st.header("Step 7: Store in ChromaDB?")
    client = chromadb.PersistentClient(path="chromadb_store")
    collection = client.get_or_create_collection("retail_chunks")

    if st.button("Yes, store in ChromaDB"):
        for i, (chunk, emb) in enumerate(zip(st.session_state.chunks, st.session_state.embeddings)):
            collection.add(ids=[str(i)], documents=[str(chunk)], embeddings=[emb.tolist()])
        st.success("✅ Data stored in ChromaDB.")
        st.session_state.stage = "retrieval"

    if st.button("Skip storing"):
        st.session_state.stage = "retrieval"

# ---------------- Stage 8: Retrieval ----------------

elif st.session_state.stage == "retrieval":
    st.header("Step 8: Retrieval")
    client = chromadb.PersistentClient(path="chromadb_store")
    collection = client.get_or_create_collection("retail_chunks")

    query = st.text_input("Enter your search query:")
    k = st.slider("How many results to show?", 1, 20, 5)

    if query:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = model.encode([query])
        results = collection.query(query_embeddings=q_emb, n_results=k)

        if results and results["documents"][0]:
            st.success("✅ Results found!")
            for doc in results["documents"][0]:
                st.write(doc)
        else:
            st.error("❌ No matching results found.")
