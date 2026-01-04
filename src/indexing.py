
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --- UPDATED IMPORTS FOR 2026 VERSION ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # <--- Changed this line
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# ----------------------------------------
# --- 1. DATA LOADING & SAMPLING ---
def load_and_sample_data(file_path, sample_size=10000):
    print("--- Step 1: Loading and Sampling Data ---")
    # Load enough rows to find our target products
    df = pd.read_csv(file_path, nrows=2000000, low_memory=False)
    
    target_products = [
        'Credit card or prepaid card',
        'Checking or savings account',
        'Personal loan',
        'Money transfer, virtual currency, or money service',
        'Payday loan, title loan, or personal loan'
    ]
    
    # Filter for products and ensure there is a story to read
    df_filtered = df[df['Product'].isin(target_products)].copy()
    df_filtered = df_filtered.dropna(subset=['Consumer complaint narrative'])
    
    # Stratified Sampling to keep the mix fair
    df_sample, _ = train_test_split(
        df_filtered, 
        train_size=min(sample_size, len(df_filtered)), 
        stratify=df_filtered['Product'], 
        random_state=42
    )
    return df_sample

# --- 2. CHUNKING ---
def create_chunks(df):
    print("--- Step 2: Chunking Text ---")
    # Project requirement: 500 chars size, 50 chars overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    documents = []
    for _, row in df.iterrows():
        content = row['Consumer complaint narrative']
        metadata = {
            "product": row['Product'],
            "complaint_id": row['Complaint ID']
        }
        # Split each story into small pieces
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            # Wrap in a LangChain Document object
            documents.append(Document(page_content=chunk, metadata=metadata))
            
    print(f"Created {len(documents)} chunks.")
    return documents

# --- 3. EMBEDDING & INDEXING ---
def create_vector_store(documents, store_path="vector_store/faiss_index"):
    print("--- Step 3: Creating Vector Store (Embeddings) ---")
    # Load the specific model requested: all-MiniLM-L6-v2
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the FAISS index (this turns text into numbers!)
    vector_db = FAISS.from_documents(documents, embeddings)
    
    # Save it locally so we don't have to do this again
    vector_db.save_local(store_path)
    print(f"Vector store saved to {store_path}")
    return vector_db

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    RAW_PATH = 'data/raw/complaints.csv'
    
    # Run the full pipeline
    data = load_and_sample_data(RAW_PATH)
    docs = create_chunks(data)
    create_vector_store(docs)
    
    print("\n✅ Task 2 Complete! Your chatbot now has a brain.")