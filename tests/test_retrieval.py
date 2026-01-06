import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def test_faiss_index():
    print("🔍 Testing FAISS Index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if index exists
    if not os.path.exists("vector_store/faiss_index"):
        print("❌ Error: Vector store not found!")
        return

    # Load and search
    vector_db = FAISS.load_local("vector_store/faiss_index", embeddings, allow_dangerous_deserialization=True)
    results = vector_db.similarity_search("credit card", k=1)
    
    if len(results) > 0:
        print(f"✅ Success! Found {len(results)} relevant complaint(s).")
    else:
        print("❌ Search returned no results.")

if __name__ == "__main__":
    test_faiss_index()