import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- UPDATED FOR 2026 VERSION ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# --------------------------------
from langchain_core.prompts import ChatPromptTemplate

# 1. SET YOUR API KEY
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

def start_creditrust_bot():
    print("--- 🤖 Initializing CrediTrust Support Bot ---")
    
    # 2. LOAD THE TASK 2 BRAIN
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the index you built from your 10,000 samples
    vector_db = FAISS.load_local(
        "vector_store/faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 3. SETUP THE LIBRARIAN (The LLM)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 4. CREATE A CUSTOM PROMPT
    # This acts as the instructions for the AI
    system_prompt = (
        "You are an expert customer insight assistant for CrediTrust Financial. "
        "Use the provided customer complaints (context) to answer the question. "
        "If the answer isn't in the context, say you don't know. "
        "Keep the answer under 3 sentences."
        "\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 5. INTEGRATE THE RETRIEVAL PIPELINE
    # This chain handles how the documents are "stuffed" into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # This chain connects the retriever (FAISS) to the question_answer_chain
    rag_chain = create_retrieval_chain(vector_db.as_retriever(), question_answer_chain)

    # 6. ASK A TEST QUESTION
    query = "Do customers feel that customer service is helpful when reporting identity theft?"
    print(f"\nAsha's Question: {query}")
    
    # In the new version, we use 'input' as the key
    response = rag_chain.invoke({"input": query})
    
    print("\n--- Answer from the Data ---")
    print(response["answer"])

if __name__ == "__main__":
    start_creditrust_bot()