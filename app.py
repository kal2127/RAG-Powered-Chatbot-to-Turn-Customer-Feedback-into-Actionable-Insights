import streamlit as st
import os
import time
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Using the 2026 'classic' package for chains
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="CrediTrust AI Analyst", layout="wide", page_icon="🛡️")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #004a99; color: white; font-weight: bold; }
    .stTextInput>div>div>input { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALIZATION & API SETUP ---
# Remember to replace this with your real sk-... key!
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

@st.cache_resource
def load_rag_system():
    """Loads the FAISS brain and the LLM once to save time."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Loading the 10,000 samples you built in Task 2
    vector_db = FAISS.load_local(
        "vector_store/faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # We use streaming=True for the 'typewriter' effect
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    
    system_prompt = (
        "You are an expert CrediTrust Analyst. Use the following complaints to answer the question. "
        "If the answer isn't in the data, say you don't know. Keep it under 4 sentences."
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Connecting the pieces
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 3}), document_chain)

# Initialize the chain
rag_chain = load_rag_system()

# --- 3. SIDEBAR (RESET BUTTON) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("Admin Controls")
    
    # This button wipes the screen clean
    if st.button("🔄 Reset Conversation"):
        st.rerun()
    
    st.write("---")
    st.info("💡 **Tip:** Ask about 'Late fees' or 'ATM withdrawals' to see how the AI summarizes the data.")

# --- 4. MAIN USER INTERFACE ---
st.title("🛡️ CrediTrust Customer Insights Dashboard")
st.markdown("Use this tool to analyze patterns in customer feedback instantly.")

# Layout with one text box and one button
col1, col2 = st.columns([4, 1])

with col1:
    user_query = st.text_input(
        "What would you like to know about the complaints?", 
        placeholder="e.g., What are the most common problems with money transfers?",
        label_visibility="collapsed"
    )

with col2:
    submit_button = st.button("🔍 Ask AI")

# --- 5. GENERATING THE RESPONSE ---
if submit_button and user_query:
    # 1. Show User Question
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Show Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing 10,000 complaints..."):
            # Get data from RAG
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            
            # Streaming effect (Typewriter)
            placeholder = st.empty()
            typed_text = ""
            for word in answer.split(" "):
                typed_text += word + " "
                placeholder.markdown(typed_text + "▌")
                time.sleep(0.04) # Speed of typing
            placeholder.markdown(answer)

            # 3. SHOW SOURCES (Evidence)
            st.markdown("---")
            st.subheader("📂 Evidence from Database")
            for i, doc in enumerate(response["context"]):
                with st.expander(f"Complaint Excerpt {i+1}"):
                    st.write(doc.page_content)