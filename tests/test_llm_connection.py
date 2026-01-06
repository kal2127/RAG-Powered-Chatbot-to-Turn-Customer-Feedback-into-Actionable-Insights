import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def test_openai_connection():
    print("🌐 Testing OpenAI Connection...")
    
    # 1. Load the hidden API key from .env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or "sk-" not in api_key:
        print("❌ Error: OpenAI API Key not found in .env file!")
        return

    try:
        # 2. Try to send a simple message
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke("Say 'Connection Successful!'")
        
        print(f"🤖 AI Response: {response.content}")
        print("✅ Success! Your API key and internet are working perfectly.")
        
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    test_openai_connection()