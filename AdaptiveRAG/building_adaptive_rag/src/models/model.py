from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv(dotenv_path='D:/GenAI-Practice/AgenticAI-Projects/.env')

### chat model
llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",                     
    temperature=0
    )

### embedding model
embed_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
    )
