import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

class LLMModel:
    def __init__(self, model_name = "llama3-70b-8192"):
        if not model_name:
            raise ValueError("Model is not defined")

        self.model = ChatGroq(
            temperature = 0,
            groq_api_key = os.getenv("groq_api_key"),
            model_name = model_name,
            )
        
    def get_model(self):
        return self.model
    
if __name__ == '__main__':
    llm_instance = LLMModel()
    llm_model = llm_instance.get_model()

    response = llm_model.invoke("Hi")

    print(response.content)
