from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv  
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview") 
def llm_response(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content 
