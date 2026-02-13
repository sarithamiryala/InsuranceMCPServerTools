from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("google_api_key")
print(api_key)
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing! Set it in .env or environment variables.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    credentials=api_key  # Pass the key explicitly
)

def llm_response(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content
