import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# override=True forces reading from .env even if the var is already set in shell
load_dotenv(override=True)

key = os.environ.get("GOOGLE_API_KEY", "NOT SET")
print(f"Using API key starting with: {key[:12]}...")

models_to_test = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

for model in models_to_test:
    try:
        llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        res = llm.invoke("Say hi")
        print(f"SUCCESS {model}: {res.content[:60]}")
    except Exception as e:
        err = str(e)[:200]
        print(f"ERROR {model}: {err}")
