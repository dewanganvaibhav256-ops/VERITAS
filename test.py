
import os
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

key = os.getenv("GEMINI_API_KEY")
if not key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=key)

try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Reply with only: API TEST SUCCESS",
    )
    print(response.text)
except Exception as exc:
    print("Error type:", type(exc).__name__)
    print("Error details:", str(exc)[:1200])

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Reply with OK"
)
print(response.text)