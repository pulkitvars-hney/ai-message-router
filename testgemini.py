from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

key = os.environ.get('GEMINI_API_KEY')
print("Key loaded:", "YES" if key else "NO")

client = genai.Client(api_key=key)
r = client.models.generate_content(
    model='gemini-3.8-flash',
    contents='say hello'
)
print(r.text)