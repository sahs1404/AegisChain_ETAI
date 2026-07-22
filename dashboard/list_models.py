"""
Run this once to see which models your Gemini API key actually has
access to. Model availability varies by account/region/age, which is
why guessing names from docs/blogs keeps 404ing.
"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Models available to your key that support generateContent:\n")
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(" -", m.name)
