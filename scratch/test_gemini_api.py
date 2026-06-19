import os
import sys
from dotenv import load_dotenv
from google import genai

# Load the environment variables from .env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
api_key = os.environ.get("GEMINI_AUTH_API_KEY")

if not api_key:
    print("API key not found in .env!")
    sys.exit(1)

print(f"Testing Gemini API with key starting with: {api_key[:5]}...")

try:
    # Initialize the client
    client = genai.Client(api_key=api_key)
    
    print("Sending request to gemini-2.5-flash...")
    # Using gemini-2.5-flash as 3.5-flash might not be generally available depending on the SDK version, 
    # but I'll follow your prompt's instruction exactly:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explain how AI works in a few words"
    )
    print("\n--- Response Received ---")
    print(response.text)
    print("-------------------------")
    print("SUCCESS: The API key is working perfectly!")
    
except Exception as e:
    print("\n--- Request Failed ---")
    print(str(e))
