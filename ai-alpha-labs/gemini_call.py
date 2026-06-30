from google import genai

PROJECT_ID = "project-7a27d46b-8896-4468-bba"
LOCATION = "us-central1"

# Use 'gemini-2.5-flash' which is a current stable model
client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location=LOCATION
)

try:
    print(f"Connecting to Vertex AI in {LOCATION}...")
    
    # Try gemini-2.5-flash as the primary choice
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents="Explain the current state of AI agents in one sentence."
    )

    print("\nGemini Response:")
    print(response.text)

except Exception as e:
    print(f"\nAn error occurred: {e}")
