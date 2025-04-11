import requests

print("Sending test prompt...")

try:
    resp = requests.post(
        "http://localhost:8000/generate",
        json={"prompt": "Write a template email to ask about job opportunity."},
        timeout=120
    )
    print("✅ Status Code:", resp.status_code)
    print("✅ Response:", resp.json())
except requests.exceptions.Timeout:
    print("⏰ Request timed out")
except Exception as e:
    print("❌ Failed:", e)
