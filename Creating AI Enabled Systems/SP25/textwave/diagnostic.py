import os
import requests
import os



os.environ["MISTRAL_API_KEY"] = "GDTId8eQPtNGoVAhqkr5hel3mKqtoD1j"
# Print environment variable status




def query_mistral(prompt):
    """
    Sends the prompt to the Mistral API and returns the response.
    Includes detailed logging of status codes and full response.
    """
    import requests
    import os
    import traceback

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-medium",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        print("[DEBUG] Sending request to Mistral API...")
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
        print(f"[DEBUG] Status Code: {response.status_code}")
        print("[DEBUG] Response Text:", response.text[:500])

        # Dump full response to log file
        with open("mistral_raw_response.log", "a", encoding="utf-8") as logf:
            logf.write(f"STATUS {response.status_code}\n{response.text}\n\n")

        if response.status_code != 200:
            print(f"[ERROR] Non-200 status from Mistral: {response.status_code}")
            return f"[ERROR] Status {response.status_code}: {response.text}"

        try:
            parsed = response.json()
            print("[DEBUG] Parsed JSON keys:", list(parsed.keys()))
            return parsed["choices"][0]["message"]["content"]
        except Exception as json_err:
            print("[ERROR] Failed to parse JSON:")
            traceback.print_exc()
            return "[ERROR] Could not decode Mistral response."

    except Exception as e:
        print("[ERROR] Exception during API request:")
        traceback.print_exc()
        return "[ERROR] Mistral API call failed."




print("[INFO] Checking for API key...")
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    print("[ERROR] API key not found in environment.")
    exit(1)
print("[INFO] API key found.")

# Prepare headers and payload
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "mistral-medium",
    "messages": [{"role": "user", "content": "Hello, are you there?"}]
}

# Perform API request
print("[INFO] Sending request to Mistral...")
try:
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    print("[INFO] Response received.")
    print("Status:", response.status_code)
    print("Body (first 500 chars):", response.text[:500])
except Exception as e:
    print("[ERROR] API request failed:")
    import traceback
    traceback.print_exc()
