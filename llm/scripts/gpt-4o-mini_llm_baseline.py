import os
import json
import requests
from time import sleep

# === CONFIGURATION ===
API_KEY = "" 
MODEL = "openai/gpt-4o-mini"
RAW_DATA_DIR = "raw_data"
OUTPUT_FILE = "classified_results.json"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://yourdomain.com",  # optional
    "X-Title": "Rudra Safety Analyzer",
}

# === STATIC PROMPT TEMPLATE (implicitly cached by OpenAI models) ===
PROMPT_TEMPLATE = """
You are analyzing static analysis results produced by **Rudra**, a Rust safety analyzer.
Each result is a JSON object with fields such as `level`, `analyzer`, `description`, and `code_snippet`.

Your task:
- Read the `code_snippet` carefully and cross-check with the `analyzer`, `description`, and `level`.
- Decide whether this report indicates a **REAL ISSUE** (genuine unsafe bug) or a **FALSE POSITIVE** (not an issue).
- If the code appears safe and the warning seems likely harmless, you can choose "false_positive".
- Otherwise, return "real_issue".
- Reply with ONLY one word: "real_issue" or "false_positive".
"""

# === FUNCTION: Query Model ===
def classify_result(json_data):
    """Send prompt + dynamic JSON to GPT model."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROMPT_TEMPLATE},  # static part → cached automatically
            {"role": "user", "content": f"Here is one result in JSON:\n\n{json.dumps(json_data, indent=2)}"}  # dynamic part
        ],
        "max_tokens": 10,
        "temperature": 0.0,
        "usage": {"include": True},  # optional: to see cache savings
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=60
        )
        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            return "error"

        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip().lower()

        if reply in ["real_issue", "false_positive"]:
            return reply
        else:
            print(f"Unexpected model reply: {reply}")
            return "unknown"
    except Exception as e:
        print(f"Request failed: {e}")
        return "error"

# === MAIN LOOP ===
def main():
    results = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}

    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".json")]
    print(f"🧩 Found {len(files)} JSON files to process...")

    for file_name in files:
        if file_name in results:
            print(f"Skipping {file_name} (already processed)")
            continue

        file_path = os.path.join(RAW_DATA_DIR, file_name)
        with open(file_path, "r") as f:
            json_data = json.load(f)

        print(f"Processing {file_name} ...")
        verdict = classify_result(json_data)
        print(f"→ {file_name}: {verdict}")

        results[file_name] = verdict

        with open(OUTPUT_FILE, "w") as out:
            json.dump(results, out, indent=4)

        sleep(1)  # avoid rate limits

    print("\n All files processed!")
    print(f"Results written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
