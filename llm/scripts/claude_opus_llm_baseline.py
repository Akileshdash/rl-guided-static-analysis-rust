import os
import json
import requests
from time import sleep

# === CONFIGURATION ===
API_KEY = ""
MODEL = "anthropic/claude-opus-4.1"
RAW_DATA_DIR = "raw_data"
OUTPUT_FILE = "classified_results_claude.json"
BATCH_SIZE = 10
MAX_RETRIES = 3
ERROR_LOG = "error_log.txt"
RAW_REPLY_DIR = "raw_replies"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://yourdomain.com",
    "X-Title": "Rudra Safety Analyzer (Claude)",
}

# Ensure raw reply directory exists
os.makedirs(RAW_REPLY_DIR, exist_ok=True)

# === STATIC PROMPT TEMPLATE ===
PROMPT_TEMPLATE = """
You are analyzing static analysis results produced by Rudra, a Rust safety analyzer.

Each result is a JSON object with fields such as `level`, `analyzer`, `description`, and `code_snippet`.

Your task:
- Read each result carefully and decide whether it represents a REAL ISSUE (unsafe bug) or a FALSE POSITIVE (harmless).
- If the code looks safe and the warning seems not dangerous, choose "false_positive".
- Otherwise, choose "real_issue".
- Respond strictly in this JSON format:

{
  "results": {
    "file1.json": "real_issue",
    "file2.json": "false_positive",
    ...
  }
}

No explanations or commentary.
"""

def classify_batch(batch_dict, batch_num):
    """Send one batched request to Claude Opus and save raw output."""
    user_content = "Here are multiple Rudra analysis results:\n\n"
    for name, data in batch_dict.items():
        user_content += f"### {name}\n{json.dumps(data, indent=2)}\n\n"
    user_content += "Now classify them as described above."

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_TEMPLATE,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_content}]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.0
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(f"API Error {response.status_code}: {response.text}")

    data = response.json()
    reply = data["choices"][0]["message"]["content"].strip()

    # Save the raw reply to file (always)
    raw_path = os.path.join(RAW_REPLY_DIR, f"batch_{batch_num:04d}.txt")
    with open(raw_path, "w") as raw_file:
        raw_file.write(reply)

    try:
        parsed = json.loads(reply)
        return parsed.get("results", {})
    except json.JSONDecodeError:
        # Log malformed JSON to separate file
        print(f"Batch {batch_num}: Unexpected non-JSON reply, saved to {raw_path}")
        with open(ERROR_LOG, "a") as log:
            log.write(f"\n=== Malformed Reply (Batch {batch_num}) ===\n")
            log.write(reply)
            log.write("\n=======================\n")
        return {}  # safely skip JSON parsing

def main():
    # Load existing results (resume support)
    results = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}

    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".json")]
    remaining = [f for f in files if f not in results]
    print(f"Found {len(files)} JSON files ({len(remaining)} remaining)")

    idx = 0
    batch_num = 1

    while idx < len(remaining):
        batch_files = remaining[idx: idx + BATCH_SIZE]
        batch_data = {}
        for fname in batch_files:
            with open(os.path.join(RAW_DATA_DIR, fname), "r") as f:
                batch_data[fname] = json.load(f)

        print(f"\nBatch {batch_num}: {len(batch_files)} files")
        batch_results = {}

        for attempt in range(MAX_RETRIES):
            try:
                batch_results = classify_batch(batch_data, batch_num)
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                sleep(3)
                if attempt == MAX_RETRIES - 1:
                    print(f"Skipping batch {batch_num} after multiple failures.")

        # Merge valid results
        if not batch_results:
            for fname in batch_files:
                results[fname] = "unclassified"
        else:
            for fname in batch_files:
                results[fname] = batch_results.get(fname, "unclassified")

        # Save progress after each batch
        with open(OUTPUT_FILE, "w") as out:
            json.dump(results, out, indent=4)

        print(f" Batch {batch_num} done → {len(batch_results)} results saved.")
        idx += BATCH_SIZE
        batch_num += 1
        sleep(2)

    print("\nAll files processed successfully!")
    print(f"Final results saved in: {OUTPUT_FILE}")
    print(f"Raw replies saved in: {RAW_REPLY_DIR}")
    print(f"Any malformed responses logged in: {ERROR_LOG}")

if __name__ == "__main__":
    main()
