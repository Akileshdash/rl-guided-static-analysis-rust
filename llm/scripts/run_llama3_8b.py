import os
import json
import subprocess

data_folder = "data"
results_file = "llm_results_llama3_8b.json"

PROMPT_TEMPLATE = """
You are analyzing static analysis results produced by **Rudra**, a Rust safety analyzer.
Each result is a JSON object with fields such as `level`, `analyzer`, `description`, and `code_snippet`.

Here is one result in JSON:

{json_data}

Your task:
- Read the `code_snippet` carefully and cross-check with the `analyzer`, `description`, and `level`.
- Decide whether this report indicates a **REAL ISSUE** (genuine unsafe bug) or a **FALSE POSITIVE** (not an issue). 
- Only choose "false_positive" if you are confident it is harmless. 
- Otherwise, return "real_issue".
- Reply with ONLY one word: "real_issue" or "false_positive".
"""

def query_ollama(prompt, model="llama3:8b"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8").strip().lower()

def main():
    # Load previous results if they exist
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}

    for filename in sorted(os.listdir(data_folder)):
        if not filename.endswith(".json"):
            continue

        # Skip if already processed
        if filename in results:
            print(f"Skipping {filename}, already in results.")
            continue

        file_path = os.path.join(data_folder, filename)
        with open(file_path, "r") as f:
            data = json.load(f)

        prompt = PROMPT_TEMPLATE.format(json_data=json.dumps(data, indent=2))
        answer = query_ollama(prompt)

        answer = answer.strip().lower().replace(" ", "").replace("-", "").replace("_","")
        if "realissue" in answer:
            label = 0
        else:
            label = 1

        results[filename] = label
        print(f"[llama3_3b] {filename} → raw: {answer} → label: {label}")

        # Save after each file (so we don’t lose progress)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

    print(f" Finished llama3:8b, saved results to {results_file}")

if __name__ == "__main__":
    main()