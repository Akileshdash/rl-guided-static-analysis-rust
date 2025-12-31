import os
import json
import subprocess

data_folder = "data"
results_file = "llm_results_codellama_70b.json"

PROMPT_TEMPLATE = """
You are analyzing static analysis results from a Rust code analyzer.
Here is one report in JSON:

{json_data}

Task:
- Decide if this report is a FALSE POSITIVE (not really an issue) or a REAL ISSUE.
- If you have **any doubt at all** or cannot confidently decide, treat it as a FALSE POSITIVE.
- Reply with ONLY one word: "false_positive" or "real_issue".
"""

def query_ollama(prompt, model="codellama:70b"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8").strip().lower()

def main():
    results = {}
    for filename in sorted(os.listdir(data_folder)):
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, "r") as f:
                data = json.load(f)

            prompt = PROMPT_TEMPLATE.format(json_data=json.dumps(data, indent=2))
            answer = query_ollama(prompt)

            if "real_issue" in answer:
                label = 0
            else:
                label = 1

            results[filename] = label
            print(f"[mixtral:8x7b] {filename} → raw: {answer} → label: {label}")

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f" Finished mixtral:8x7b, saved results to {results_file}")

if __name__ == "__main__":
    main()
