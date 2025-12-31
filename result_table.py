import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)

LLM_FILES = {
    "RL + Fuzzing (Ours)": "RL_output/rl_with_fuzzer.json",
    "RL (No Fuzzing)": "RL_output/rl.json",
    "Claude Opus 4.1": "llm/jsons/llm_results_claude_opus4_1.json",
    "Llama3 70B": "llm/jsons/llm_results_llama3_70b.json",
    "ChatGPT-4o Mini": "llm/jsons/llm_results_gpt-4o-mini.json",
    "CodeLlama 34B": "llm/jsons/llm_results_codellama_34b.json",
    "Mixtral 8x7B": "llm/jsons/llm_results_mixtral.json",
    "Llama3 8B": "llm/jsons/llm_results_llama3_8b.json",
}

GROUND_TRUTH_FILE = "ground_truth.json"


def load_ground_truth():
    with open(GROUND_TRUTH_FILE, "r") as f:
        data = json.load(f)
    return {entry["file"]: entry["false_positive"] for entry in data}


def evaluate_model(pred_file, ground_truth):
    # Load predictions
    with open(pred_file, "r") as f:
        preds = json.load(f)

    # Only compare overlapping files
    common = [f for f in ground_truth if f in preds]

    if len(common) == 0:
        return (0, 0, 0, 0, 0, 0.5, 0)

    y_true = np.array([ground_truth[f] for f in common])
    y_pred = np.array([preds[f] for f in common])

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # AUC metrics (safe handling)
    try:
        auc_roc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc_roc = 0.5

    try:
        auc_pr = average_precision_score(y_true, y_pred)
    except ValueError:
        auc_pr = 0.0

    return acc, prec, rec, f1, mcc, auc_roc, auc_pr


def run_all():
    ground_truth = load_ground_truth()

    print("| Approach | Accuracy | Precision | Recall | F1 Score | MCC | AUC-ROC | AUC-PR |")
    print("|----------|----------|-----------|--------|----------|-----|---------|--------|")

    for name, file in LLM_FILES.items():
        acc, prec, rec, f1, mcc, auc_roc, auc_pr = evaluate_model(file, ground_truth)

        print(
            f"| {name} | "
            f"{acc:.3f} | {prec:.3f} | {rec:.3f} | {f1:.3f} | "
            f"{mcc:.3f} | {auc_roc:.3f} | {auc_pr:.3f} |"
        )


if __name__ == "__main__":
    run_all()
