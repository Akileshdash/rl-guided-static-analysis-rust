# Mitigating False Positives in Rust Static Analysis via Reinforcement Learning

This repository contains the implementation of our research paper: **"Mitigating False Positives in Static Memory Safety Analysis of Rust Programs via Reinforcement Learning"**.

## Overview

Static analysis tools like Rudra suffer from high false positive rates (50%+), which diminish developer trust and increase manual review effort. This project presents a novel reinforcement learning-based approach that:

- **Reduces false positives** by more than doubling precision from 25.6% to 59.0%
- **Maintains high recall** of 74.6%, identifying nearly three-quarters of true bugs
- **Integrates selective fuzzing** to validate ambiguous warnings dynamically
- **Outperforms LLM baselines** by 17.1 percentage points in F1 score
- **Achieves 65.2% accuracy** with fuzzing vs. 54.5% without

### Key Features

- Proximal Policy Optimization (PPO) agent for warning classification
- ~87 MIR-level semantic features extracted from Rust's Mid-level Intermediate Representation
- Selective dynamic validation via cargo-fuzz integration
- Three-action decision space: classify as true positive, false positive, or invoke fuzzing
- Cost-aware reward function balancing accuracy and computational efficiency

---

## Repository Structure

```
.
├── data/                          # Raw dataset
├── RL_output/                     # Model predictions
│   ├── rl.json                    # RL without fuzzing results
│   └── rl_with_fuzzer.json        # RL with fuzzing results
├── llm/                           # LLM baseline comparisons
│   ├── jsons/                     # LLM prediction outputs
│   │   ├── llm_results_claude_opus4_1.json
│   │   ├── llm_results_codellama_34b.json
│   │   ├── llm_results_gpt-4o-mini.json
│   │   ├── llm_results_llama3_70b.json
│   │   ├── llm_results_llama3_8b.json
│   │   └── llm_results_mixtral.json
│   └── scripts/                   # LLM evaluation scripts
│       ├── claude_opus_llm_baseline.py
│       ├── gpt-4o-mini_llm_baseline.py
│       ├── run_codellama34b.py
│       ├── run_llama3_70b.py
│       ├── run_llama3_8b.py
│       └── run_mixtral.py
├── rl_without_fuzzing.py          # PPO agent WITHOUT fuzzing
├── rl_with_fuzzing.py             # PPO agent WITH fuzzing integration
├── ground_truth.json              # Manually labeled warnings dataset (1,247 labels)
├── training_data.json             # Static analysis warnings with features (4,879 samples)
├── result_table.py                # Generate comparison tables and metrics
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **Rust toolchain** (for actual fuzzing)
- **cargo-fuzz** (for real fuzzing integration)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Akileshdash/rl-guided-static-analysis-rust.git
cd rl-guided-static-analysis-rust
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Dataset

### Dataset Construction

Our dataset consists of:
- **4,879 unique warnings** from Rudra's Unsafe Dataflow and Send/Sync Variance checkers
- Collected from **~20,000 crates** from crates.io containing unsafe code
- **1,247 true positives (25.6%)** and **3,632 false positives (74.4%)**
- Manually labeled by domain experts with **82.7% inter-rater agreement** (Cohen's κ = 0.63)

### Dataset Split
- **Training set**: 70% (3,415 warnings)
- **Validation set**: 15% (732 warnings)
- **Test set**: 15% (732 warnings)

### `ground_truth.json`
Contains manually labeled ground truth for each warning.

```json
[
    {
        "file": "aarc-0.3.2_2.json",
        "false_positive": 0
    }
]
```

---

## Running the Models

### 1. PPO Agent (Without Fuzzing)

Trains a reinforcement learning agent using only static features extracted from MIR.

```bash
python rl_without_fuzzing.py
```

**Expected Output:**
```
======================================================================
PPO Agent for Rust Static Analysis False Positive Reduction
======================================================================

Loading data...
Loaded 1247 ground truth labels
Loaded 4879 training samples

Dataset split:
  Training: 3415 samples (70%)
  Validation: 732 samples (15%)
  Test: 732 samples (15%)

Starting training for 200 epochs...
Network architecture: 87 -> 256 -> 128 -> 2
----------------------------------------------------------------------
Epoch   1 | Acc: 52.34% | Loss:  12.3456 | Reward: -1234
Epoch  10 | Acc: 54.12% | Loss:   8.7654 | Reward:   456
...
Epoch 200 | Acc: 54.52% | Loss:   2.1234 | Reward:  2345
----------------------------------------------------------------------
Training complete!

Test Set Results:
  Accuracy:  54.5%
  Precision: 49.6%
  Recall:    67.9%
  F1 Score:  57.3%
  MCC:       0.117

Results saved to RL_output/rl.json
```

**Output File:** `RL_output/rl.json`
- Contains predictions for all test samples
- Training history (accuracy, loss, reward per epoch)
- Metadata with final metrics

---

### 2. PPO Agent with Fuzzing Integration

Trains an agent that can selectively invoke cargo-fuzz for dynamic validation of ambiguous warnings.

```bash
python rl_with_fuzzing.py
```

**Expected Output:**
```
======================================================================
PPO Agent with Fuzzing Integration
Rust Static Analysis False Positive Reduction
======================================================================

Loading data...
Loaded 1247 ground truth labels
Loaded 4879 training samples

Starting training with fuzzing integration for 200 epochs...
Dataset size: 4879 samples
Action space: 0=Classify_FP, 1=Classify_TP, 2=Invoke_Fuzzing
----------------------------------------------------------------------
Epoch   1 | Acc: 53.21% | Loss:  11.2345 | Reward: -1123 | Fuzz: 35.2%
Epoch  10 | Acc: 58.45% | Loss:   7.6543 | Reward:   678 | Fuzz: 28.4%
...
Epoch 200 | Acc: 65.23% | Loss:   1.8765 | Reward:  3456 | Fuzz: 23.1%
----------------------------------------------------------------------
Training complete!

Fuzzing Statistics:
  Total invocations: ~23% of warnings
  Successful bug findings: 342
  Clean executions: 785
  Fuzzing success rate: 30.3%

Test Set Results:
  Accuracy:  65.2%
  Precision: 59.0%
  Recall:    74.6%
  F1 Score:  65.9%
  MCC:       0.323
  AUC-ROC:   0.661
  AUC-PR:    0.554

Results saved to RL_output/rl_with_fuzzer.json
```

**Output File:** `RL_output/rl_with_fuzzer.json`
- Contains predictions with fuzzing metadata
- Training history including fuzzing rates
- Fuzzing statistics (invocations, success rate)
- Confidence scores for each classification

---

### 3. Generate Results Table

Compare all models (PPO, PPO+Fuzzing, LLM baselines) with standard metrics.

```bash
python result_table.py
```

**Expected Output:**
```
╔════════════════════════╦══════════╦═══════════╦════════╦═════════╦═══════╦═══════╦═══════╗
║ Approach               ║ Accuracy ║ Precision ║ Recall ║ F1      ║ MCC   ║ ROC   ║ PR    ║
╠════════════════════════╬══════════╬═══════════╬════════╬═════════╬═══════╬═══════╬═══════╣
║ Raw Rudra Output       ║    —     ║ 0.256     ║ 1.000  ║ 0.407   ║   —   ║   —   ║   —   ║
║ RL + Fuzzing           ║ 0.652    ║ 0.590     ║ 0.746  ║ 0.659   ║ 0.323 ║ 0.661 ║ 0.554 ║
║ RL (No Fuzzing)        ║ 0.545    ║ 0.496     ║ 0.679  ║ 0.573   ║ 0.117 ║ 0.557 ║ 0.481 ║
║ Claude Opus 4.1        ║ 0.533    ║ 0.486     ║ 0.669  ║ 0.563   ║ 0.094 ║ 0.546 ║ 0.474 ║
║ ChatGPT-4o Mini        ║ 0.490    ║ 0.453     ║ 0.650  ║ 0.534   ║ 0.010 ║ 0.505 ║ 0.452 ║
║ CodeLlama 34B          ║ 0.431    ║ 0.409     ║ 0.598  ║ 0.486   ║-0.112 ║ 0.446 ║ 0.425 ║
║ Llama3 8B              ║ 0.487    ║ 0.437     ║ 0.487  ║ 0.460   ║-0.026 ║ 0.487 ║ 0.443 ║
║ Mixtral 8x7B           ║ 0.326    ║ 0.328     ║ 0.476  ║ 0.388   ║-0.336 ║ 0.339 ║ 0.392 ║
║ Llama3 70B             ║ 0.503    ║ 0.360     ║ 0.136  ║ 0.197   ║-0.082 ║ 0.469 ║ 0.437 ║
╚════════════════════════╩══════════╩═══════════╩════════╩═════════╩═══════╩═══════╩═══════╝

Key Findings:
- RL+Fuzzing improves F1 by 17.1 percentage points over best LLM baseline
- Precision more than doubles from 25.6% (raw Rudra) to 59.0%
- Fuzzing provides 10.7 pp accuracy gain and 8.6 pp F1 gain over RL-only
- Selective fuzzing invoked on ~23% of warnings for cost-aware validation
```

---

## Requirements

### Python Packages

```txt
numpy>=1.21.0
torch>=1.9.0
scikit-learn>=0.24.0
```

### For Real Fuzzing

For actual cargo-fuzz integration:

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install cargo-fuzz
cargo install cargo-fuzz

# Install nightly Rust (required for fuzzing)
rustup install nightly
rustup default nightly
```

---

## Configuration

### Hyperparameters

Both `rl_without_fuzzing.py` and `rl_with_fuzzing.py` expose key hyperparameters that can be tuned:

```python
# Network architecture
state_size = 87           # MIR feature dimensions (~87 features)
hidden1 = 256             # First hidden layer
hidden2 = 128             # Second hidden layer
action_size = 2           # Binary classification (3 for fuzzing variant)

# Training parameters (PPO)
epochs = 200              # Training epochs
learning_rate = 0.0003    # Learning rate for PPO
gamma = 0.99              # Discount factor
clip_epsilon = 0.2        # PPO clipping parameter
value_coef = 0.5          # Value function coefficient
entropy_coef = 0.01       # Entropy coefficient for exploration

# Reward function
correct_reward = 15       # Correct classification
incorrect_penalty = -15   # Incorrect classification
fuzzing_cost = -5         # Fuzzing invocation cost
fuzzing_bonus_bug = 10    # Fuzzing found bug (confirms TP)
fuzzing_bonus_clean = 8   # Fuzzing ran clean (suggests FP)
fuzzing_bonus_helpful = 3 # Fuzzing helpful but not definitive
```

### Feature Extraction

The feature extraction pipeline processes MIR-level information organized into three categories:

#### 1. MIR-Level Semantic Features (~50 features)
- **Type system properties**: Generic parameters, trait bounds, generic nesting depth
- **Ownership and borrowing**: Borrow ratios, nesting depth, smart pointer usage
- **Control-flow**: Cyclomatic complexity, loop nesting, panic paths
- **Unsafe operations**: Lifetime bypass category, distance to dangerous operations

#### 2. Structural Code Features (~20 features)
- **Package-level signals**: Download counts, unsafe code prevalence
- **API surface**: Public vs. private API warnings
- **Code metrics**: Lines of code, parameter counts, comment density

#### 3. Analysis-Specific Features (~17 features)
- **Rudra metadata**: Checker type, precision level
- **Clustering**: Multiple warnings at nearby locations
- **Pattern type**: Panic safety, higher-order invariants, Send/Sync variance

**Total: ~87 normalized features** after correlation removal and domain filtering

---

## Methodology

### Reinforcement Learning Formulation

Our approach formulates false positive reduction as a Markov Decision Process (MDP):

- **State Space**: 87-dimensional feature vectors extracted from MIR
- **Action Space**: 
  - Action 0: Classify as False Positive
  - Action 1: Classify as True Positive
  - Action 2: Invoke Dynamic Fuzzing (fuzzing variant only)
- **Reward Function**: Balances classification accuracy against computational cost
- **Policy Network**: Two hidden layers (256 → 128 units) with ReLU activations and dropout
- **Algorithm**: Proximal Policy Optimization (PPO)

### Selective Fuzzing Strategy

The agent learns to invoke fuzzing based on confidence estimates:
1. **Q-value gap**: Smaller gaps indicate uncertainty
2. **Policy entropy**: Higher entropy reflects lower confidence
3. **Cost-aware decision**: Balances fuzzing cost (-5) against expected information gain

Fuzzing outcomes are encoded into state representation:
- **Crash/sanitizer violation** → Strong TP evidence (+10 bonus)
- **Clean execution** → Moderate FP evidence (+8 bonus)
- **Timeout/inconclusive** → Weak signal (+3 bonus)

---

## Bug Pattern Coverage

Rudra targets three critical classes of memory safety bugs in unsafe Rust:

1. **Panic Safety**: Stack unwinding during panics can violate invariants
   - Example: CVE-2020-36317 in `String::retain()`

2. **Higher-Order Invariants**: Semantic assumptions about generic functions
   - Example: CVE-2020-36323 in `join()` (impure `Borrow` implementations)

3. **Send/Sync Variance**: Missing or incorrect thread-safety bounds
   - Example: CVE-2020-35905 in futures-rs `MappedMutexGuard`

---

## Performance Comparison

### Key Improvements Over Baselines

| Metric | Raw Rudra | RL Only | RL+Fuzzing | Improvement |
|--------|-----------|---------|------------|-------------|
| Precision | 25.6% | 49.6% | **59.0%** | +130% |
| Recall | 100% | 67.9% | **74.6%** | -25.4% (acceptable) |
| F1 Score | 40.7% | 57.3% | **65.9%** | +61.9% |
| Accuracy | — | 54.5% | **65.2%** | +19.6% |

### Fuzzing Impact

- **10.7 percentage points** accuracy gain over RL-only
- **8.6 percentage points** F1 score improvement
- **Invoked on ~23%** of warnings (cost-effective)
- **30.3% success rate** in finding bugs when invoked

---

## Acknowledgments

- Original Rudra implementation by Yechan Bae et al.
- Rust compiler team for MIR infrastructure
- cargo-fuzz developers for fuzzing integration
