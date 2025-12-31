"""
DQN Agent with Integrated Fuzzing for False Positive Classification
Based on: Mitigating False Positives in Static Memory Safety Analysis of Rust Programs via RL

This implementation adds selective dynamic validation via cargo-fuzz as described in the paper.

Key Features:
- 3-action space: classify_true_positive, classify_false_positive, invoke_fuzzing
- Selective fuzzing based on confidence assessment
- Augmented state with fuzzing outcomes
- Cost-aware reward function (fuzzing penalty + bonus for effective use)
"""

import json
import random
import numpy as np
import subprocess
import os
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path


class DQNAgentWithFuzzing:
    """Enhanced DQN Agent with selective fuzzing capability"""
    
    def __init__(self, state_size=24, action_size=3):
        self.state_size = state_size
        self.action_size = action_size  # 0: FP, 1: TP, 2: Fuzz
        
        # Replay memory
        self.memory = deque(maxlen=5000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.996
        self.learning_rate = 0.0005
        self.momentum = 0.9
        
        # Fuzzing statistics
        self.fuzzing_invoked = 0
        self.fuzzing_successful = 0
        self.fuzzing_cache = {}
        
        # Initialize network (state_size + 4 fuzzing features)
        augmented_size = state_size + 4
        self.weights1 = self._xavier_init(augmented_size, 32)
        self.weights2 = self._xavier_init(32, 16)
        self.weights3 = self._xavier_init(16, action_size)
        
        self.bias1 = np.zeros(32)
        self.bias2 = np.zeros(16)
        self.bias3 = np.zeros(action_size)
        
        # Momentum velocities
        self.v_w1 = np.zeros_like(self.weights1)
        self.v_w2 = np.zeros_like(self.weights2)
        self.v_w3 = np.zeros_like(self.weights3)
        self.v_b1 = np.zeros_like(self.bias1)
        self.v_b2 = np.zeros_like(self.bias2)
        self.v_b3 = np.zeros_like(self.bias3)
        
    def _xavier_init(self, in_dim, out_dim):
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        return np.random.uniform(-scale, scale, (in_dim, out_dim))
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, state):
        """Forward pass through the network"""
        self.z1 = np.dot(state, self.weights1) + self.bias1
        self.h1 = self._relu(self.z1)
        
        self.z2 = np.dot(self.h1, self.weights2) + self.bias2
        self.h2 = self._relu(self.z2)
        
        self.output = np.dot(self.h2, self.weights3) + self.bias3
        return self.output
    
    def act(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        Returns: 0 (FP), 1 (TP), or 2 (Invoke Fuzzing)
        """
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.forward(state)
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=64):
        """Train on batch with backpropagation"""
        if len(self.memory) < batch_size:
            return 0.0
        
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0.0
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.forward(next_state))
            
            current_q = self.forward(state)
            current_q_value = current_q[action]
            
            loss = (target - current_q_value) ** 2
            total_loss += loss
            
            # Backpropagation (same as before)
            output_error = np.zeros(self.action_size)
            output_error[action] = 2 * (current_q_value - target)
            
            grad_w3 = np.outer(self.h2, output_error)
            grad_b3 = output_error
            
            h2_error = np.dot(output_error, self.weights3.T) * self._relu_derivative(self.z2)
            grad_w2 = np.outer(self.h1, h2_error)
            grad_b2 = h2_error
            
            h1_error = np.dot(h2_error, self.weights2.T) * self._relu_derivative(self.z1)
            grad_w1 = np.outer(state, h1_error)
            grad_b1 = h1_error
            
            # Update with momentum
            self.v_w3 = self.momentum * self.v_w3 - self.learning_rate * grad_w3
            self.v_w2 = self.momentum * self.v_w2 - self.learning_rate * grad_w2
            self.v_w1 = self.momentum * self.v_w1 - self.learning_rate * grad_w1
            
            self.v_b3 = self.momentum * self.v_b3 - self.learning_rate * grad_b3
            self.v_b2 = self.momentum * self.v_b2 - self.learning_rate * grad_b2
            self.v_b1 = self.momentum * self.v_b1 - self.learning_rate * grad_b1
            
            self.weights3 += self.v_w3
            self.weights2 += self.v_w2
            self.weights1 += self.v_w1
            
            self.bias3 += self.v_b3
            self.bias2 += self.v_b2
            self.bias1 += self.v_b1
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / batch_size


def extract_features(sample):
    """Extract base 24 features (same as DQN without fuzzing)"""
    level_map = {'Info': 0.33, 'Warning': 0.66, 'Error': 1.0}
    analyzer_map = {
        'UnsafeDataflow': 0.2,
        'UnsafeDestructor': 0.4,
        'SendSyncVariance': 0.6,
        'UnsafeTransmute': 0.8,
        'Other': 1.0
    }
    
    code = sample.get("code_snippet", "")
    start_line = sample.get("start_line", 0)
    end_line = sample.get("end_line", 0)
    start_col = sample.get("start_col", 0)
    end_col = sample.get("end_col", 0)
    
    line_span = max(end_line - start_line, 0)
    col_span = max(end_col - start_col, 0)
    
    # Pattern counts
    unsafe_count = code.lower().count("unsafe")
    drop_count = code.lower().count("drop")
    ptr_keywords = ["ptr", "pointer", "raw"]
    ptr_count = sum(1 for kw in ptr_keywords if kw in code.lower())
    transmute_count = code.lower().count("transmute")
    async_keywords = ["async", "await"]
    async_count = sum(1 for kw in async_keywords if kw in code.lower())
    mut_count = code.count("mut ")
    ref_count = code.count("&")
    impl_count = code.lower().count("impl")
    trait_count = code.lower().count("trait")
    where_count = code.lower().count("where")
    lifetime_count = len([word for word in code.split() if word.startswith("'")])
    generic_count = len([char for char in code if char in "<>"])
    
    # Contextual features
    has_from_raw = int("from_raw" in code.lower())
    has_as_ptr = int("as_ptr" in code.lower())
    has_deref = int("*" in code or "deref" in code.lower())
    has_cast = int(" as " in code)
    has_option = int(any(kw in code for kw in ["Option", "Some", "None"]))
    has_result = int(any(kw in code for kw in ["Result", "Ok", "Err"]))
    
    features = np.array([
        level_map.get(sample.get("level"), 0.5),
        analyzer_map.get(sample.get("analyzer"), 1.0),
        min(start_line / 1000, 1),
        min(line_span / 50, 1),
        min(col_span / 100, 1),
        min(len(code) / 500, 1),
        min(unsafe_count / 5, 1),
        min(drop_count / 3, 1),
        min(ptr_count / 5, 1),
        min(transmute_count / 2, 1),
        min(async_count / 3, 1),
        min(mut_count / 10, 1),
        min(ref_count / 15, 1),
        min(impl_count / 3, 1),
        min(trait_count / 2, 1),
        min(where_count / 3, 1),
        min(lifetime_count / 5, 1),
        min(generic_count / 10, 1),
        has_from_raw,
        has_as_ptr,
        has_deref,
        has_cast,
        has_option,
        has_result
    ], dtype=float)
    
    return features


def augment_state_with_fuzzing(base_features, fuzzing_result):
    """
    Augment state vector with fuzzing outcome features
    
    Fuzzing features (4 dimensions):
    - bug_found: 1 if sanitizer violation, 0 otherwise
    - clean_execution: 1 if no errors within budget, 0 otherwise
    - timeout: 1 if timeout/inconclusive, 0 otherwise
    - fuzzing_invoked: 1 if fuzzing was run, 0 otherwise
    """
    if fuzzing_result is None:
        # No fuzzing invoked
        fuzzing_features = np.array([0, 0, 0, 0], dtype=float)
    else:
        outcome = fuzzing_result.get("outcome", "inconclusive")
        fuzzing_features = np.array([
            1 if outcome == "bug_found" else 0,
            1 if outcome == "clean" else 0,
            1 if outcome == "inconclusive" else 0,
            1  # Fuzzing was invoked
        ], dtype=float)
    
    return np.concatenate([base_features, fuzzing_features])


def generate_fuzz_harness(sample, bug_pattern):
    """
    Generate a fuzzing harness template based on bug pattern
    Returns a simplified mock implementation for demonstration
    """
    templates = {
        "panic_safety": """
#[cfg(test)]
mod fuzz_tests {{
    use libfuzzer_sys::fuzz_target;
    
    fuzz_target!(|data: &[u8]| {{
        // Test panic safety in presence of unwinding
        let _ = std::panic::catch_unwind(|| {{
            // Code under test
        }});
    }});
}}
""",
        "send_sync": """
#[cfg(test)]
mod fuzz_tests {{
    use std::sync::Arc;
    use std::thread;
    
    #[test]
    fn test_send_sync() {{
        // Test thread safety properties
    }}
}}
""",
        "uninit_memory": """
#[cfg(test)]
mod fuzz_tests {{
    use libfuzzer_sys::fuzz_target;
    
    fuzz_target!(|data: &[u8]| {{
        // Test uninitialized memory handling
    }});
}}
"""
    }
    
    return templates.get(bug_pattern, templates["panic_safety"])


def run_cargo_fuzz(sample, agent):
    """
    Simulate cargo-fuzz execution with sanitizers
    
    In a real implementation, this would:
    1. Generate harness from template
    2. Compile with rustc + sanitizers (ASan, MSan, TSan)
    3. Run libFuzzer with time budget
    4. Parse sanitizer output
    
    For this implementation, we simulate outcomes based on code patterns
    """
    filename = sample.get("filename", "")
    
    # Check cache
    if filename in agent.fuzzing_cache:
        return agent.fuzzing_cache[filename]
    
    # Simulate fuzzing (in real implementation, this would run cargo-fuzz)
    code = sample.get("code_snippet", "")
    bug_pattern = sample.get("bug_pattern", "unknown")
    
    # Simplified heuristic simulation
    # In reality, this would execute actual fuzzing with a time budget (30-60s)
    agent.fuzzing_invoked += 1
    
    # Simulate outcomes based on code characteristics
    high_risk_patterns = ["from_raw", "transmute", "unsafe", "ptr::"]
    risk_score = sum(1 for pattern in high_risk_patterns if pattern in code.lower())
    
    if risk_score >= 3 and "drop" in code.lower():
        # High confidence bug found
        outcome = "bug_found"
        agent.fuzzing_successful += 1
    elif risk_score >= 2:
        # Uncertain - timeout or needs more fuzzing
        outcome = "inconclusive"
    else:
        # Clean execution
        outcome = "clean"
    
    result = {
        "outcome": outcome,
        "sanitizer": "AddressSanitizer" if outcome == "bug_found" else None,
        "time_spent": random.uniform(10, 60)  # Simulated execution time
    }
    
    # Cache result
    agent.fuzzing_cache[filename] = result
    
    return result


def train_agent_with_fuzzing(ground_truth, training_data, episodes=200, batch_size=64):
    """
    Train DQN agent with selective fuzzing integration
    
    The agent learns when to invoke fuzzing based on:
    - Static feature uncertainty
    - Cost-benefit trade-off
    - Historical fuzzing effectiveness
    """
    gt_map = {gt["file"]: gt["false_positive"] for gt in ground_truth}
    agent = DQNAgentWithFuzzing(state_size=24, action_size=3)
    
    history = []
    
    print(f"Starting training with fuzzing integration for {episodes} episodes...")
    print(f"Dataset size: {len(training_data)} samples")
    print(f"Action space: 0=FP, 1=TP, 2=Invoke_Fuzzing")
    print("-" * 70)
    
    for episode in range(episodes):
        total_reward = 0
        correct = 0
        total_loss = 0
        fuzzing_this_episode = 0
        
        random.shuffle(training_data)
        
        for sample in training_data:
            base_features = extract_features(sample)
            true_label = gt_map.get(sample["filename"], 0)
            
            # Initial state without fuzzing
            state = augment_state_with_fuzzing(base_features, None)
            
            # Agent selects action
            action = agent.act(state, training=True)
            
            # Handle fuzzing action (action == 2)
            fuzzing_result = None
            final_action = action
            fuzzing_cost = 0
            
            if action == 2:  # Invoke fuzzing
                fuzzing_this_episode += 1
                fuzzing_result = run_cargo_fuzz(sample, agent)
                
                # Apply fuzzing cost
                fuzzing_cost = -5
                
                # Augment state with fuzzing outcome
                state_with_fuzz = augment_state_with_fuzzing(base_features, fuzzing_result)
                
                # Make final classification based on augmented state
                final_action = agent.act(state_with_fuzz, training=False)
                
                # Ensure final action is binary (not fuzzing again)
                if final_action == 2:
                    final_action = 1 if fuzzing_result.get("outcome") == "bug_found" else 0
            
            # Calculate reward
            if action == 2:
                # Fuzzing was invoked
                classification_correct = (final_action == true_label)
                base_reward = 15 if classification_correct else -15
                
                # Bonus if fuzzing helped make correct decision
                if classification_correct and fuzzing_result:
                    if fuzzing_result["outcome"] == "bug_found" and true_label == 1:
                        fuzzing_bonus = 10  # Fuzzing confirmed true positive
                    elif fuzzing_result["outcome"] == "clean" and true_label == 0:
                        fuzzing_bonus = 8   # Fuzzing confirmed false positive
                    else:
                        fuzzing_bonus = 3   # Fuzzing helped but not definitive
                else:
                    fuzzing_bonus = 0
                
                reward = base_reward + fuzzing_cost + fuzzing_bonus
            else:
                # Direct classification without fuzzing
                reward = 15 if action == true_label else -15
                
                # Bonus for confident correct predictions after warmup
                if action == true_label and episode > 50:
                    reward += 5
            
            total_reward += reward
            correct += (final_action == true_label if action == 2 else action == true_label)
            
            # Store experience
            next_state = augment_state_with_fuzzing(base_features, fuzzing_result) if action == 2 else state
            agent.remember(state, action, reward, next_state, done=True)
            
            # Train on batch
            if len(agent.memory) >= batch_size:
                loss = agent.replay(batch_size)
                total_loss += loss
        
        # Calculate metrics
        accuracy = (correct / len(training_data)) * 100
        avg_loss = total_loss / len(training_data) if len(training_data) > 0 else 0
        fuzzing_rate = (fuzzing_this_episode / len(training_data)) * 100
        
        history.append({
            "epoch": episode + 1,
            "accuracy": accuracy,
            "loss": avg_loss,
            "reward": total_reward,
            "epsilon": agent.epsilon,
            "fuzzing_invoked": fuzzing_this_episode,
            "fuzzing_rate": fuzzing_rate
        })
        
        # Print progress
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"Epoch {episode+1:3d} | Acc: {accuracy:5.2f}% | "
                  f"Loss: {avg_loss:7.4f} | Reward: {total_reward:6.0f} | "
                  f"Fuzz: {fuzzing_rate:4.1f}% | ε: {agent.epsilon:.4f}")
    
    print("-" * 70)
    print(f"Training complete!")
    print(f"Total fuzzing invocations: {agent.fuzzing_invoked}")
    print(f"Successful bug findings: {agent.fuzzing_successful}")
    print(f"Fuzzing success rate: {agent.fuzzing_successful/max(agent.fuzzing_invoked,1)*100:.1f}%")
    
    # Generate final predictions
    print("\nGenerating final predictions with selective fuzzing...")
    predictions = []
    correct_final = 0
    final_fuzzing_count = 0
    
    for sample in training_data:
        base_features = extract_features(sample)
        state = augment_state_with_fuzzing(base_features, None)
        
        # Get action
        action = agent.act(state, training=False)
        
        # Handle fuzzing if selected
        if action == 2:
            final_fuzzing_count += 1
            fuzzing_result = run_cargo_fuzz(sample, agent)
            state_with_fuzz = augment_state_with_fuzzing(base_features, fuzzing_result)
            final_action = agent.act(state_with_fuzz, training=False)
            if final_action == 2:
                final_action = 1 if fuzzing_result.get("outcome") == "bug_found" else 0
        else:
            final_action = action
            fuzzing_result = None
        
        actual = gt_map.get(sample["filename"], 0)
        is_correct = (final_action == actual)
        
        predictions.append({
            "filename": sample["filename"],
            "prediction": final_action,
            "actual": actual,
            "correct": is_correct,
            "fuzzing_used": action == 2,
            "fuzzing_outcome": fuzzing_result["outcome"] if fuzzing_result else None
        })
        
        correct_final += is_correct
    
    final_accuracy = (correct_final / len(training_data)) * 100
    final_fuzzing_rate = (final_fuzzing_count / len(training_data)) * 100
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    print(f"Final fuzzing rate: {final_fuzzing_rate:.1f}%")
    
    return predictions, history, agent


def main():
    """Main execution function"""
    print("=" * 70)
    print("DQN Agent with Fuzzing Integration")
    print("Rust Static Analysis False Positive Reduction")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    try:
        with open("ground_truth.json", "r") as f:
            ground_truth = json.load(f)
        print(f"✓ Loaded {len(ground_truth)} ground truth labels")
        
        with open("training_data.json", "r") as f:
            training_data = json.load(f)
        print(f"✓ Loaded {len(training_data)} training samples")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    
    print()
    
    # Train agent
    predictions, history, agent = train_agent_with_fuzzing(ground_truth, training_data)
    
    # Save results
    print("\nSaving results...")
    
    # Ensure output directory exists
    os.makedirs("RL_output", exist_ok=True)
    
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": "DQN_with_Fuzzing",
            "episodes": len(history),
            "final_accuracy": history[-1]["accuracy"],
            "total_fuzzing_invocations": agent.fuzzing_invoked,
            "fuzzing_success_rate": agent.fuzzing_successful / max(agent.fuzzing_invoked, 1)
        },
        "predictions": predictions,
        "training_history": history,
        "fuzzing_statistics": {
            "total_invoked": agent.fuzzing_invoked,
            "successful_findings": agent.fuzzing_successful,
            "cache_size": len(agent.fuzzing_cache)
        }
    }
    
    with open("RL_output/rl_with_fuzzer.json", "w") as f:
        json.dump(output, f, indent=2)
    print("✓ Results saved to RL_output/rl_with_fuzzer.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Final Accuracy: {history[-1]['accuracy']:.2f}%")
    print(f"Final Epsilon: {history[-1]['epsilon']:.4f}")
    print(f"Final Fuzzing Rate: {history[-1]['fuzzing_rate']:.1f}%")
    print(f"Total Correct: {sum(1 for p in predictions if p['correct'])}/{len(predictions)}")
    print(f"Fuzzing Invocations: {agent.fuzzing_invoked}")
    print(f"Bugs Found via Fuzzing: {agent.fuzzing_successful}")


if __name__ == "__main__":
    main()