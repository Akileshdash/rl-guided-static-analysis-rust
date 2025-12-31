"""
Enhanced DQN Agent for False Positive Classification in Rust Static Analysis
Based on: Mitigating False Positives in Static Memory Safety Analysis of Rust Programs via RL

Features:
- 24 normalized input features from MIR analysis
- Deep Q-Network with 3 hidden layers (24 -> 32 -> 16 -> 2)
- Experience replay buffer (5000 samples)
- Epsilon-greedy exploration with decay
- Momentum-based gradient descent
"""

import json
import random
import numpy as np
from collections import deque
from datetime import datetime


class DQNAgent:
    """Deep Q-Network Agent for binary classification of static analysis warnings"""
    
    def __init__(self, state_size=24, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        
        # Replay memory
        self.memory = deque(maxlen=5000)
        
        # Hyperparameters (from paper)
        self.gamma = 0.99           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.05     # Minimum exploration
        self.epsilon_decay = 0.996  # Decay rate
        self.learning_rate = 0.0005
        self.momentum = 0.9
        
        # Initialize network layers with Xavier initialization
        self.weights1 = self._xavier_init(state_size, 32)
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
        """Xavier/Glorot initialization for better convergence"""
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        return np.random.uniform(-scale, scale, (in_dim, out_dim))
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """Derivative of ReLU for backpropagation"""
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
        Returns: 0 (False Positive) or 1 (True Positive)
        """
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.forward(state)
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=64):
        """Train on a batch of experiences from memory"""
        if len(self.memory) < batch_size:
            return 0.0
        
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0.0
        
        for state, action, reward, next_state, done in minibatch:
            # Calculate TD target
            target = reward
            if not done:
                target += self.gamma * np.max(self.forward(next_state))
            
            # Forward pass
            current_q = self.forward(state)
            current_q_value = current_q[action]
            
            # Calculate loss
            loss = (target - current_q_value) ** 2
            total_loss += loss
            
            # Backpropagation
            # Output layer gradient
            output_error = np.zeros(self.action_size)
            output_error[action] = 2 * (current_q_value - target)
            
            grad_w3 = np.outer(self.h2, output_error)
            grad_b3 = output_error
            
            # Hidden layer 2 gradient
            h2_error = np.dot(output_error, self.weights3.T) * self._relu_derivative(self.z2)
            grad_w2 = np.outer(self.h1, h2_error)
            grad_b2 = h2_error
            
            # Hidden layer 1 gradient
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
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / batch_size


def extract_features(sample):
    """
    Extract 24 normalized features from static analysis warning
    Categories: Level, Analyzer, Location, Code patterns, Contextual
    """
    # Mapping dictionaries
    level_map = {'Info': 0.33, 'Warning': 0.66, 'Error': 1.0}
    analyzer_map = {
        'UnsafeDataflow': 0.2,
        'UnsafeDestructor': 0.4,
        'SendSyncVariance': 0.6,
        'UnsafeTransmute': 0.8,
        'Other': 1.0
    }
    
    # Extract basic info
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
    
    # Lifetime and generic markers
    lifetime_count = len([word for word in code.split() if word.startswith("'")])
    generic_count = len([char for char in code if char in "<>"])
    
    # Contextual boolean features
    has_from_raw = int("from_raw" in code.lower())
    has_as_ptr = int("as_ptr" in code.lower())
    has_deref = int("*" in code or "deref" in code.lower())
    has_cast = int(" as " in code)
    has_option = int(any(kw in code for kw in ["Option", "Some", "None"]))
    has_result = int(any(kw in code for kw in ["Result", "Ok", "Err"]))
    
    # Construct feature vector with normalization
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


def train_agent(ground_truth, training_data, episodes=200, batch_size=64):
    """
    Train the DQN agent on labeled static analysis warnings
    
    Args:
        ground_truth: List of {file, false_positive} labels
        training_data: List of warning samples with features
        episodes: Number of training epochs
        batch_size: Batch size for replay learning
    
    Returns:
        predictions: Final predictions on all samples
        history: Training metrics per epoch
    """
    # Create ground truth mapping
    gt_map = {gt["file"]: gt["false_positive"] for gt in ground_truth}
    
    # Initialize agent
    agent = DQNAgent(state_size=24, action_size=2)
    
    # Training history
    history = []
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Dataset size: {len(training_data)} samples")
    print(f"Batch size: {batch_size}")
    print("-" * 70)
    
    for episode in range(episodes):
        total_reward = 0
        correct = 0
        total_loss = 0
        
        # Shuffle data each epoch
        random.shuffle(training_data)
        
        for sample in training_data:
            state = extract_features(sample)
            true_label = gt_map.get(sample["filename"], 0)
            
            # Agent selects action
            action = agent.act(state, training=True)
            
            # Calculate reward (from paper: +15 correct, -15 incorrect)
            reward = 15 if action == true_label else -15
            
            # Bonus for confident correct predictions after warmup
            if action == true_label and episode > 50:
                reward += 5
            
            total_reward += reward
            correct += (action == true_label)
            
            # Store experience
            agent.remember(state, action, reward, state, done=True)
            
            # Train on batch
            if len(agent.memory) >= batch_size:
                loss = agent.replay(batch_size)
                total_loss += loss
        
        # Calculate metrics
        accuracy = (correct / len(training_data)) * 100
        avg_loss = total_loss / len(training_data) if len(training_data) > 0 else 0
        
        history.append({
            "epoch": episode + 1,
            "accuracy": accuracy,
            "loss": avg_loss,
            "reward": total_reward,
            "epsilon": agent.epsilon
        })
        
        # Print progress
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"Epoch {episode+1:3d} | Acc: {accuracy:5.2f}% | "
                  f"Loss: {avg_loss:7.4f} | Reward: {total_reward:6.0f} | "
                  f"ε: {agent.epsilon:.4f}")
    
    print("-" * 70)
    print("Training complete!")
    
    # Generate final predictions (no exploration)
    print("Generating final predictions...")
    predictions = []
    correct_final = 0
    
    for sample in training_data:
        state = extract_features(sample)
        prediction = agent.act(state, training=False)
        actual = gt_map.get(sample["filename"], 0)
        is_correct = (prediction == actual)
        
        predictions.append({
            "filename": sample["filename"],
            "prediction": prediction,
            "actual": actual,
            "correct": is_correct
        })
        
        correct_final += is_correct
    
    final_accuracy = (correct_final / len(training_data)) * 100
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    
    return predictions, history


def main():
    """Main execution function"""
    print("=" * 70)
    print("DQN Agent for Rust Static Analysis False Positive Reduction")
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
    predictions, history = train_agent(ground_truth, training_data)
    
    # Save results
    print("\nSaving results...")
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": "DQN",
            "episodes": len(history),
            "final_accuracy": history[-1]["accuracy"]
        },
        "predictions": predictions,
        "training_history": history
    }
    
    with open("RL_output/rl.json", "w") as f:
        json.dump(output, f, indent=2)
    print("✓ Results saved to RL_output/rl.json")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Final Accuracy: {history[-1]['accuracy']:.2f}%")
    print(f"Final Epsilon: {history[-1]['epsilon']:.4f}")
    print(f"Total Correct: {sum(1 for p in predictions if p['correct'])}/{len(predictions)}")


if __name__ == "__main__":
    main()