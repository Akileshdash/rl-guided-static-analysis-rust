"""
Changes
1.State Input : 8 to 24 normalized features 
2. Reward Function:
    +15 for correct, -15 for incorrect, with bonus for confident predictions.
    Encourages both accuracy and confidence.
3. Network Architecture :
   24 -> 32 -> 16 -> 2 MLP (deeper than before).
   ReLU activations, Xavier initialization, momentum optimizer (β = 0.9).
   Learns Q-values ( Q(s, a) ).
4. Training Strategy:
    200 epochs, batch size 64.
    Replay buffer = 5000 experiences.
    v = 0.99 (longer-term reward), LR = 0.0005, ε-decay = 0.996.
    Gradual shift from exploration → exploitation.
5. Learning Loop:
    For each sample: extract features → pick action (ε-greedy) → get reward → store → train from replay.
    Update Q-network using TD target

"""

import json
import random
import numpy as np
from time import sleep

# -------------------------------
# Feature extraction with normalization
# -------------------------------
def extract_features(sample):
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
    ptr_count = len([_ for _ in ["ptr", "pointer", "raw"] if _ in code])
    transmut_count = code.lower().count("transmute")
    async_count = len([_ for _ in ["async", "await"] if _ in code])
    mut_count = code.count("mut ")
    ref_count = code.count("&")
    impl_count = code.lower().count("impl")
    trait_count = code.lower().count("trait")
    where_count = code.lower().count("where")
    lifetime_count = len([_ for _ in code.split() if _.startswith("'")])
    generic_count = len([_ for _ in code.split() if "<" in _ and ">" in _])
    
    # Contextual features
    has_from_raw = int("from_raw" in code)
    has_as_ptr = int("as_ptr" in code)
    has_deref = int("*" in code or "deref" in code)
    has_cast = int(" as " in code)
    has_option = int(any(x in code for x in ["Option", "Some", "None"]))
    has_result = int(any(x in code for x in ["Result", "Ok", "Err"]))
    
    features = [
        level_map.get(sample.get("level"), 0.5),
        analyzer_map.get(sample.get("analyzer"), 1.0),
        min(start_line / 1000, 1),
        min(line_span / 50, 1),
        min(col_span / 100, 1),
        min(len(code) / 500, 1),
        min(unsafe_count / 5, 1),
        min(drop_count / 3, 1),
        min(ptr_count / 5, 1),
        min(transmut_count / 2, 1),
        min(async_count / 3, 1),
        min(mut_count / 10, 1),
        min(ref_count / 15, 1),
        min(impl_count / 3, 1),
        min(trait_count / 2, 1),
        min(where_count / 3, 1),
        min(lifetime_count / 5, 1),
        min(generic_count / 5, 1),
        has_from_raw,
        has_as_ptr,
        has_deref,
        has_cast,
        has_option,
        has_result
    ]
    return np.array(features, dtype=float)


# -------------------------------
# Improved DQN Agent
# -------------------------------
class ImprovedDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.996
        self.lr = 0.0005
        self.momentum = 0.9

        # Network layers
        self.weights1 = self.xavier_init(state_size, 32)
        self.weights2 = self.xavier_init(32, 16)
        self.weights3 = self.xavier_init(16, action_size)
        self.bias1 = np.zeros(32)
        self.bias2 = np.zeros(16)
        self.bias3 = np.zeros(action_size)

        # Momentum velocities
        self.v1 = np.zeros_like(self.weights1)
        self.v2 = np.zeros_like(self.weights2)
        self.v3 = np.zeros_like(self.weights3)
        self.vb1 = np.zeros_like(self.bias1)
        self.vb2 = np.zeros_like(self.bias2)
        self.vb3 = np.zeros_like(self.bias3)

    def xavier_init(self, in_dim, out_dim):
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        return np.random.uniform(-scale, scale, (in_dim, out_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, state):
        self.h1 = self.relu(np.dot(state, self.weights1) + self.bias1)
        self.h2 = self.relu(np.dot(self.h1, self.weights2) + self.bias2)
        self.output = np.dot(self.h2, self.weights3) + self.bias3
        return self.output

    def act(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.forward(state)
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 5000:
            self.memory.pop(0)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0

        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.forward(next_state))
            current_q = self.forward(state)[action]
            loss = (target - current_q) ** 2
            total_loss += loss

            # Output layer
            error = 2 * (current_q - target)
            grad3 = np.outer(self.h2, error)
            self.v3 = self.momentum * self.v3 - self.lr * grad3
            self.weights3 += self.v3
            self.vb3 = self.momentum * self.vb3 - self.lr * error
            self.bias3 += self.vb3

            # Hidden2 layer
            h2_error = error * self.weights3[action, :] * self.relu_derivative(self.h2)
            grad2 = np.outer(self.h1, h2_error)
            self.v2 = self.momentum * self.v2 - self.lr * grad2
            self.weights2 += self.v2
            self.vb2 = self.momentum * self.vb2 - self.lr * h2_error
            self.bias2 += self.vb2

            # Hidden1 layer
            h1_error = np.dot(h2_error, self.weights2.T) * self.relu_derivative(self.h1)
            grad1 = np.outer(state, h1_error)
            self.v1 = self.momentum * self.v1 - self.lr * grad1
            self.weights1 += self.v1
            self.vb1 = self.momentum * self.vb1 - self.lr * h1_error
            self.bias1 += self.vb1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / batch_size


# -------------------------------
# Training loop
# -------------------------------
def train_agent(ground_truth, training_data, episodes=200, batch_size=64):
    # Map filenames to ground truth
    gt_map = {gt["file"]: gt["false_positive"] for gt in ground_truth}
    agent = ImprovedDQNAgent(state_size=24, action_size=2)
    history = []

    for ep in range(episodes):
        total_reward = 0
        correct = 0
        total_loss = 0
        random.shuffle(training_data)

        for sample in training_data:
            state = extract_features(sample)
            true_label = gt_map.get(sample["filename"], 0)
            action = agent.act(state, training=True)
            reward = 15 if action == true_label else -15
            if action == true_label and ep > 50:
                reward += 5
            total_reward += reward
            correct += action == true_label

            agent.remember(state, action, reward, state, True)
            if len(agent.memory) >= batch_size:
                total_loss += agent.replay(batch_size)

        acc = (correct / len(training_data)) * 100
        avg_loss = total_loss / len(training_data)
        history.append({"epoch": ep+1, "accuracy": acc, "loss": avg_loss, "reward": total_reward})
        print(f"Epoch {ep+1:3d} | Accuracy: {acc:.2f}% | Loss: {avg_loss:.4f} | Reward: {total_reward}")
        sleep(0.01)  # Simulate async

    # Final predictions
    predictions = []
    for sample in training_data:
        state = extract_features(sample)
        pred = agent.act(state, training=False)
        predictions.append({
            "filename": sample["filename"],
            "prediction": pred,
            "actual": gt_map.get(sample["filename"], 0),
            "correct": pred == gt_map.get(sample["filename"], 0)
        })

    return predictions, history


if __name__ == "__main__":
    # Load JSON files
    with open("ground_truth.json") as f:
        ground_truth = json.load(f)
    with open("training_data.json") as f:
        training_data = json.load(f)

    predictions, history = train_agent(ground_truth, training_data)
