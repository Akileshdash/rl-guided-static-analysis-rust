# tabular Q-learning concept,
# extended with a neural network approximator (a minimalist DQN) to a classification-style task, 
# learning via reinforcement (reward)

import json
import random
import numpy as np


# -------------------------------
# Feature extraction
# -------------------------------
def extract_features(sample):
    level_map = {'Info': 0, 'Warning': 1, 'Error': 2}
    analyzer_map = {
        'UnsafeDataflow': 0,
        'UnsafeDestructor': 1,
        'SendSyncVariance': 2,
        'UnsafeTransmute': 3
    }

    level = level_map.get(sample.get("level"), 0)
    analyzer = analyzer_map.get(sample.get("analyzer"), 0)
    start_line = sample.get("start_line", 0)
    end_line = sample.get("end_line", 0)
    code = sample.get("code_snippet", "")

    return np.array([
        level,
        analyzer,
        start_line,
        end_line,
        end_line - start_line,
        len(code),
        code.count("unsafe"),
        code.lower().count("drop")
    ], dtype=float)


# -------------------------------
# Simple DQN Agent
# -------------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 0.001

        # Initialize weights
        self.weights1 = np.random.uniform(-0.5, 0.5, (state_size, 16))
        self.weights2 = np.random.uniform(-0.5, 0.5, (16, action_size))
        self.bias1 = np.zeros(16)
        self.bias2 = np.zeros(action_size)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, state):
        hidden = self.relu(np.dot(state, self.weights1) + self.bias1)
        output = np.dot(hidden, self.weights2) + self.bias2
        return hidden, output

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        _, output = self.forward(state)
        return np.argmax(output)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0

        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0

        for state, action, reward, next_state, done in minibatch:
            _, next_output = self.forward(next_state)
            target = reward if done else reward + self.gamma * np.max(next_output)

            hidden, output = self.forward(state)
            current_q = output[action]
            loss = (target - current_q) ** 2
            total_loss += loss

            # Simple gradient update
            error = target - current_q
            self.weights2[:, action] += self.lr * error * hidden
            self.bias2[action] += self.lr * error

            grad_hidden = (hidden > 0).astype(float)
            for i in range(self.state_size):
                self.weights1[i, :] += self.lr * error * 0.1 * grad_hidden * state[i]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / batch_size


# -------------------------------
# Training pipeline
# -------------------------------
def train_agent(ground_truth_path, training_data_paths, episodes=100, batch_size=32):
    # Load ground truth
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
    ground_truth_map = {item["file"]: item["false_positive"] for item in ground_truth}

    # Load training data
    training_data = []
    for path in training_data_paths:
        with open(path, "r") as f:
            data = json.load(f)
            data["filename"] = path.split("/")[-1]
            training_data.append(data)

    agent = DQNAgent(state_size=8, action_size=2)

    for ep in range(episodes):
        total_reward = 0
        correct = 0
        random.shuffle(training_data)

        for sample in training_data:
            state = extract_features(sample)
            truth = ground_truth_map.get(sample["filename"], 0)
            action = agent.act(state)
            reward = 10 if action == truth else -10

            agent.remember(state, action, reward, state, True)
            total_reward += reward
            if action == truth:
                correct += 1

        loss = agent.replay(batch_size)
        accuracy = correct / len(training_data) * 100

        print(f"Epoch {ep+1:3d} | Accuracy: {accuracy:.2f}% | Loss: {loss:.4f} | Reward: {total_reward}")

    # Final predictions
    predictions = []
    for sample in training_data:
        state = extract_features(sample)
        pred = agent.act(state)
        truth = ground_truth_map.get(sample["filename"], 0)
        predictions.append({
            "filename": sample["filename"],
            "prediction": pred,
            "actual": truth,
            "correct": pred == truth
        })

    correct_preds = sum(p["correct"] for p in predictions)
    print(f"\n Final Accuracy: {correct_preds / len(predictions) * 100:.2f}% ({correct_preds}/{len(predictions)})")

    return predictions

if __name__ == "__main__":
    preds = train_agent(
        ground_truth_path="ground_truth.json",
        training_data_paths=[],
        episodes=20
    )
