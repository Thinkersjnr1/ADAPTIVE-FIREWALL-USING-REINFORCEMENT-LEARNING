import numpy as np
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import preprocess
from src.environment import FirewallEnvironment
from src.dqn_agent import DQNAgent

EPISODES      = 50
MAX_STEPS     = 2000
TARGET_UPDATE = 5
SAVE_PATH     = "models/dqn_firewall.keras"
CHECKPOINT    = "models/checkpoint.json"

def save_checkpoint(episode, best_reward, epsilon):
    os.makedirs("models", exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump({
            "episode"     : episode,
            "best_reward" : best_reward,
            "epsilon"     : epsilon
        }, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return None

def train():
    print("=" * 50)
    print("  ADAPTIVE FIREWALL — DQN TRAINING")
    print("=" * 50)

    X_train, X_test, y_train, y_test, scaler, le = preprocess()

    env   = FirewallEnvironment(X_train, y_train)
    agent = DQNAgent(
        state_size  = env.get_state_size(),
        action_size = env.get_action_size()
    )

    start_episode = 1
    best_reward   = -np.inf

    checkpoint = load_checkpoint()
    if checkpoint and os.path.exists(SAVE_PATH):
        start_episode = checkpoint["episode"] + 1
        best_reward   = checkpoint["best_reward"]
        agent.epsilon = checkpoint["epsilon"]
        agent.load(SAVE_PATH)
        print(f"Resuming from episode {start_episode}")
        print(f"Best reward so far : {best_reward:.1f}")
        print(f"Epsilon            : {agent.epsilon:.4f}")
    else:
        start_episode = 1
        best_reward   = -np.inf
        print("Starting fresh training...")

    print("=" * 50)

    for episode in range(start_episode, EPISODES + 1):
        state        = env.reset()
        total_reward = 0
        steps        = 0
        correct      = 0
        step_count   = 0

        while not env.done and step_count < MAX_STEPS:
            action                   = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            total_reward += reward
            steps        += 1
            step_count   += 1
            if reward > 0:
                correct += 1
            state = next_state

        accuracy = (correct / steps) * 100 if steps > 0 else 0

        if episode % TARGET_UPDATE == 0:
            agent.update_target()
            print(f"  >> Target network updated")

        print(
            f"Episode {episode:>3}/{EPISODES} | "
            f"Steps: {steps:>5,} | "
            f"Reward: {total_reward:>8.1f} | "
            f"Accuracy: {accuracy:>6.2f}% | "
            f"Epsilon: {agent.epsilon:.4f}"
        )

        if total_reward > best_reward:
            best_reward = total_reward

        agent.save(SAVE_PATH)
        save_checkpoint(episode, best_reward, agent.epsilon)

    print("\nTraining complete!")
    print(f"Best reward : {best_reward:.1f}")
    print(f"Model saved : {SAVE_PATH}")

if __name__ == "__main__":
    train()