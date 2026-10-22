import numpy as np
import os
import sys
import json
import time
import random
import argparse
import tensorflow as tf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment import FirewallEnvironment
from src.dqn_agent import DQNAgent
from src.sanity_check import run_sanity_checks

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0,
                     help='Training RNG seed (Python/NumPy/TensorFlow). '
                          'Dataset is unaffected -- preprocess.py uses a '
                          'fixed random_state=42, so train/test data is '
                          'identical across seeds.')
parser.add_argument('--algo', choices=['dqn', 'ddqn'], default='dqn',
                     help='dqn = standard DQN target (max_a Q_target). '
                          'ddqn = Double DQN (select via online net, '
                          'evaluate via target net) -- reduces overestimation bias.')
parser.add_argument('--no-smote', action='store_true',
                     help='Ablation: skip SMOTE, train on naturally imbalanced classes.')
parser.add_argument('--no-target-network', action='store_true',
                     help='Ablation: use the online network itself for target '
                          'computation instead of a separately-synced target network.')
parser.add_argument('--replay-size', type=int, default=50000,
                     help='Ablation: replay buffer capacity (paper/baseline default: 50000).')
parser.add_argument('--dataset', choices=['cicids2017', 'ddos2019', 'cicids2018', 'unsw'], default='cicids2017',
                     help='Which dataset/preprocessing pipeline to train on. '
                          'ddos2019 uses src/preprocess_ddos2019.py, cicids2018 '
                          'uses src/preprocess_cicids2018.py, unsw uses '
                          'src/preprocess_unsw.py -- edit DATA_FOLDER at the '
                          'top of the relevant file first.')
parser.add_argument('--holdout-class', type=str, default=None,
                     help='Leave-one-class-out zero-day proxy experiment: '
                          'completely exclude this attack class (exact label '
                          'string, e.g. "SQL Injection") from training, keeping '
                          'it only in the test set, to measure whether the '
                          'model flags a genuinely unseen attack category as '
                          'malicious. Only supported for --dataset cicids2018 '
                          'currently.')
parser.add_argument('--fp-penalty', type=float, default=-0.5,
                     help='Reward for a false positive (benign wrongly blocked). '
                          'Default -0.5 matches Table I. More negative = more '
                          'conservative about blocking (targets high FPR).')
args = parser.parse_args()

SEED               = args.seed
DOUBLE_DQN         = (args.algo == 'ddqn')
USE_SMOTE          = not args.no_smote
USE_TARGET_NETWORK = not args.no_target_network
REPLAY_SIZE        = args.replay_size
DATASET            = args.dataset
HOLDOUT_CLASS      = args.holdout_class
FP_PENALTY         = args.fp_penalty

if HOLDOUT_CLASS is not None and DATASET != 'cicids2018':
    raise ValueError("--holdout-class is currently only supported for --dataset cicids2018")

if DATASET == 'ddos2019':
    from src.preprocess_ddos2019 import preprocess
elif DATASET == 'cicids2018':
    from src.preprocess_cicids2018 import preprocess
elif DATASET == 'unsw':
    from src.preprocess_unsw import preprocess
else:
    from src.preprocess import preprocess

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

EPISODES   = 500      # matches paper Table II
MAX_STEPS  = 2000

# Self-describing filename: only non-default settings are tagged, so the
# plain CICIDS2017 baseline run stays as dqn_firewall_seed0.keras / seed1
# / etc, while any dataset change, ablation, or algorithm change is
# unambiguous from the filename alone.
_tags = []
if DATASET != 'cicids2017':    _tags.append(DATASET)
if DOUBLE_DQN:                 _tags.append("ddqn")
if not USE_SMOTE:              _tags.append("nosmote")
if not USE_TARGET_NETWORK:     _tags.append("notarget")
if REPLAY_SIZE != 50000:       _tags.append(f"buf{REPLAY_SIZE}")
if HOLDOUT_CLASS:              _tags.append("holdout_" + HOLDOUT_CLASS.replace(" ", "").replace("-", ""))
if FP_PENALTY != -0.5:         _tags.append(f"fppen{FP_PENALTY}")
_suffix = ("_" + "_".join(_tags)) if _tags else ""

SAVE_PATH  = f"models/dqn_firewall{_suffix}_seed{SEED}.keras"
CHECKPOINT = f"models/checkpoint{_suffix}_seed{SEED}.json"
# NOTE: epsilon decay and target-network update cadence are step-counted
# inside DQNAgent per Table II (epsilon: linear over 200,000 steps; target
# update: every 500 steps) -- not configured here.

def save_checkpoint(episode, best_reward, train_step_count):
    os.makedirs("models", exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump({
            "episode"          : episode,
            "best_reward"      : best_reward,
            "train_step_count" : train_step_count,
            "seed"             : SEED
        }, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        try:
            with open(CHECKPOINT) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # A checkpoint file that exists but is empty/corrupted (e.g. the
            # previous run was interrupted mid-write) should not crash
            # training -- treat it as "no valid checkpoint" and start fresh.
            print(f"WARNING: checkpoint file {CHECKPOINT} is unreadable ({e}). "
                  f"Starting fresh instead of resuming.")
            return None
    return None

def train():
    print("=" * 50)
    print("  ADAPTIVE FIREWALL — DQN TRAINING")
    print(f"  Dataset           : {DATASET}")
    if HOLDOUT_CLASS:
        print(f"  Holdout class     : {HOLDOUT_CLASS} (excluded from training)")
    if FP_PENALTY != -0.5:
        print(f"  FP penalty        : {FP_PENALTY} (default -0.5)")
    print(f"  Seed              : {SEED}")
    print(f"  Algorithm         : {'Double DQN' if DOUBLE_DQN else 'Standard DQN'}")
    print(f"  SMOTE             : {'ON' if USE_SMOTE else 'OFF (ablation)'}")
    print(f"  Target network    : {'ON' if USE_TARGET_NETWORK else 'OFF (ablation)'}")
    print(f"  Replay buffer size: {REPLAY_SIZE:,}")
    print(f"  Save path         : {SAVE_PATH}")
    print("=" * 50)

    X_train, X_test, y_train, y_test, scaler, le = preprocess(use_smote=USE_SMOTE, holdout_class=HOLDOUT_CLASS)
    run_sanity_checks(X_train, X_test, y_train, y_test, le)

    env   = FirewallEnvironment(X_train, y_train, fp_penalty=FP_PENALTY)
    agent = DQNAgent(
        state_size         = env.get_state_size(),
        action_size        = env.get_action_size(),
        memory_size        = REPLAY_SIZE,
        use_target_network = USE_TARGET_NETWORK,
        double_dqn         = DOUBLE_DQN
    )

    start_episode = 1
    best_reward   = -np.inf

    checkpoint = load_checkpoint()
    if checkpoint and os.path.exists(SAVE_PATH):
        start_episode          = checkpoint["episode"] + 1
        best_reward            = checkpoint["best_reward"]
        agent.train_step_count = checkpoint["train_step_count"]
        agent._update_epsilon()
        agent.load(SAVE_PATH)
        print(f"Resuming from episode {start_episode}")
        print(f"Best reward so far : {best_reward:.1f}")
        print(f"Epsilon            : {agent.epsilon:.4f}")
        print(f"Train steps so far : {agent.train_step_count:,}")
    else:
        start_episode = 1
        best_reward   = -np.inf
        print("Starting fresh training (500 episodes, paper-matched schedule)...")

    print("=" * 50)
    t_start = time.time()

    for episode in range(start_episode, EPISODES + 1):
        ep_t0        = time.time()
        state        = env.reset()
        total_reward = 0
        steps        = 0
        correct      = 0
        step_count   = 0

        while not env.done and step_count < MAX_STEPS:
            action                   = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()  # epsilon decay + target sync now happen inside replay(), step-counted per Table II
            total_reward += reward
            steps        += 1
            step_count   += 1
            if reward > 0:
                correct += 1
            state = next_state

        accuracy = (correct / steps) * 100 if steps > 0 else 0
        ep_time  = time.time() - ep_t0

        print(
            f"Episode {episode:>3}/{EPISODES} | "
            f"Steps: {steps:>5,} | "
            f"Reward: {total_reward:>8.1f} | "
            f"Accuracy: {accuracy:>6.2f}% | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"TrainSteps: {agent.train_step_count:>7,} | "
            f"EpTime: {ep_time:>5.1f}s"
        )

        if total_reward > best_reward:
            best_reward = total_reward

        agent.save(SAVE_PATH)
        save_checkpoint(episode, best_reward, agent.train_step_count)

    total_time = time.time() - t_start
    print("\nTraining complete!")
    print(f"Best reward       : {best_reward:.1f}")
    print(f"Total train steps : {agent.train_step_count:,}")
    print(f"Total wall time   : {total_time/60:.1f} min")
    print(f"Model saved       : {SAVE_PATH}")

if __name__ == "__main__":
    train()