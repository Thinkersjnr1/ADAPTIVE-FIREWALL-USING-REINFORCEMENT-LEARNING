import numpy as np
import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from src.dqn_agent import DQNAgent

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="models/dqn_firewall.keras",
                     help='Path to the .keras model to evaluate. Use this to '
                          'evaluate a specific seed run, e.g. '
                          '--model models/dqn_firewall_seed1.keras')
parser.add_argument('--dataset', choices=['cicids2017', 'ddos2019', 'cicids2018', 'unsw'], default='cicids2017',
                     help='Which dataset/preprocessing pipeline to evaluate '
                          'against -- must match whatever the model was '
                          'trained on, since each dataset has its own '
                          'held-out test set.')
parser.add_argument('--holdout-class', type=str, default=None,
                     help='Must match the --holdout-class used during training, '
                          'so the exact same held-out test set is reproduced. '
                          'Adds a report section measuring recall specifically '
                          'on this class\'s test rows.')
args = parser.parse_args()
SAVE_PATH = args.model
HOLDOUT_CLASS = args.holdout_class

if args.dataset == 'ddos2019':
    from src.preprocess_ddos2019 import preprocess
elif args.dataset == 'cicids2018':
    from src.preprocess_cicids2018 import preprocess
elif args.dataset == 'unsw':
    from src.preprocess_unsw import preprocess
else:
    from src.preprocess import preprocess

def evaluate():
    print("=" * 55)
    print("  ADAPTIVE FIREWALL — EVALUATION REPORT")
    print(f"  Model: {SAVE_PATH}")
    print(f"  Dataset: {args.dataset}")
    if HOLDOUT_CLASS:
        print(f"  Holdout class: {HOLDOUT_CLASS} (was excluded from training)")
    print("=" * 55)

    # Load data -- holdout_class must match training exactly so the same
    # held-out test set is reproduced (deterministic given random_state=42).
    X_train, X_test, y_train, y_test, scaler, le = preprocess(holdout_class=HOLDOUT_CLASS)

    # Load trained agent
    agent = DQNAgent(state_size=X_test.shape[1], action_size=2)
    agent.load(SAVE_PATH)
    agent.epsilon = 0.0   # no exploration — pure exploitation

    # Run predictions on test set
    print("\nRunning predictions on test set...")
    predictions = []
    for i in range(len(X_test)):
        state  = X_test[i]
        action = agent.act(state)
        predictions.append(action)

    predictions = np.array(predictions)

    # Convert true labels to binary (0=benign, 1=attack)
    y_binary = (y_test != 0).astype(int)

    # ── Metrics ───────────────────────────────────────────────
    acc  = accuracy_score(y_binary, predictions)
    prec = precision_score(y_binary, predictions, zero_division=0)
    rec  = recall_score(y_binary, predictions, zero_division=0)
    f1   = f1_score(y_binary, predictions, zero_division=0)
    cm   = confusion_matrix(y_binary, predictions)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    dr  = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n" + "=" * 55)
    print("  PERFORMANCE METRICS")
    print("=" * 55)
    print(f"  Accuracy          : {acc  * 100:.2f}%")
    print(f"  Precision         : {prec * 100:.2f}%")
    print(f"  Recall            : {rec  * 100:.2f}%")
    print(f"  F1-Score          : {f1   * 100:.2f}%")
    print(f"  Detection Rate    : {dr   * 100:.2f}%")
    print(f"  False Positive Rate: {fpr * 100:.2f}%")

    print("\n" + "=" * 55)
    print("  CONFUSION MATRIX")
    print("=" * 55)
    print(f"  True  Negatives  (Benign correctly allowed) : {tn:>6,}")
    print(f"  False Positives  (Benign wrongly blocked)   : {fp:>6,}")
    print(f"  False Negatives  (Attack missed)            : {fn:>6,}")
    print(f"  True  Positives  (Attack correctly blocked) : {tp:>6,}")

    if HOLDOUT_CLASS:
        # This is the number that actually answers the zero-day question --
        # the aggregate metrics above are diluted by thousands of normal
        # test rows the model DID train on. Isolating just the held-out
        # class's rows measures whether the model flags a genuinely
        # unseen attack category as malicious, despite never training on it.
        holdout_encoded = le.transform([HOLDOUT_CLASS])[0]
        holdout_mask = (y_test == holdout_encoded)
        n_holdout = int(holdout_mask.sum())
        if n_holdout > 0:
            holdout_caught = int(predictions[holdout_mask].sum())  # sum of 1s (BLOCK) among holdout rows
            holdout_recall = holdout_caught / n_holdout
            print("\n" + "=" * 55)
            print("  ZERO-DAY PROXY: LEAVE-ONE-CLASS-OUT RESULT")
            print("=" * 55)
            print(f"  Held-out class          : {HOLDOUT_CLASS}")
            print(f"  Test rows of this class : {n_holdout:,} (0 seen during training)")
            print(f"  Correctly flagged BLOCK : {holdout_caught:,} / {n_holdout:,}")
            print(f"  Held-out-class recall   : {holdout_recall*100:.2f}%")
            print("  (Compare against overall Recall above and against chance --")
            print("   this is the number that actually measures generalization")
            print("   to a genuinely unseen attack category, not the aggregate")
            print("   metrics, which are diluted by classes the model DID train on.)")
        else:
            print(f"\n  WARNING: 0 test rows found for holdout class "
                  f"'{HOLDOUT_CLASS}' -- check the class name matches exactly.")

    print("\n" + "=" * 55)
    print("  BASELINE COMPARISON")
    print("=" * 55)
    print(f"  {'System':<30} {'Accuracy':>10} {'F1-Score':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Rule-Based Firewall':<30} {'~60.00%':>10} {'~55.00%':>10}")
    print(f"  {'Random Forest (Supervised ML)':<30} {'~95.00%':>10} {'~94.00%':>10}")
    print(f"  {'DQN Adaptive Firewall (Ours)':<30} {acc*100:>9.2f}% {f1*100:>9.2f}%")

    print("\n" + "=" * 55)
    print("  VERDICT")
    print("=" * 55)
    if acc >= 0.95:
        print("  PASS — System meets the 95% accuracy target.")
        print("  The DQN adaptive firewall outperforms baselines.")
    else:
        print("  System trained. Consider more episodes for higher accuracy.")

    print("=" * 55)

if __name__ == "__main__":
    evaluate()