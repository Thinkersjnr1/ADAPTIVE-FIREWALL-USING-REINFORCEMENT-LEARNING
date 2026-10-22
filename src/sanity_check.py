"""
Sanity checks run automatically after preprocessing, before any training
begins. Exists specifically because of a real bug: UNSW-NB15's "Analysis"
attack category sorted alphabetically before "BENIGN", silently inverting
the label==0-means-benign assumption every reward/decision check in this
codebase relies on -- and that wasn't caught until AFTER a full ~12-hour
training run had already completed on inverted labels. These checks fail
LOUDLY, in seconds, before training starts -- instead of silently
producing a bad model many hours later.
"""
import numpy as np


def run_sanity_checks(X_train, X_test, y_train, y_test, label_encoder):
    errors = []

    # 1. The exact bug that caused all this: BENIGN must be encoded as 0,
    #    since environment.py and dashboard.py both hard-code
    #    `label == 0` to mean "this is benign traffic."
    classes = list(label_encoder.classes_)
    if len(classes) == 0 or classes[0] != "BENIGN":
        errors.append(
            f"label index 0 is '{classes[0] if classes else '(none)'}', not "
            f"'BENIGN'. Every reward/decision check in this codebase assumes "
            f"label==0 means benign -- training on this would silently invert "
            f"the reward signal. Full class order: {classes}"
        )

    # 2. Need at least one benign and one attack example in both splits,
    #    or the binary distinction the reward function is built around
    #    doesn't actually exist in the data.
    for name, y in [("y_train", y_train), ("y_test", y_test)]:
        if int(np.sum(y == 0)) == 0:
            errors.append(f"{name} has zero benign (label==0) samples.")
        if int(np.sum(y != 0)) == 0:
            errors.append(f"{name} has zero attack samples.")

    # 3. Feature counts must match between train and test, or the model's
    #    fixed input layer will silently misalign features.
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(
            f"X_train has {X_train.shape[1]} features but X_test has "
            f"{X_test.shape[1]} -- these must match."
        )

    # 4. No NaN/inf should ever survive preprocessing into the arrays a
    #    model actually trains on.
    for name, X in [("X_train", X_train), ("X_test", X_test)]:
        if not np.all(np.isfinite(X)):
            errors.append(f"{name} contains NaN or infinite values.")

    if errors:
        print("\n" + "=" * 55)
        print("  SANITY CHECK FAILED -- refusing to start training")
        print("=" * 55)
        for e in errors:
            print(f"  - {e}")
        print("=" * 55)
        raise RuntimeError(
            f"{len(errors)} sanity check(s) failed (see above). "
            f"Training was NOT started -- fix the preprocessing pipeline first."
        )

    print(f"[Sanity check] OK -- BENIGN=0, {len(classes)} classes, "
          f"{X_train.shape[0]:,} train / {X_test.shape[0]:,} test rows, "
          f"{X_train.shape[1]} features, no NaN/inf.")