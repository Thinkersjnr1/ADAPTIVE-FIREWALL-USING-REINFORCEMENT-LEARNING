"""Model evaluation, comparison and robustness testing.

Everything here is plain numpy and works on the two-column Q matrix the dashboard
already uses (column 0 = Q(ALLOW), column 1 = Q(BLOCK)). Class index 0 is treated
as benign traffic, matching the rest of the app."""
import numpy as np


def _pct(n, d, default=0.0):
    return round(float(n) / d * 100, 2) if d else default


def predictions(q):
    return np.argmax(q, axis=1)


def scores(q):
    """One number per flow: how strongly the model prefers BLOCK over ALLOW."""
    return q[:, 1] - q[:, 0]


def binary_metrics(y_bin, pred):
    y_bin = np.asarray(y_bin).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y_bin == 1)).sum())
    fp = int(((pred == 1) & (y_bin == 0)).sum())
    fn = int(((pred == 0) & (y_bin == 1)).sum())
    tn = int(((pred == 0) & (y_bin == 0)).sum())
    precision = _pct(tp, tp + fp)
    recall = _pct(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": int(len(y_bin)),
        "accuracy": _pct(tp + tn, len(y_bin)), "precision": precision, "recall": recall,
        "f1": f1, "fpr": _pct(fp, fp + tn),
    }


def per_class(y_test, classes, pred):
    """For every traffic class: how many flows, how many were blocked, and how often the
    decision was right (blocked for attacks, allowed for benign)."""
    y_test = np.asarray(y_test).astype(int)
    pred = np.asarray(pred).astype(int)
    rows = []
    for ci, name in enumerate(classes):
        mask = y_test == ci
        n = int(mask.sum())
        if not n:
            continue
        blocked = int(pred[mask].sum())
        correct = (n - blocked) if ci == 0 else blocked
        rows.append({"label": str(name), "benign": ci == 0, "support": n, "blocked": blocked,
                     "allowed": n - blocked, "correct": correct, "rate": _pct(correct, n)})
    return rows


def _downsample(arrs, n):
    size = len(arrs[0])
    if size <= n:
        return [a.tolist() for a in arrs]
    idx = np.unique(np.linspace(0, size - 1, n).astype(int))
    return [a[idx].tolist() for a in arrs]


def curves(y_bin, score, points=120):
    """ROC and precision-recall curves with AUC / average precision. Returns None
    values when the evaluation set has only one class."""
    y = np.asarray(y_bin).astype(int)
    s = np.asarray(score, dtype=float)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return {"auc": None, "ap": None, "roc": None, "pr": None}
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tps, fps = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[np.where(np.diff(ss))[0], len(ys) - 1]  # last index of each distinct score
    tpr = np.r_[0.0, tps[last] / pos]
    fpr = np.r_[0.0, fps[last] / neg]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    prec = tps[last] / (tps[last] + fps[last])
    rec = tps[last] / pos
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    roc = _downsample([fpr, tpr], points)
    pr = _downsample([rec, prec], points)
    return {"auc": round(auc, 5), "ap": round(ap, 5),
            "roc": {"fpr": roc[0], "tpr": roc[1]}, "pr": {"recall": pr[0], "precision": pr[1]}}


def full_report(q, y_test, classes):
    y_test = np.asarray(y_test).astype(int)
    y_bin = (y_test != 0).astype(int)
    pred = predictions(q)
    out = {"metrics": binary_metrics(y_bin, pred), "classes": per_class(y_test, classes, pred)}
    out.update(curves(y_bin, scores(q)))
    return out


def compare(qa, qb, y_test, classes):
    """Side-by-side comparison of two models on the same evaluation set."""
    y_test = np.asarray(y_test).astype(int)
    y_bin = (y_test != 0).astype(int)
    pa, pb = predictions(qa), predictions(qb)
    ra, rb = full_report(qa, y_test, classes), full_report(qb, y_test, classes)
    ok_a, ok_b = pa == y_bin, pb == y_bin
    by_label_b = {c["label"]: c for c in rb["classes"]}
    rows = []
    for c in ra["classes"]:
        other = by_label_b.get(c["label"], {})
        rows.append({"label": c["label"], "support": c["support"], "a": c["rate"], "b": other.get("rate", 0.0)})
    return {
        "a": {k: ra[k] for k in ("metrics", "auc", "ap")},
        "b": {k: rb[k] for k in ("metrics", "auc", "ap")},
        "agreement": _pct((pa == pb).sum(), len(pa)),
        "only_a_correct": int((ok_a & ~ok_b).sum()),
        "only_b_correct": int((~ok_a & ok_b).sum()),
        "both_wrong": int((~ok_a & ~ok_b).sum()),
        "classes": rows,
    }


# ---- robustness ---------------------------------------------------------------------------------
EPSILONS = (0.01, 0.02, 0.05, 0.1)


def robustness(qfn, X_test, y_test, grad_fn=None, sample=2000, seed=0, eps_list=EPSILONS):
    """How easily can the model be pushed off its decision?

    * Noise test: random jitter of each feature (inputs are scaled 0-1, so eps=0.05 is
      a 5% nudge). Reported as the share of decisions that flip.
    * Evasion test (needs gradients): a small deliberate nudge in the direction that
      most reduces the model's preference to BLOCK (FGSM). Reported as the share of
      attacks that were blocked before and are allowed after.
    """
    rng = np.random.default_rng(seed)
    y_bin = (np.asarray(y_test) != 0).astype(int)
    base_pred = predictions(qfn(X_test))

    def pick(mask):
        idx = np.where(mask)[0]
        if len(idx) > sample:
            idx = rng.choice(idx, sample, replace=False)
        return idx

    caught = pick((y_bin == 1) & (base_pred == 1))      # attacks the model currently stops
    normal = pick((y_bin == 0) & (base_pred == 0))      # normal flows currently allowed
    result = {"sample_attacks": int(len(caught)), "sample_normal": int(len(normal)),
              "eps": list(eps_list), "noise_attack_flip": [], "noise_normal_flip": [],
              "evasion": None, "evasion_note": None}

    for eps in eps_list:
        for name, idx, want in (("noise_attack_flip", caught, 1), ("noise_normal_flip", normal, 0)):
            if not len(idx):
                result[name].append(None)
                continue
            Xn = np.clip(X_test[idx] + rng.normal(0, eps, X_test[idx].shape), 0, 1)
            result[name].append(_pct((predictions(qfn(Xn)) != want).sum(), len(idx)))

    if grad_fn is None:
        result["evasion_note"] = "Evasion testing needs a differentiable model. It is skipped for this one."
    elif not len(caught):
        result["evasion_note"] = "No correctly blocked attacks were available to test."
    else:
        try:
            Xc = X_test[caught]
            direction = np.sign(grad_fn(Xc))             # direction that raises Q(ALLOW) - Q(BLOCK)
            result["evasion"] = [_pct((predictions(qfn(np.clip(Xc + eps * direction, 0, 1))) == 0).sum(), len(caught))
                                 for eps in eps_list]
        except Exception as exc:  # gradients unavailable in this environment
            result["evasion_note"] = f"Evasion testing could not run: {exc}"
    return result
