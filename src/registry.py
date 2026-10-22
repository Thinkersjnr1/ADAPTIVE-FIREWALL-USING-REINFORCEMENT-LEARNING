"""Model registry: finds every model the dashboard can use, loads them on demand,
and caches each one's predictions and evaluation so switching is instant.

Two kinds of model are supported:
  * "dqn"      any .keras / .h5 file in the models folder (loaded with DQNAgent)
  * "baseline" Random Forest (and XGBoost if installed), trained on the cached
               training split the first time they are used, then saved to disk.

Every loaded model exposes the same thing: a matrix of two values per flow,
Q(ALLOW) and Q(BLOCK). For baselines these are the class probabilities, so the rest
of the dashboard does not need to know which kind of model it is talking to."""
import os
import threading
import time

import numpy as np

from src import evaluation

BASELINES = {
    "baseline_rf": ("Random Forest baseline",
                    "Classic tree ensemble trained on the same data. Trained the first time you use it."),
    "baseline_xgb": ("XGBoost baseline",
                     "Gradient-boosted trees trained on the same data. Trained the first time you use it."),
}


def _humanize(stem):
    return stem.replace("_", " ").replace("-", " ").strip().capitalize()


class ModelRegistry:
    def __init__(self, models_dir, make_agent):
        self.models_dir = models_dir
        self.make_agent = make_agent          # callable(state_size) -> agent with .load(path) and .model
        self.data = None                      # dict: X_train, X_test, y_train, y_test, classes
        self.entries = {}
        self._lock = threading.RLock()

    # ---- setup -------------------------------------------------------------------------------
    def set_data(self, data):
        self.data = data

    def discover(self):
        """Refresh the list from disk. Already-loaded models keep their state."""
        found = {}
        if os.path.isdir(self.models_dir):
            for fn in sorted(os.listdir(self.models_dir)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() in (".keras", ".h5"):
                    found[stem] = {"name": _humanize(stem), "kind": "dqn",
                                   "path": os.path.join(self.models_dir, fn),
                                   "description": "Deep Q-network firewall policy."}
        try:
            import sklearn  # noqa: F401
            found["baseline_rf"] = {"name": BASELINES["baseline_rf"][0], "kind": "baseline",
                                    "path": None, "description": BASELINES["baseline_rf"][1]}
        except ImportError:
            pass
        try:
            import xgboost  # noqa: F401
            found["baseline_xgb"] = {"name": BASELINES["baseline_xgb"][0], "kind": "baseline",
                                     "path": None, "description": BASELINES["baseline_xgb"][1]}
        except ImportError:
            pass
        with self._lock:
            for mid, meta in found.items():
                if mid not in self.entries:
                    self.entries[mid] = {"id": mid, "status": "available", "error": None, "params": None,
                                         "loaded_at": None, "lock": threading.Lock(), "robust": None, **meta}
            for mid in list(self.entries):
                if mid not in found and self.entries[mid]["status"] != "ready":
                    del self.entries[mid]
        return self.list()

    def list(self):
        with self._lock:
            out = []
            for e in self.entries.values():
                ev = e.get("eval") or {}
                m = ev.get("metrics") or {}
                out.append({"id": e["id"], "name": e["name"], "kind": e["kind"], "status": e["status"],
                            "error": e["error"], "description": e["description"], "params": e["params"],
                            "accuracy": m.get("accuracy"), "f1": m.get("f1"), "fpr": m.get("fpr"),
                            "auc": ev.get("auc"), "loaded_at": e["loaded_at"]})
            # dqn models first, then baselines, each alphabetical
            return sorted(out, key=lambda r: (r["kind"] != "dqn", r["name"]))

    def get(self, mid):
        with self._lock:
            return self.entries.get(mid)

    def ready(self, mid):
        e = self.get(mid)
        return e if e and e["status"] == "ready" else None

    # ---- loading ----------------------------------------------------------------------------------
    def load(self, mid):
        """Load a model, blocking until it is ready. Safe to call from several threads:
        the second caller waits for the first and then returns the same entry."""
        e = self.get(mid)
        if e is None:
            raise KeyError(f"Unknown model '{mid}'.")
        if self.data is None:
            raise RuntimeError("The evaluation data isn't ready yet.")
        with e["lock"]:
            if e["status"] == "ready":
                return e
            e["status"], e["error"] = "loading", None
            try:
                t0 = time.time()
                q = self._load_dqn(e) if e["kind"] == "dqn" else self._load_baseline(e)
                e["q_vals"] = q
                e["eval"] = evaluation.full_report(q, self.data["y_test"], self.data["classes"])
                acc = e["eval"]["metrics"]["accuracy"] / 100.0
                e["health"] = {"model_path": e["path"] or e["name"], "params": e["params"],
                               "accuracy": acc, "total_test": int(len(self.data["y_test"]))}
                e["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                e["load_seconds"] = round(time.time() - t0, 1)
                e["status"] = "ready"
            except Exception as exc:
                e["status"], e["error"] = "error", str(exc)[:300]
                raise
        return e

    def load_async(self, mid):
        e = self.get(mid)
        if e and e["status"] in ("available", "error"):
            threading.Thread(target=self._swallow, args=(mid,), daemon=True).start()
        return e

    def _swallow(self, mid):
        try:
            self.load(mid)
        except Exception:
            pass  # the error is stored on the entry and shown in the UI

    def _load_dqn(self, e):
        X = self.data["X_test"]
        agent = self.make_agent(X.shape[1])
        agent.load(e["path"])
        agent.epsilon = 0.0
        model = agent.model
        expected = getattr(model, "input_shape", (None, None))[-1]
        if expected not in (None, X.shape[1]):
            raise ValueError(f"This model expects {expected} input features but the evaluation data has {X.shape[1]}.")

        def qfn(Xq):
            return model.predict(np.asarray(Xq, dtype="float32"), batch_size=1024, verbose=0)

        def grad_fn(Xq):
            """Gradient of Q(ALLOW) - Q(BLOCK) with respect to the inputs."""
            import tensorflow as tf
            chunks = []
            for i in range(0, len(Xq), 2048):
                x = tf.convert_to_tensor(np.asarray(Xq[i:i + 2048], dtype="float32"))
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    q = model(x, training=False)
                    margin = q[:, 0] - q[:, 1]
                chunks.append(tape.gradient(margin, x).numpy())
            return np.concatenate(chunks)

        e.update(model=model, qfn=qfn, grad_fn=grad_fn, params=int(model.count_params()))
        return qfn(X)

    def _load_baseline(self, e):
        import joblib
        d = self.data
        X, n_feat = d["X_test"], d["X_test"].shape[1]
        cache = os.path.join(self.models_dir, f"{e['id']}.joblib")
        clf = None
        if os.path.exists(cache):
            try:
                clf = joblib.load(cache)
                if getattr(clf, "n_features_in_", n_feat) != n_feat:
                    clf = None
            except Exception:
                clf = None
        if clf is None:
            if d.get("X_train") is None or d.get("y_train") is None:
                raise RuntimeError("Training data isn't available to fit a baseline.")
            Xtr, ytr = d["X_train"], (np.asarray(d["y_train"]) != 0).astype(int)
            if len(Xtr) > 150_000:
                idx = np.random.default_rng(0).choice(len(Xtr), 150_000, replace=False)
                Xtr, ytr = Xtr[idx], ytr[idx]
            if e["id"] == "baseline_xgb":
                from xgboost import XGBClassifier
                clf = XGBClassifier(n_estimators=200, max_depth=6, n_jobs=-1, tree_method="hist", random_state=0)
            else:
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=60, max_depth=20, min_samples_leaf=2, n_jobs=-1,
                                             random_state=0)
            clf.fit(Xtr, ytr)
            try:
                os.makedirs(self.models_dir, exist_ok=True)
                joblib.dump(clf, cache)
            except Exception:
                pass

        def qfn(Xq):
            p = clf.predict_proba(np.asarray(Xq))[:, 1].astype("float32")
            return np.stack([1 - p, p], axis=1)

        e.update(model=clf, qfn=qfn, grad_fn=None, params=None)
        return qfn(X)

    # ---- robustness ---------------------------------------------------------------------------------
    def run_robustness(self, mid, sample=2000):
        """Start the robustness test in the background. Poll entry['robust'] for the result."""
        e = self.ready(mid)
        if e is None:
            raise RuntimeError("Load the model first.")
        if e.get("robust") and e["robust"]["status"] == "running":
            return e["robust"]
        e["robust"] = {"status": "running", "started": time.strftime("%H:%M:%S")}

        def work():
            try:
                res = evaluation.robustness(e["qfn"], self.data["X_test"], self.data["y_test"],
                                            grad_fn=e.get("grad_fn"), sample=sample)
                e["robust"] = {"status": "done", "result": res, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            except Exception as exc:
                e["robust"] = {"status": "error", "error": str(exc)[:300]}
        threading.Thread(target=work, daemon=True).start()
        return e["robust"]
