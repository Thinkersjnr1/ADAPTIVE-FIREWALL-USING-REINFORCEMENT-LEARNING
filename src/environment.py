import numpy as np

class FirewallEnvironment:
    def __init__(self, X, y, tp_reward=1.0, tn_reward=0.5, fn_penalty=-1.0, fp_penalty=-0.5):
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        self.n_actions = 2        # 0 = ALLOW, 1 = BLOCK
        self.current_index = 0
        self.done = False
        # Configurable reward weights -- defaults match Table I exactly.
        # Exposed for reward-tuning experiments (e.g. penalizing false
        # positives more heavily on datasets where FPR is too high).
        self.tp_reward = tp_reward
        self.tn_reward = tn_reward
        self.fn_penalty = fn_penalty
        self.fp_penalty = fp_penalty

    def reset(self):
        """Shuffle data and start from beginning each episode."""
        perm = np.random.permutation(self.n_samples)
        self.X = self.X[perm]
        self.y = self.y[perm]
        self.current_index = 0
        self.done = False
        return self.X[0]          # return first state

    def step(self, action):
        """
        Agent takes action on current network packet.
        Returns: next_state, reward, done
        """
        true_label = self.y[self.current_index]

        # 0 = BENIGN, anything else = ATTACK
        is_attack = int(true_label != 0)

        # Reward function (asymmetric — missing attack is worse, by default)
        if action == 1 and is_attack == 1:
            reward = self.tp_reward   # True Positive  — attack correctly blocked
        elif action == 0 and is_attack == 0:
            reward = self.tn_reward   # True Negative  — benign correctly allowed
        elif action == 0 and is_attack == 1:
            reward = self.fn_penalty  # False Negative — attack missed
        else:
            reward = self.fp_penalty  # False Positive — benign wrongly blocked

        # Move to next packet
        self.current_index += 1

        if self.current_index >= self.n_samples:
            self.done = True
            next_state = self.X[0]
        else:
            next_state = self.X[self.current_index]

        return next_state, reward, self.done

    def get_state_size(self):
        return self.n_features

    def get_action_size(self):
        return self.n_actions