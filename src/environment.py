import numpy as np

class FirewallEnvironment:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        self.n_actions = 2        # 0 = ALLOW, 1 = BLOCK
        self.current_index = 0
        self.done = False

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

        # Reward function (asymmetric — missing attack is worse)
        if action == 1 and is_attack == 1:
            reward = 1.0          # True Positive  — attack correctly blocked
        elif action == 0 and is_attack == 0:
            reward = 0.5          # True Negative  — benign correctly allowed
        elif action == 0 and is_attack == 1:
            reward = -1.0         # False Negative — attack missed (worst)
        else:
            reward = -0.5         # False Positive — benign wrongly blocked

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