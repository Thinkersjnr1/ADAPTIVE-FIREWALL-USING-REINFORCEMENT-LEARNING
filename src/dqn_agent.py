import numpy as np
import random
import os
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size,
                 memory_size=50000,
                 use_target_network=True,
                 double_dqn=False):
        self.state_size    = state_size
        self.action_size   = action_size
        self.memory        = deque(maxlen=memory_size)
        self.gamma         = 0.95
        self.epsilon_start = 1.0
        self.epsilon       = self.epsilon_start
        self.epsilon_min   = 0.01
        # Paper Table II: "ε decay steps: 200,000 (Linear over training)".
        self.epsilon_decay_steps = 200000
        self.learning_rate = 0.001
        self.batch_size    = 64
        # Paper Table II: "Target update (C): 500 steps".
        self.target_update_steps = 500
        self.train_step_count    = 0
        # Ablation switches:
        self.use_target_network = use_target_network  # False = ablate target network
        self.double_dqn          = double_dqn           # True  = Double DQN target computation
        self.model         = self._build_model()
        self.target_model  = self._build_model()  # built even when unused, so save()/load() stay consistent
        self.update_target()

    def _build_model(self):
        model = Sequential([
            Dense(256, input_dim=self.state_size, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64,  activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        # Paper Table II: "Gradient clip norm: 1.0" -- this was missing
        # entirely from the original optimizer config.
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Direct call instead of predict() -- skips predict()'s extra
        # overhead (progress-bar setup, input validation) for a single
        # small array. Same weights, same output, just less call cost.
        q_values = self.model(state.reshape(1, -1), training=False).numpy()
        return np.argmax(q_values[0])

    def _update_epsilon(self):
        """Linear decay from epsilon_start to epsilon_min over
        epsilon_decay_steps *training* steps (Table II), replacing the old
        exponential per-call decay."""
        frac = min(1.0, self.train_step_count / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_min - self.epsilon_start)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch       = random.sample(self.memory, self.batch_size)
        states      = np.array([e[0] for e in batch])
        actions     = np.array([e[1] for e in batch])
        rewards     = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones       = np.array([e[4] for e in batch])
        targets     = self.model(states, training=False).numpy()

        # Target Q-value computation:
        #   Standard DQN : max_a' Q_eval(s', a')
        #   Double DQN   : Q_eval(s', argmax_a' Q_online(s', a'))
        #                  -- action *selection* uses the online network,
        #                  action *evaluation* uses eval_net, decoupling the
        #                  two to reduce DQN's overestimation bias.
        #   Target-network ablation: eval_net becomes the online model
        #   itself instead of a separately-synced copy.
        eval_net    = self.target_model if self.use_target_network else self.model
        next_q_eval = eval_net(next_states, training=False).numpy()

        if self.double_dqn:
            next_q_online = self.model(next_states, training=False).numpy()
            next_actions  = np.argmax(next_q_online, axis=1)
            next_q_values = next_q_eval[np.arange(self.batch_size), next_actions]
        else:
            next_q_values = np.amax(next_q_eval, axis=1)

        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * next_q_values[i]
        self.model.train_on_batch(states, targets)

        # Step-counted bookkeeping per Table II: epsilon decays linearly over
        # 200,000 steps, and the target network hard-updates every 500 steps
        # (skipped entirely under the target-network ablation).
        self.train_step_count += 1
        self._update_epsilon()
        if self.use_target_network and self.train_step_count % self.target_update_steps == 0:
            self.update_target()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"  Model saved → {path}")

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target()
        print(f"  Model loaded ← {path}")