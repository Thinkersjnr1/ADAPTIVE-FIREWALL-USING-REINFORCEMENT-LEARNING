import numpy as np
import random
import os
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size    = state_size
        self.action_size   = action_size
        self.memory        = deque(maxlen=50000)
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size    = 64
        self.model         = self._build_model()
        self.target_model  = self._build_model()
        self.update_target()

    def _build_model(self):
        model = Sequential([
            Dense(256, input_dim=self.state_size, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64,  activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch       = random.sample(self.memory, self.batch_size)
        states      = np.array([e[0] for e in batch])
        actions     = np.array([e[1] for e in batch])
        rewards     = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones       = np.array([e[4] for e in batch])
        targets     = self.model.predict(states, verbose=0)
        next_q      = self.target_model.predict(next_states, verbose=0)
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q[i])
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"  Model saved → {path}")

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target()
        print(f"  Model loaded ← {path}")