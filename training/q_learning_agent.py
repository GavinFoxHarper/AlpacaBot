#!/usr/bin/env python3
"""
Q-Learning Agent for LAEF Trading System
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging

class LAEFAgent:
    """Q-Learning Agent for trade decisions"""
    
    def __init__(self, state_size=12, model_path=None, pretrained=True):
        self.state_size = state_size
        self.model_path = model_path or 'models/laef_agent.keras'
        
        # Build or load model
        self.model = self._build_model()
        
        if pretrained and os.path.exists(self.model_path):
            self.load_model()
            
        self.training_memory = []
        self.batch_size = 32
        
    def _build_model(self):
        """Build neural network model"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_size,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(3, activation='linear')  # Q-values for [hold, buy, sell]
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def predict_q_values(self, state):
        """Predict Q-values for actions"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
            
        if state.ndim == 1:
            state = state.reshape(1, -1)
            
        return self.model.predict(state, verbose=0)[0]
        
    def train_on_batch(self, states, targets):
        """Train on a batch of data"""
        return self.model.train_on_batch(states, targets)
        
    def update_from_memory(self, gamma=0.95):
        """Train on stored experience"""
        if len(self.training_memory) < self.batch_size:
            return
            
        batch = np.random.choice(self.training_memory, self.batch_size)
        
        states = []
        targets = []
        
        for state, action, reward, next_state in batch:
            target = reward
            if next_state is not None:
                next_q_values = self.predict_q_values(next_state)
                target = reward + gamma * np.max(next_q_values)
                
            target_f = self.predict_q_values(state)
            target_f[action] = target
            
            states.append(state)
            targets.append(target_f)
            
        return self.train_on_batch(np.array(states), np.array(targets))
        
    def add_to_memory(self, state, action, reward, next_state):
        """Store experience for training"""
        self.training_memory.append((state, action, reward, next_state))
        
        # Limit memory size
        if len(self.training_memory) > 10000:
            self.training_memory = self.training_memory[-10000:]
            
    def save_model(self, path=None):
        """Save model to disk"""
        save_path = path or self.model_path
        self.model.save(save_path)
        
    def load_model(self, path=None):
        """Load model from disk"""
        load_path = path or self.model_path
        if os.path.exists(load_path):
            self.model = load_model(load_path)