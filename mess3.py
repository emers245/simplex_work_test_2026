#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mess3: The following is a Python class for the Mess3 process from Marzen & Crutchfield (2017). 
It defines the transition matrices, stationary distribution, and methods for generating sequences 
and updating beliefs based on observed tokens."""

import numpy as np
import matplotlib.pyplot as plt

class Mess3:
    """
    Mess3 process from Marzen & Crutchfield (2017).
    3-state edge-emitting HMM parameterized by (alpha, x).
    
    Derived quantities:
        beta = (1 - alpha) / 2
        y = 1 - 2x
    
    Transition matrices T^(0), T^(1), T^(2) are from
    Shai et al. (2026) Appendix C.1.1, equations 22-24.
    """
    
    def __init__(self, alpha=0.6, x=0.15):
        self.alpha = alpha
        self.x = x
        self.beta = (1 - alpha) / 2
        self.y = 1 - 2 * x
        
        a, b, xv, yv = self.alpha, self.beta, self.x, self.y
        
        # Token-labeled transition matrices (eq 22-24)
        # T^(z)_{s,s'} = P(s', z | s)
        self.T = np.array([
            # T^(0)
            [[a*yv, b*xv, b*xv],
             [a*xv, b*yv, b*xv],
             [a*xv, b*xv, b*yv]],
            # T^(1)
            [[b*yv, a*xv, b*xv],
             [b*xv, a*yv, b*xv],
             [b*xv, a*xv, b*yv]],
            # T^(2)
            [[b*yv, b*xv, a*xv],
             [b*xv, b*yv, a*xv],
             [b*xv, b*xv, a*yv]],
        ])  # shape: (3, 3, 3) = (token, from_state, to_state)
        
        # Net transition matrix (eq 25): sum over tokens
        self.T_net = self.T.sum(axis=0)  # shape: (3, 3)
        
        # Stationary distribution (uniform for Mess3)
        self.pi = np.array([1/3, 1/3, 1/3])
        
        # Verify: pi should be a left eigenvector of T_net with eigenvalue 1
        assert np.allclose(self.pi @ self.T_net, self.pi), \
            "Stationary distribution check failed"
    
    def generate_sequence(self, length, initial_state=None, rng=None):
        """
        Generate a sequence of tokens and hidden states.
        
        Returns:
            tokens: array of shape (length,) with values in {0, 1, 2}
            states: array of shape (length+1,) — includes initial state
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if initial_state is None:
            # Sample initial state from stationary distribution
            initial_state = rng.choice(3, p=self.pi)
        
        states = np.zeros(length + 1, dtype=int)
        tokens = np.zeros(length, dtype=int)
        states[0] = initial_state
        
        for t in range(length):
            s = states[t]
            # Joint distribution over (token, next_state) given current state
            # Row s of each T^(z) gives P(s'|s, z)*P(z|s) ... but actually
            # T^(z)_{s,s'} = P(s', z | s), so the row sums over s' and z = 1
            # We need to sample (z, s') jointly from T[:, s, :] 
            probs = self.T[:, s, :].flatten()  # shape (9,): (token, next_state)
            idx = rng.choice(9, p=probs)
            tokens[t] = idx // 3
            states[t + 1] = idx % 3
        
        return tokens, states
    
    def belief_update(self, eta, token):
        """
        Update predictive vector (belief state) given observed token.
        
        eta^(x1:l+1) = eta^(x1:l) T^(x_{l+1}) / (eta^(x1:l) T^(x_{l+1}) 1)
        
        Args:
            eta: current belief vector, shape (3,)
            token: observed token in {0, 1, 2}
        Returns:
            updated belief vector, shape (3,)
        """
        unnormed = eta @ self.T[token]
        return unnormed / unnormed.sum()
    
    def belief_trajectory(self, tokens, initial_belief=None):
        """
        Compute the sequence of belief states induced by a token sequence.
        
        Returns:
            beliefs: array of shape (len(tokens)+1, 3), starting from initial belief
        """
        if initial_belief is None:
            initial_belief = self.pi.copy()
        
        beliefs = np.zeros((len(tokens) + 1, 3))
        beliefs[0] = initial_belief
        
        for t, tok in enumerate(tokens):
            beliefs[t + 1] = self.belief_update(beliefs[t], tok)
        
        return beliefs