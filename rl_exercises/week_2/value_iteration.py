from __future__ import annotations

from typing import Any, Tuple

import gymnasium
import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.environments import MarsRover


class ValueIteration(AbstractAgent):
    """Agent that computes an optimal policy via Value Iteration.

    Parameters
    ----------
    env : MarsRover or gymnasium.Env
        The target environment, must expose `.states`, `.actions`,
        `.transition_matrix` and `.get_reward_per_action()`.
    gamma : float, default=0.9
        Discount factor for future rewards.
    seed : int, default=333
        Random seed for tie‐breaking among equally‐good actions.

    Attributes
    ----------
    V : np.ndarray, shape (n_states,)
        The computed optimal value function.
    pi : np.ndarray, shape (n_states,)
        The greedy policy derived from V.
    policy_fitted : bool
        Whether value iteration has been run yet.
    """

    def __init__(
        self,
        env: MarsRover | gymnasium.Env,
        gamma: float = 0.9,
        seed: int = 333,
        **kwargs: dict,
    ) -> None:
        if hasattr(env, "unwrapped"):
            env = env.unwrapped  # type: ignore
        super().__init__(**kwargs)

        self.env = env
        self.gamma = gamma
        self.seed = seed

        # TODO: Extract MDP components from the environment
        self.S = env.states
        self.A = env.actions
        self.T = env.get_transition_matrix()
        self.R_sa = env.get_reward_per_action()
        self.n_states = len(env.states)
        self.n_actions = len(env.actions)

        # placeholders
        self.V = np.zeros(self.n_states, dtype=float)
        self.pi = np.zeros(self.n_states, dtype=int)
        self.policy_fitted = False

    def update_agent(self, *args: tuple[Any], **kwargs: dict) -> None:
        """Run value iteration and store the resulting V and π."""
        if self.policy_fitted:
            return

        V_opt, pi_opt = value_iteration(
            T=self.T,
            R_sa=self.R_sa,
            gamma=self.gamma,
            seed=self.seed,
        )

        # TODO: Call value_iteration() with extracted MDP components
        self.V = V_opt
        self.pi = pi_opt
        self.policy_fitted = True

    def predict_action(
        self,
        observation: int,
        info: dict | None = None,
        evaluate: bool = False,
    ) -> tuple[int, dict]:
        """Choose action = π(observation). Runs update if needed."""
        if not self.policy_fitted:
            self.update_agent()

        # TODO: Return action from learned policy
        # raise NotImplementedError("predict_action() is not implemented.")
        return self.pi[observation], {}


def value_iteration(
    *,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
    seed: int | None = None,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Value Iteration on a finite MDP.

    Solves for
        V*(s) = max_a [ R_sa[s,a] + γ ∑_{s'} T[s,a,s'] V*(s') ]
    and then
        π*(s) = argmax_a [ R_sa[s,a] + γ ∑_{s'} T[s,a,s'] V*(s') ].

    Parameters
    ----------
    T : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R_sa : np.ndarray, shape (n_states, n_actions)
        Rewards for each (state, action).
    gamma : float
        Discount factor (0 ≤ γ < 1).
    seed : int or None
        RNG seed for tie‐breaking among equal actions.
    epsilon : float
        Stopping threshold on max value‐update difference.

    Returns
    -------
    V : np.ndarray, shape (n_states,)
        Optimal state‐value function.
    pi : np.ndarray, shape (n_states,)
        Greedy policy w.r.t. V, with random tie‐breaking.
    """
    n_states, n_actions = R_sa.shape
    V = np.zeros(n_states, dtype=float)
    rng = np.random.default_rng(seed)  # uncomment this
    pi = np.zeros(n_states, dtype=int)

    # TODO: update V using the Q values until convergence
    # reference: https://gibberblot.github.io/rl-notes/single-agent/value-iteration.html
    while True:
        delta = 0
        for state in range(n_states):
            old_v = V[state]
            Q = np.zeros(n_actions)
            for action in range(n_actions):
                Q[action] = R_sa[state, action] + gamma * np.sum(
                    T[state, action, :] * V
                )
            V[state] = np.max(Q)
            delta = max(delta, abs(old_v - V[state]))
        if delta < epsilon:
            break

    # TODO: Extract the greedy policy from V and update pi

    for state in range(n_states):
        Q = np.zeros(n_actions)
        for action in range(n_actions):
            Q[action] = R_sa[state, action] + gamma * np.sum(T[state, action, :] * V)

        # random tie-breaking in case we have multiple maxima for a state
        pi[state] = rng.choice(np.flatnonzero(Q == Q.max()))

    return V, pi
