from __future__ import annotations

from typing import Any

import warnings

import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.environments import MarsRover


class PolicyIteration(AbstractAgent):
    """
    Policy Iteration Agent.

    This agent performs standard tabular policy iteration on an environment
    with known transition dynamics and rewards. The policy is evaluated and
    improved until convergence.

    Parameters
    ----------
    env : MarsRover
        Environment instance. This class is designed specifically for the MarsRover env.
    gamma : float, optional
        Discount factor for future rewards, by default 0.9.
    seed : int, optional
        Random seed for policy initialization, by default 333.
    filename : str, optional
        Path to save/load the policy, by default "policy.npy".
    """

    def __init__(
        self,
        env: MarsRover,
        gamma: float = 0.9,
        seed: int = 333,
        filename: str = "policy.npy",
        **kwargs: dict,
    ) -> None:
        if hasattr(env, "unwrapped"):
            env = env.unwrapped  # type: ignore[assignment]
        self.env = env
        self.seed = seed
        self.filename = filename
        rng = np.random.default_rng(
            seed=self.seed
        )  # Uncomment and use this line if you need a random seed for reproducibility

        super().__init__(**kwargs)

        self.n_obs = self.env.observation_space.n  # type: ignore[attr-defined]
        self.n_actions = self.env.action_space.n  # type: ignore[attr-defined]

        # TODO: Get the MDP components (states, actions, transitions, rewards)
        self.S = env.states
        self.A = env.actions
        self.T = env.get_transition_matrix()
        # self.R = env.get_reward_per_action
        self.gamma = gamma
        self.R_sa = env.get_reward_per_action()

        # TODO: Initialize policy and Q-values
        # initialize policy randomly
        self.pi = rng.integers(low=0, high=self.n_actions, size=self.n_obs)
        # initialize Q-values to zeros
        self.Q = np.zeros((self.n_obs, self.n_actions))

        self.policy_fitted: bool = False
        self.steps: int = 0

    def predict_action(  # type: ignore[override]
        self, observation: int, info: dict | None = None, evaluate: bool = False
    ) -> tuple[int, dict]:
        """
        Predict an action using the current policy.

        Parameters
        ----------
        observation : int
            The current observation/state.
        info : dict or None, optional
            Additional info passed during prediction (unused).
        evaluate : bool, optional
            Evaluation mode toggle (unused here), by default False.

        Returns
        -------
        tuple[int, dict]
            The selected action and an empty info dictionary.
        """
        # TODO: Return the action according to current policy
        # raise NotImplementedError("predict_action() is not implemented.")
        action = self.pi[observation]
        return action, {}

    def update_agent(self, *args: tuple, **kwargs: dict) -> None:
        """Run policy iteration to compute the optimal policy and state-action values."""
        if not self.policy_fitted:
            # TODO: Call policy iteration with initialized values
            # raise NotImplementedError("update_agent() is not implemented.")

            MDP = (self.S, self.A, self.T, self.R_sa, self.gamma)
            self.Q, self.pi, self.steps = policy_iteration(self.Q, self.pi, MDP)
            print(f"Policy iteration converged in {self.steps} steps.")
            print(f"Final policy: {self.pi}")

            self.policy_fitted = True

    def save(self, *args: tuple[Any], **kwargs: dict) -> None:
        """
        Save the learned policy to a `.npy` file.

        Raises
        ------
        Warning
            If the policy has not yet been fitted.
        """
        if self.policy_fitted:
            np.save(self.filename, np.array(self.pi))
        else:
            warnings.warn("Tried to save policy but policy is not fitted yet.")

    def load(self, *args: tuple[Any], **kwargs: dict) -> np.ndarray:
        """
        Load the policy from file.

        Returns
        -------
        np.ndarray
            The loaded policy array.
        """
        self.pi = np.load(self.filename)
        self.policy_fitted = True
        return self.pi


def policy_evaluation(
    pi: np.ndarray,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Perform policy evaluation for a fixed policy.

    Parameters
    ----------
    pi : np.ndarray
        The current policy (array of actions).
    T : np.ndarray
        Transition probabilities T[s, a, s'].
    R_sa : np.ndarray
        Reward matrix R[s, a].
    gamma : float
        Discount factor.
    epsilon : float, optional
        Convergence threshold, by default 1e-8.

    Returns
    -------
    np.ndarray
        The evaluated value function V[s] for all states.
    """
    nS = R_sa.shape[0]
    V = np.zeros(nS)

    # TODO: imüplement Poolicy Evaluation for all states

    # reference: https://gurpreet-ai.github.io/policy-evaluation-deep-reinforcement-learning-series/

    while True:
        delta = 0
        for state in range(nS):
            old_v = V[state]
            action = pi[state]
            # estimate new value
            V[state] = R_sa[state, action] + gamma * np.sum(T[state, action, :] * V)

            # track the condition to stop
            delta = max(delta, abs(old_v - V[state]))
        if delta < epsilon:
            break

    return V


def policy_improvement(
    V: np.ndarray,
    T: np.ndarray,
    R_sa: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Improve the current policy based on the value function.

    Parameters
    ----------
    V : np.ndarray
        Current value function.
    T : np.ndarray
        Transition probabilities T[s, a, s'].
    R_sa : np.ndarray
        Reward matrix R[s, a].
    gamma : float
        Discount factor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Q-function and the improved policy.
    """
    nS, nA = R_sa.shape
    Q = np.zeros((nS, nA))
    pi_new = None
    # TODO: imüplement Poolicy Evaluation for all states

    for state in range(nS):
        for action in range(nA):
            Q[state, action] = R_sa[state, action] + gamma * np.sum(
                T[state, action, :] * V
            )

    # obtain new policy from Q-values
    pi_new = np.argmax(Q, axis=1)

    return Q, pi_new


def policy_iteration(
    Q: np.ndarray,
    pi: np.ndarray,
    MDP: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Full policy iteration loop until convergence.

    Parameters
    ----------
    Q : np.ndarray
        Initial Q-table (can be zeros).
    pi : np.ndarray
        Initial policy.
    MDP : tuple
        A tuple (S, A, T, R_sa, gamma) representing the MDP.
    epsilon : float, optional
        Convergence threshold for value updates, by default 1e-8.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        Final Q-table, final policy, and number of improvement steps.
    """
    S, A, T, R_sa, gamma = MDP

    # TODO: Combine evaluation and improvement in a loop.

    iterations = 0
    is_converged = False

    while not is_converged:
        # evaluate policy
        V = policy_evaluation(pi, T, R_sa, gamma, epsilon)
        # improve policy
        Q, pi_new = policy_improvement(V, T, R_sa, gamma)
        # check for convergence
        if np.array_equal(pi, pi_new):
            is_converged = True
        else:
            pi = pi_new
            iterations += 1

    return Q, pi, iterations


def check_environment():
    env = MarsRover()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Transition matrix shape: {env.get_transition_matrix().shape}")
    print(f"Reward matrix shape: {env.get_reward_per_action().shape}")

    # Try a simple reset and step
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    next_obs, reward, _, _, _ = env.step(1)
    print(f"Next observation after action 0: {next_obs}")

    return env


if __name__ == "__main__":
    env = check_environment()
    algo = PolicyIteration(env=MarsRover())
    algo.update_agent()
