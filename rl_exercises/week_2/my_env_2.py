from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        transition_probabilities: np.ndarray = np.ones((2, 2)),
        rewards: list[float] = [0, 5],
        horizon: int = 10,
        seed: int | None = None,
    ):
        """Initializes the observation and action space for the environment."""
        self.rng = np.random.default_rng(seed)

        self.rewards = list(rewards)
        self.P = np.array(transition_probabilities)
        self.horizon = int(horizon)
        self.current_steps = 0
        self.state = 0  # start at state 0

        # spaces
        n = self.P.shape[0]
        self.observation_space = gym.spaces.Discrete(n)
        self.action_space = gym.spaces.Discrete(2)

        # helpers
        self.states = np.arange(n)
        self.actions = np.arange(2)

        # transition matrix
        self.transition_matrix = self.T = self.get_transition_matrix()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            Seed for environment reset (unused).
        options : dict, optional
            Additional reset options (unused).

        Returns
        -------
        state : int
            Initial state (always 0).
        info : dict
            An empty info dictionary.
        """
        self.current_steps = 0
        self.state = 0
        return self.state, {}

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int
            Action to take (0: move to state 0, 1: move to state 1).

        Returns
        -------
        next_state : int
            The resulting state after taking the action.
        reward : float
            The reward at the new state.
        terminated : bool
            Whether the episode ended due to task success (always False here).
        truncated : bool
            Whether the episode ended due to reaching the time limit.
        info : dict
            An empty dictionary.
        """
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1

        # stochastic flip with prob 1 - P[state, action]
        p = float(self.P[self.state, action])
        follow = self.rng.random() < p
        a_used = action if follow else 1 - action

        delta = -1 if a_used == 0 else 1
        next_state = max(0, min(self.states[-1], self.state + delta))

        reward = float(self.rewards[self.state])
        terminated = False
        truncated = self.current_steps >= self.horizon

        return next_state, reward, terminated, truncated, {}

    def get_reward_per_action(self) -> np.ndarray:
        """
        Return the reward function R[s, a] for each (state, action) pair.

        Returns
        -------
        R : np.ndarray
            A (num_states, num_actions) array of rewards.
        """
        nS, nA = self.observation_space.n, self.action_space.n
        R = np.zeros((nS, nA), dtype=float)
        for s in range(nS):
            for a in range(nA):
                nxt = max(0, min(nS - 1, s + (-1 if a == 0 else 1)))
                R[s, a] = float(self.rewards[nxt])
        return R

    def get_transition_matrix(
        self,
        S: np.ndarray | None = None,
        A: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Construct a deterministic transition matrix T[s, a, s'].

        Parameters
        ----------
        S : np.ndarray, optional
            Array of states. Uses internal states if None.
        A : np.ndarray, optional
            Array of actions. Uses internal actions if None.
        P : np.ndarray, optional
            Action success probabilities. Uses internal P if None.

        Returns
        -------
        T : np.ndarray
            A (num_states, num_actions, num_states) tensor where
            T[s, a, s'] = probability of transitioning to s' from s via a.
        """
        if S is None or A is None or P is None:
            S, A, P = self.states, self.actions, self.P

        nS, nA = len(S), len(A)
        T = np.zeros((nS, nA, nS), dtype=float)
        for s in S:
            for a in A:
                s_next = max(0, min(nS - 1, s + (-1 if a == 0 else 1)))
                T[s, a, s_next] = float(P[s, a])
        return T


class PartialObsWrapper(gym.Wrapper):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env: gym.Env,
        noise: float = 0.1,
        decay: float = 0.01,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        env : gym.Env
            The base environment.
        noise : float
            Initial probability of noisy observation.
        decay : float
            Decay rate for noise per timestep (e.g., 0.01).
        seed : int or None
            RNG seed.
        """
        super().__init__(env)
        assert 0.0 <= noise <= 1.0, "noise must be in [0,1]"
        self.noise = noise
        self.decay = decay
        self.rng = np.random.default_rng(seed)
        self.timestep = 0

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[int, dict[str, Any]]:
        self.timestep = 0  # reset the step count at the beginning of each episode
        true_obs, info = self.env.reset(seed=seed, options=options)
        return self._noisy_obs(true_obs), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.timestep += 1
        true_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy_obs(true_obs), reward, terminated, truncated, info

    def _current_noise(self) -> float:
        """
        Compute noise at the current timestep using exponential decay.
        """
        return self.noise * np.exp(-self.decay * self.timestep)

    def _noisy_obs(self, true_obs: int) -> int:
        """
        Possibly corrupt the observation using current noise level.
        """
        noise_level = self._current_noise()
        if self.rng.random() < noise_level:
            n = self.observation_space.n
            others = [s for s in range(n) if s != true_obs]
            return int(self.rng.choice(others))
        return int(true_obs)
