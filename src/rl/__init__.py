"""Reinforcement learning for the laser-conveyor dispatching problem (version 0.5.0).

The dispatching problem is a semi-Markov decision process (SMDP): a decision is
taken whenever the laser is free and a part can be served. See docs/methods/rl.md
and docs/formal_formulation.md §4.

* :mod:`rl.features` — observation vector, action space ("serve the head of
  conveyor k") and action masks; shared by training and by the ``rl`` policy.
* :mod:`rl.env`      — Gymnasium environment around the simulator.
* :mod:`rl.train`    — behaviour cloning from a teacher policy + PPO fine-tuning.

Optional dependencies: ``requirements-rl.txt`` (gymnasium, torch,
stable-baselines3, sb3-contrib). The simulator does not need them.
"""
