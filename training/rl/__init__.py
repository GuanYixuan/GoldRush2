"""Minimal RL environment adapters for GoldRush2 training."""

from .env import SingleAgentEnvConfig, SingleAgentGoldRushEnv
from .eval import EvaluationCase, EvaluationConfig, EvaluationResult, EvaluationSuite, evaluate_policy
from .eval_mp import EvalEpisodeSummary, EvalTask, ParallelEvalConfig, ParallelEvalPool, evaluate_parallel
from .multiprocess_rollout import MultiprocessRolloutConfig, MultiprocessRolloutPool, collect_multiprocess_ppo_rollouts
from .ppo import (
    PpoConfig,
    PpoEvaluation,
    PpoUpdateStats,
    collect_ppo_rollouts,
    critic_only_update,
    evaluate_actions,
    ppo_update,
)
from .ppo_buffer import PpoBatch, PpoMiniBatch, PpoTransition
from .privileged_critic_features import (
    extract_privileged_critic_features,
    privileged_critic_channel_names,
    privileged_critic_feature_schema,
    privileged_critic_scalar_names,
)
from .rewards import TerminalWinMarginGoldGainReward, WinLossReward
from .rollout import EpisodeBatch, Trajectory, Transition
from .runtime_policy import RuntimeAwarePolicy, RuntimePolicyWrapper
from .sampler import BatchRolloutSampler
from .types import ResetResult, StepResult

__all__ = [
    "BatchRolloutSampler",
    "EpisodeBatch",
    "EvaluationCase",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationSuite",
    "EvalEpisodeSummary",
    "EvalTask",
    "MultiprocessRolloutConfig",
    "MultiprocessRolloutPool",
    "ParallelEvalConfig",
    "ParallelEvalPool",
    "PpoBatch",
    "PpoConfig",
    "PpoEvaluation",
    "PpoMiniBatch",
    "PpoTransition",
    "PpoUpdateStats",
    "ResetResult",
    "RuntimeAwarePolicy",
    "RuntimePolicyWrapper",
    "SingleAgentEnvConfig",
    "SingleAgentGoldRushEnv",
    "StepResult",
    "TerminalWinMarginGoldGainReward",
    "Trajectory",
    "Transition",
    "WinLossReward",
    "collect_multiprocess_ppo_rollouts",
    "collect_ppo_rollouts",
    "critic_only_update",
    "evaluate_actions",
    "evaluate_parallel",
    "evaluate_policy",
    "extract_privileged_critic_features",
    "privileged_critic_channel_names",
    "privileged_critic_feature_schema",
    "privileged_critic_scalar_names",
    "ppo_update",
]
