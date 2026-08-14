from __future__ import annotations

from simulator.config import EpisodeConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.mechanisms.maps import MapPool, SpawnConfig
from training.opponents import OpponentSpec
from training.rl import SingleAgentEnvConfig
from training.rl.eval import EvaluationConfig


ANCHOR_EVAL_SEEDS = tuple(range(2026080200, 2026080210))
ANCHOR_EVAL_MAP_IDS = (1, 2, 3)
ANCHOR_EVAL_OPPONENT_SPECS = (
    OpponentSpec(
        kind="python",
        name="greedy_visible_gold",
        params={
            "target_score": "value_per_step",
            "avoid_bombs": True,
            "risk_weight": 1.0,
            "vp_policy": "never",
        },
    ),
    OpponentSpec(kind="python", name="fast_probe_v3_like"),
)


def anchor_eval_config(
    *,
    episode: EpisodeConfig | None = None,
    mechanisms: RoundStepMechanisms | None = None,
    map_pool: MapPool | None = None,
    spawn: SpawnConfig | None = None,
) -> EvaluationConfig:
    return EvaluationConfig.anchor_grid(
        env_config=SingleAgentEnvConfig(episode=EpisodeConfig() if episode is None else episode, opponent_spec=None),
        seeds=ANCHOR_EVAL_SEEDS,
        map_ids=ANCHOR_EVAL_MAP_IDS,
        opponent_specs=ANCHOR_EVAL_OPPONENT_SPECS,
        tag="anchor_eval_v1",
        mechanisms=mechanisms,
        map_pool=map_pool,
        spawn=spawn,
    )


__all__ = [
    "ANCHOR_EVAL_MAP_IDS",
    "ANCHOR_EVAL_OPPONENT_SPECS",
    "ANCHOR_EVAL_SEEDS",
    "anchor_eval_config",
]
