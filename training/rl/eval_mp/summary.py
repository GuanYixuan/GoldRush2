from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from statistics import mean, median
from typing import Any

from .types import EvalEpisodeSummary


def summarize_eval(summaries: Sequence[EvalEpisodeSummary]) -> dict[str, Any]:
    if not summaries:
        return {
            "episode_count": 0,
            "mean_agent_net_gold": 0.0,
            "mean_opponent_net_gold": 0.0,
            "win_rate": 0.0,
            "mean_margin": 0.0,
        }
    return {
        **_summary_for_group(summaries),
        "by_setting": _group_summary(summaries, lambda item: item.setting),
        "by_map": _group_summary(summaries, lambda item: str(item.map_id)),
        "by_opponent": _group_summary(summaries, lambda item: item.opponent),
    }


def _group_summary(
    summaries: Sequence[EvalEpisodeSummary],
    key_fn: Callable[[EvalEpisodeSummary], str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EvalEpisodeSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[key_fn(summary)].append(summary)
    return {key: _summary_for_group(items) for key, items in sorted(grouped.items())}


def _summary_for_group(summaries: Sequence[EvalEpisodeSummary]) -> dict[str, Any]:
    margins = [item.margin for item in summaries]
    agent_pickup_gold = [item.agent_pickup_gold for item in summaries]
    agent_bomb_loss = [item.agent_bomb_lost_gold for item in summaries]
    return {
        "episode_count": len(summaries),
        "mean_agent_net_gold": mean(item.agent_net_gold for item in summaries),
        "mean_opponent_net_gold": mean(item.opponent_net_gold for item in summaries),
        "win_rate": mean(1.0 if item.agent_won else 0.0 for item in summaries),
        "mean_margin": mean(margins),
        "median_margin": median(margins),
        "min_margin": min(margins),
        "max_margin": max(margins),
        "mean_episode_length": mean(item.episode_length for item in summaries),
        "mean_agent_pickup_gold": mean(agent_pickup_gold),
        "mean_agent_pickups": mean(item.agent_pickups for item in summaries),
        "mean_agent_bomb_lost_gold": mean(agent_bomb_loss),
        "mean_agent_bomb_triggers": mean(item.agent_bomb_triggers for item in summaries),
        "mean_agent_trample_penalty": mean(item.agent_trample_penalty for item in summaries),
        "mean_agent_vision_spent": mean(item.agent_vision_spent for item in summaries),
        "mean_opponent_pickup_gold": mean(item.opponent_pickup_gold for item in summaries),
        "mean_opponent_pickups": mean(item.opponent_pickups for item in summaries),
        "mean_opponent_bomb_lost_gold": mean(item.opponent_bomb_lost_gold for item in summaries),
        "mean_opponent_bomb_triggers": mean(item.opponent_bomb_triggers for item in summaries),
        "agent_bomb_loss_ratio": sum(agent_bomb_loss) / max(sum(agent_bomb_loss) + sum(agent_pickup_gold), 1),
    }
