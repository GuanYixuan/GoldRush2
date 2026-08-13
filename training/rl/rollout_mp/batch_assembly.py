from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from simulator.errors import SimulatorRuleError
from training.opponents import OpponentSpec
from training.rl.ppo_buffer import PpoBatch

from .shared_memory import TransitionSharedMemory
from .types import EpisodeTask


def initial_tasks(
    *,
    seed: int,
    pair_count: int,
    map_ids: Sequence[int] | None,
    opponent_specs: Sequence[OpponentSpec] | None,
) -> list[EpisodeTask]:
    tasks: list[EpisodeTask] = []
    for pair_index in range(pair_count):
        episode_seed = seed + pair_index
        pair_id = f"pair-{pair_index:06d}-seed-{episode_seed}"
        tasks.append(
            EpisodeTask(
                task_id=f"{pair_id}-first",
                pair_id=pair_id,
                pair_role="first",
                seed=episode_seed,
                map_id=map_ids[pair_index % len(map_ids)] if map_ids is not None else None,
                agent_player_id=1,
                opponent_spec=opponent_specs[pair_index % len(opponent_specs)] if opponent_specs is not None else None,
                transition_slot=pair_index * 2,
            )
        )
    return tasks


def episode_payloads_to_batch(payloads: list[dict[str, Any]], transition_shared: TransitionSharedMemory) -> PpoBatch:
    import numpy as np

    if not payloads:
        raise SimulatorRuleError("multiprocess rollout produced no episode payloads")

    actor_planes: list[Any] = []
    actor_scalars: list[Any] = []
    fast_scalars: list[Any] = []
    critic_planes: list[Any] = []
    critic_scalars: list[Any] = []
    actions: list[Any] = []
    k: list[Any] = []
    order: list[Any] = []
    vp: list[Any] = []
    threshold_raw: list[Any] = []
    threshold_int: list[Any] = []
    old_logprob: list[Any] = []
    values: list[Any] = []
    rewards: list[Any] = []
    dones: list[Any] = []
    round_indices: list[Any] = []
    episode_ids: list[str] = []
    map_ids: list[int] = []
    agent_player_ids: list[int] = []
    infos: list[dict[str, Any]] = []

    for payload in payloads:
        transition_slot = int(payload["transition_slot"])
        episode_length = int(payload["episode_length"])
        if episode_length <= 0:
            raise SimulatorRuleError(f"episode payload {payload['task_id']!r} has no transitions")
        if not 0 <= transition_slot < int(transition_shared.done.shape[0]):
            raise SimulatorRuleError(f"transition slot out of range for {payload['task_id']!r}: {transition_slot}")
        if episode_length > int(transition_shared.done.shape[1]):
            raise SimulatorRuleError(
                f"episode payload {payload['task_id']!r} length {episode_length} exceeds shared transition length "
                f"{int(transition_shared.done.shape[1])}"
            )
        actor_planes.append(transition_shared.actor_planes[transition_slot, :episode_length])
        actor_scalars.append(transition_shared.actor_scalars[transition_slot, :episode_length])
        fast_scalars.append(transition_shared.fast_scalars[transition_slot, :episode_length])
        critic_planes.append(transition_shared.critic_planes[transition_slot, :episode_length])
        critic_scalars.append(transition_shared.critic_scalars[transition_slot, :episode_length])
        actions.append(transition_shared.actions[transition_slot, :episode_length])
        k.append(transition_shared.k[transition_slot, :episode_length])
        order.append(transition_shared.order[transition_slot, :episode_length])
        vp.append(transition_shared.vp[transition_slot, :episode_length])
        threshold_raw.append(transition_shared.threshold_raw[transition_slot, :episode_length])
        threshold_int.append(transition_shared.threshold_int[transition_slot, :episode_length])
        old_logprob.append(transition_shared.old_logprob[transition_slot, :episode_length])
        values.append(transition_shared.value[transition_slot, :episode_length])
        rewards.append(transition_shared.reward[transition_slot, :episode_length])
        dones.append(transition_shared.done[transition_slot, :episode_length])
        round_indices.append(transition_shared.round_index[transition_slot, :episode_length])
        episode_ids.extend((f"{payload['pair_id']}-{payload['pair_role']}",) * episode_length)
        map_ids.extend((int(payload["map_id"]),) * episode_length)
        agent_player_ids.extend((int(payload["agent_player_id"]),) * episode_length)
        episode_infos = tuple(payload["infos"])
        if len(episode_infos) != episode_length:
            raise SimulatorRuleError(
                f"episode payload {payload['task_id']!r} infos length {len(episode_infos)} != {episode_length}"
            )
        infos.extend(episode_infos)

    return PpoBatch.from_arrays(
        spatial_planes=np.concatenate(actor_planes, axis=0),
        scalars=np.concatenate(actor_scalars, axis=0),
        fast_scalars=np.concatenate(fast_scalars, axis=0),
        critic_planes=np.concatenate(critic_planes, axis=0),
        critic_scalars=np.concatenate(critic_scalars, axis=0),
        actions=np.concatenate(actions, axis=0),
        k=np.concatenate(k, axis=0),
        order=np.concatenate(order, axis=0),
        vp=np.concatenate(vp, axis=0),
        threshold_raw=np.concatenate(threshold_raw, axis=0),
        threshold_int=np.concatenate(threshold_int, axis=0),
        old_logprob=np.concatenate(old_logprob, axis=0),
        values=np.concatenate(values, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        dones=np.concatenate(dones, axis=0),
        episode_ids=tuple(episode_ids),
        round_indices=np.concatenate(round_indices, axis=0),
        map_ids=tuple(map_ids),
        agent_player_ids=np.asarray(agent_player_ids, dtype=np.int64),
        infos=tuple(infos),
    )
