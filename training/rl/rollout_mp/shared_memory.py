from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

from training.rl.privileged_critic_features import SCALAR_FEATURES as CRITIC_SCALAR_FEATURES
from training.rl.privileged_critic_features import SPATIAL_CHANNELS as CRITIC_SPATIAL_CHANNELS


ACTOR_SPATIAL_CHANNELS = 43
ACTOR_SCALAR_FEATURES = 10
FAST_SCALAR_FEATURES = 2
GRID_SIZE = 17


@dataclass
class FeatureSharedMemory:
    actor_planes_shm: shared_memory.SharedMemory
    actor_scalars_shm: shared_memory.SharedMemory
    fast_scalars_shm: shared_memory.SharedMemory
    critic_planes_shm: shared_memory.SharedMemory
    critic_scalars_shm: shared_memory.SharedMemory
    actor_planes: Any
    actor_scalars: Any
    fast_scalars: Any
    critic_planes: Any
    critic_scalars: Any

    def config(self) -> dict[str, Any]:
        return {
            "actor_planes": shared_array_config(self.actor_planes_shm, self.actor_planes),
            "actor_scalars": shared_array_config(self.actor_scalars_shm, self.actor_scalars),
            "fast_scalars": shared_array_config(self.fast_scalars_shm, self.fast_scalars),
            "critic_planes": shared_array_config(self.critic_planes_shm, self.critic_planes),
            "critic_scalars": shared_array_config(self.critic_scalars_shm, self.critic_scalars),
        }

    def close(self) -> None:
        for shm in self._shms():
            shm.close()

    def unlink(self) -> None:
        for shm in self._shms():
            shm.unlink()

    def _shms(self) -> tuple[shared_memory.SharedMemory, ...]:
        return (
            self.actor_planes_shm,
            self.actor_scalars_shm,
            self.fast_scalars_shm,
            self.critic_planes_shm,
            self.critic_scalars_shm,
        )


@dataclass
class TransitionSharedMemory:
    actor_planes_shm: shared_memory.SharedMemory
    actor_scalars_shm: shared_memory.SharedMemory
    fast_scalars_shm: shared_memory.SharedMemory
    critic_planes_shm: shared_memory.SharedMemory
    critic_scalars_shm: shared_memory.SharedMemory
    actions_shm: shared_memory.SharedMemory
    k_shm: shared_memory.SharedMemory
    order_shm: shared_memory.SharedMemory
    vp_shm: shared_memory.SharedMemory
    threshold_raw_shm: shared_memory.SharedMemory
    threshold_int_shm: shared_memory.SharedMemory
    old_logprob_shm: shared_memory.SharedMemory
    value_shm: shared_memory.SharedMemory
    reward_shm: shared_memory.SharedMemory
    done_shm: shared_memory.SharedMemory
    round_index_shm: shared_memory.SharedMemory
    actor_planes: Any
    actor_scalars: Any
    fast_scalars: Any
    critic_planes: Any
    critic_scalars: Any
    actions: Any
    k: Any
    order: Any
    vp: Any
    threshold_raw: Any
    threshold_int: Any
    old_logprob: Any
    value: Any
    reward: Any
    done: Any
    round_index: Any

    def config(self) -> dict[str, Any]:
        return {
            "actor_planes": shared_array_config(self.actor_planes_shm, self.actor_planes),
            "actor_scalars": shared_array_config(self.actor_scalars_shm, self.actor_scalars),
            "fast_scalars": shared_array_config(self.fast_scalars_shm, self.fast_scalars),
            "critic_planes": shared_array_config(self.critic_planes_shm, self.critic_planes),
            "critic_scalars": shared_array_config(self.critic_scalars_shm, self.critic_scalars),
            "actions": shared_array_config(self.actions_shm, self.actions),
            "k": shared_array_config(self.k_shm, self.k),
            "order": shared_array_config(self.order_shm, self.order),
            "vp": shared_array_config(self.vp_shm, self.vp),
            "threshold_raw": shared_array_config(self.threshold_raw_shm, self.threshold_raw),
            "threshold_int": shared_array_config(self.threshold_int_shm, self.threshold_int),
            "old_logprob": shared_array_config(self.old_logprob_shm, self.old_logprob),
            "value": shared_array_config(self.value_shm, self.value),
            "reward": shared_array_config(self.reward_shm, self.reward),
            "done": shared_array_config(self.done_shm, self.done),
            "round_index": shared_array_config(self.round_index_shm, self.round_index),
        }

    def close(self) -> None:
        for shm in self._shms():
            shm.close()

    def unlink(self) -> None:
        for shm in self._shms():
            shm.unlink()

    def _shms(self) -> tuple[shared_memory.SharedMemory, ...]:
        return (
            self.actor_planes_shm,
            self.actor_scalars_shm,
            self.fast_scalars_shm,
            self.critic_planes_shm,
            self.critic_scalars_shm,
            self.actions_shm,
            self.k_shm,
            self.order_shm,
            self.vp_shm,
            self.threshold_raw_shm,
            self.threshold_int_shm,
            self.old_logprob_shm,
            self.value_shm,
            self.reward_shm,
            self.done_shm,
            self.round_index_shm,
        )


def create_feature_shared_memory(num_workers: int) -> FeatureSharedMemory:
    import numpy as np

    created: list[shared_memory.SharedMemory] = []

    def create_array(shape: tuple[int, ...], dtype: Any) -> tuple[shared_memory.SharedMemory, Any]:
        np_dtype = np.dtype(dtype)
        shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * np_dtype.itemsize)
        created.append(shm)
        return shm, np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)

    try:
        actor_planes_shm, actor_planes = create_array((num_workers, ACTOR_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32)
        actor_scalars_shm, actor_scalars = create_array((num_workers, ACTOR_SCALAR_FEATURES), np.float32)
        fast_scalars_shm, fast_scalars = create_array((num_workers, FAST_SCALAR_FEATURES), np.float32)
        critic_planes_shm, critic_planes = create_array((num_workers, CRITIC_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32)
        critic_scalars_shm, critic_scalars = create_array((num_workers, CRITIC_SCALAR_FEATURES), np.float32)
    except Exception:
        for shm in created:
            shm.close()
            shm.unlink()
        raise

    return FeatureSharedMemory(
        actor_planes_shm=actor_planes_shm,
        actor_scalars_shm=actor_scalars_shm,
        fast_scalars_shm=fast_scalars_shm,
        critic_planes_shm=critic_planes_shm,
        critic_scalars_shm=critic_scalars_shm,
        actor_planes=actor_planes,
        actor_scalars=actor_scalars,
        fast_scalars=fast_scalars,
        critic_planes=critic_planes,
        critic_scalars=critic_scalars,
    )


def attach_feature_shared_memory(config: dict[str, Any]) -> FeatureSharedMemory:
    actor_planes_shm, actor_planes = attach_array(config, "actor_planes")
    attached = [actor_planes_shm]
    try:
        actor_scalars_shm, actor_scalars = attach_array(config, "actor_scalars")
        attached.append(actor_scalars_shm)
        fast_scalars_shm, fast_scalars = attach_array(config, "fast_scalars")
        attached.append(fast_scalars_shm)
        critic_planes_shm, critic_planes = attach_array(config, "critic_planes")
        attached.append(critic_planes_shm)
        critic_scalars_shm, critic_scalars = attach_array(config, "critic_scalars")
    except Exception:
        for shm in attached:
            shm.close()
        raise
    return FeatureSharedMemory(
        actor_planes_shm=actor_planes_shm,
        actor_scalars_shm=actor_scalars_shm,
        fast_scalars_shm=fast_scalars_shm,
        critic_planes_shm=critic_planes_shm,
        critic_scalars_shm=critic_scalars_shm,
        actor_planes=actor_planes,
        actor_scalars=actor_scalars,
        fast_scalars=fast_scalars,
        critic_planes=critic_planes,
        critic_scalars=critic_scalars,
    )


def create_transition_shared_memory(*, episode_count: int, round_count: int) -> TransitionSharedMemory:
    import numpy as np

    created: list[shared_memory.SharedMemory] = []

    def create_array(shape: tuple[int, ...], dtype: Any) -> tuple[shared_memory.SharedMemory, Any]:
        np_dtype = np.dtype(dtype)
        shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * np_dtype.itemsize)
        created.append(shm)
        return shm, np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)

    try:
        actor_planes_shm, actor_planes = create_array((episode_count, round_count, ACTOR_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32)
        actor_scalars_shm, actor_scalars = create_array((episode_count, round_count, ACTOR_SCALAR_FEATURES), np.float32)
        fast_scalars_shm, fast_scalars = create_array((episode_count, round_count, FAST_SCALAR_FEATURES), np.float32)
        critic_planes_shm, critic_planes = create_array((episode_count, round_count, CRITIC_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32)
        critic_scalars_shm, critic_scalars = create_array((episode_count, round_count, CRITIC_SCALAR_FEATURES), np.float32)
        actions_shm, actions = create_array((episode_count, round_count, 6), np.int64)
        k_shm, k = create_array((episode_count, round_count), np.int64)
        order_shm, order = create_array((episode_count, round_count), np.int64)
        vp_shm, vp = create_array((episode_count, round_count), np.int64)
        threshold_raw_shm, threshold_raw = create_array((episode_count, round_count), np.float32)
        threshold_int_shm, threshold_int = create_array((episode_count, round_count), np.int64)
        old_logprob_shm, old_logprob = create_array((episode_count, round_count), np.float32)
        value_shm, value = create_array((episode_count, round_count), np.float32)
        reward_shm, reward = create_array((episode_count, round_count), np.float32)
        done_shm, done = create_array((episode_count, round_count), np.bool_)
        round_index_shm, round_index = create_array((episode_count, round_count), np.int64)
    except Exception:
        for shm in created:
            shm.close()
            shm.unlink()
        raise

    return TransitionSharedMemory(
        actor_planes_shm=actor_planes_shm,
        actor_scalars_shm=actor_scalars_shm,
        fast_scalars_shm=fast_scalars_shm,
        critic_planes_shm=critic_planes_shm,
        critic_scalars_shm=critic_scalars_shm,
        actions_shm=actions_shm,
        k_shm=k_shm,
        order_shm=order_shm,
        vp_shm=vp_shm,
        threshold_raw_shm=threshold_raw_shm,
        threshold_int_shm=threshold_int_shm,
        old_logprob_shm=old_logprob_shm,
        value_shm=value_shm,
        reward_shm=reward_shm,
        done_shm=done_shm,
        round_index_shm=round_index_shm,
        actor_planes=actor_planes,
        actor_scalars=actor_scalars,
        fast_scalars=fast_scalars,
        critic_planes=critic_planes,
        critic_scalars=critic_scalars,
        actions=actions,
        k=k,
        order=order,
        vp=vp,
        threshold_raw=threshold_raw,
        threshold_int=threshold_int,
        old_logprob=old_logprob,
        value=value,
        reward=reward,
        done=done,
        round_index=round_index,
    )


def attach_transition_shared_memory(config: dict[str, Any]) -> TransitionSharedMemory:
    attached: list[shared_memory.SharedMemory] = []

    def attach(key: str) -> tuple[shared_memory.SharedMemory, Any]:
        shm, array = attach_array(config, key)
        attached.append(shm)
        return shm, array

    try:
        actor_planes_shm, actor_planes = attach("actor_planes")
        actor_scalars_shm, actor_scalars = attach("actor_scalars")
        fast_scalars_shm, fast_scalars = attach("fast_scalars")
        critic_planes_shm, critic_planes = attach("critic_planes")
        critic_scalars_shm, critic_scalars = attach("critic_scalars")
        actions_shm, actions = attach("actions")
        k_shm, k = attach("k")
        order_shm, order = attach("order")
        vp_shm, vp = attach("vp")
        threshold_raw_shm, threshold_raw = attach("threshold_raw")
        threshold_int_shm, threshold_int = attach("threshold_int")
        old_logprob_shm, old_logprob = attach("old_logprob")
        value_shm, value = attach("value")
        reward_shm, reward = attach("reward")
        done_shm, done = attach("done")
        round_index_shm, round_index = attach("round_index")
    except Exception:
        for shm in attached:
            shm.close()
        raise

    return TransitionSharedMemory(
        actor_planes_shm=actor_planes_shm,
        actor_scalars_shm=actor_scalars_shm,
        fast_scalars_shm=fast_scalars_shm,
        critic_planes_shm=critic_planes_shm,
        critic_scalars_shm=critic_scalars_shm,
        actions_shm=actions_shm,
        k_shm=k_shm,
        order_shm=order_shm,
        vp_shm=vp_shm,
        threshold_raw_shm=threshold_raw_shm,
        threshold_int_shm=threshold_int_shm,
        old_logprob_shm=old_logprob_shm,
        value_shm=value_shm,
        reward_shm=reward_shm,
        done_shm=done_shm,
        round_index_shm=round_index_shm,
        actor_planes=actor_planes,
        actor_scalars=actor_scalars,
        fast_scalars=fast_scalars,
        critic_planes=critic_planes,
        critic_scalars=critic_scalars,
        actions=actions,
        k=k,
        order=order,
        vp=vp,
        threshold_raw=threshold_raw,
        threshold_int=threshold_int,
        old_logprob=old_logprob,
        value=value,
        reward=reward,
        done=done,
        round_index=round_index,
    )


def attach_array(config: dict[str, Any], key: str) -> tuple[shared_memory.SharedMemory, Any]:
    import numpy as np

    spec = config[key]
    shm = shared_memory.SharedMemory(name=str(spec["name"]))
    shape = tuple(int(value) for value in spec["shape"])
    array = np.ndarray(shape, dtype=np.dtype(str(spec["dtype"])), buffer=shm.buf)
    return shm, array


def shared_array_config(shm: shared_memory.SharedMemory, array: Any) -> dict[str, Any]:
    return {
        "name": shm.name,
        "shape": tuple(int(value) for value in array.shape),
        "dtype": str(array.dtype),
    }
