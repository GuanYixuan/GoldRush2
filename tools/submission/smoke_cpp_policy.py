#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.duel import DuelConfig, DuelOrderMode, run_duel
from simulator.observation.sdk import GameInput
from simulator.types import GameOutput
from training.opponents import EpisodeContext
from training.opponents.runners import build_runner
from training.scripts.train_ppo import opponent_spec_from_name


GRID_SIZE = 17
MAX_NPCS = 7
MOVE_BUDGET = 6
REGION_COUNT = 5


class CPosition(ctypes.Structure):
    _fields_ = [("row", ctypes.c_int), ("col", ctypes.c_int)]


class CNpcInfo(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("pos", CPosition)]


class CRegionStat(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("enter", ctypes.c_int),
        ("leave", ctypes.c_int),
        ("gold_generated", ctypes.c_int),
        ("gold_collected", ctypes.c_int),
        ("gold_remaining", ctypes.c_int),
        ("occupants", ctypes.c_int),
    ]


class CSnapshot(ctypes.Structure):
    _fields_ = [("window_begin", ctypes.c_int), ("window_end", ctypes.c_int), ("regions", CRegionStat * REGION_COUNT)]


class CGameInput(ctypes.Structure):
    _fields_ = [
        ("round", ctypes.c_int),
        ("grid", (ctypes.c_int * GRID_SIZE) * GRID_SIZE),
        ("my_units", CPosition * 2),
        ("my_units_gold", ctypes.c_int * 2),
        ("gold_opp", ctypes.c_int),
        ("visible_enemies", CPosition * 2),
        ("num_visible_npcs", ctypes.c_int),
        ("visible_npcs", CNpcInfo * MAX_NPCS),
        ("snapshot_valid", ctypes.c_int),
        ("snapshot", CSnapshot),
    ]


class CGameOutput(ctypes.Structure):
    _fields_ = [
        ("actions", ctypes.c_int * MOVE_BUDGET),
        ("k", ctypes.c_int),
        ("order", ctypes.c_int),
        ("vp", ctypes.c_int),
    ]


class CppPolicy:
    def __init__(self, so_path: Path) -> None:
        self.lib = ctypes.CDLL(str(so_path.resolve()))
        self.lib.moveDecision.argtypes = [ctypes.POINTER(CGameInput)]
        self.lib.moveDecision.restype = CGameOutput
        self.lib.policyRuntimeReady.restype = ctypes.c_int

    def start_episode(self, *, player_id: int) -> None:
        _ = player_id

    def ready(self) -> bool:
        return bool(self.lib.policyRuntimeReady())

    def __call__(self, game_input: GameInput) -> GameOutput:
        c_input = _to_c_input(game_input)
        c_output = self.lib.moveDecision(ctypes.byref(c_input))
        return GameOutput(
            actions=tuple(int(c_output.actions[i]) for i in range(MOVE_BUDGET)),
            k=int(c_output.k),
            order=int(c_output.order),
            vp=int(c_output.vp),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a generated C++ GoldRush policy .so.")
    parser.add_argument("--so", type=Path, required=True)
    parser.add_argument("--map-id", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026081001)
    parser.add_argument("--round-count", type=int, default=20)
    parser.add_argument("--agent-player-id", type=int, default=1, choices=(1, 2))
    parser.add_argument("--opponent", default="fast_probe_v3_bfs")
    args = parser.parse_args()

    policy = CppPolicy(args.so)
    if not policy.ready():
        raise RuntimeError("C++ policy runtime is not ready; check local LD_LIBRARY_PATH for libonnxruntime")

    opponent_spec = opponent_spec_from_name(args.opponent)
    opponent = build_runner(opponent_spec)
    agent_player_id = int(args.agent_player_id)
    opponent_player_id = 2 if agent_player_id == 1 else 1
    policy.start_episode(player_id=agent_player_id)
    opponent.reset(
        int(args.seed),
        EpisodeContext(
            player_id=opponent_player_id,
            opponent_id=agent_player_id,
            map_id=int(args.map_id),
            seed=int(args.seed),
            tags=("cpp_policy_smoke",),
        ),
    )
    players = {agent_player_id: policy, opponent_player_id: opponent}
    result = run_duel(
        players[1],
        players[2],
        config=DuelConfig(
            episode=EpisodeConfig(rules=RulesConfig(round_count=int(args.round_count)), seed=int(args.seed), map_id=int(args.map_id)),
            order_mode=DuelOrderMode.AGENT_AFTER_OPPONENT,
            agent_player_id=agent_player_id,
        ),
        record_replay=False,
    )
    scores = {
        "agent_net_gold": result.state.players[agent_player_id].net_gold,
        "opponent_net_gold": result.state.players[opponent_player_id].net_gold,
        "agent_vision_spent": result.state.players[agent_player_id].vision_spent,
        "winner_id": result.game_result.winner_id,
    }
    scores["margin"] = scores["agent_net_gold"] - scores["opponent_net_gold"]
    scores["agent_won"] = result.game_result.winner_id == agent_player_id
    print(json.dumps(scores, ensure_ascii=True, sort_keys=True))


def _to_c_input(game_input: GameInput) -> CGameInput:
    c_input = CGameInput()
    c_input.round = int(game_input.round)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            c_input.grid[row][col] = int(game_input.grid[row][col])
    for index, (row, col) in enumerate(game_input.my_units):
        c_input.my_units[index] = CPosition(int(row), int(col))
    for index, gold in enumerate(game_input.my_units_gold):
        c_input.my_units_gold[index] = int(gold)
    c_input.gold_opp = int(game_input.gold_opp)
    for index, (row, col) in enumerate(game_input.visible_enemies[:2]):
        c_input.visible_enemies[index] = CPosition(int(row), int(col))
    for index in range(len(game_input.visible_enemies), 2):
        c_input.visible_enemies[index] = CPosition(-1, -1)
    visible_npcs = list(game_input.visible_npcs[:MAX_NPCS])
    c_input.num_visible_npcs = len(visible_npcs)
    for index, npc in enumerate(visible_npcs):
        c_input.visible_npcs[index] = CNpcInfo(int(npc.id), CPosition(int(npc.row), int(npc.col)))
    for index in range(len(visible_npcs), MAX_NPCS):
        c_input.visible_npcs[index] = CNpcInfo(0, CPosition(-1, -1))

    c_input.snapshot_valid = 1 if game_input.snapshot_valid and game_input.snapshot is not None else 0
    if game_input.snapshot is not None:
        c_input.snapshot.window_begin = int(game_input.snapshot.window_begin)
        c_input.snapshot.window_end = int(game_input.snapshot.window_end)
        for index, region in enumerate(game_input.snapshot.regions[:REGION_COUNT]):
            c_input.snapshot.regions[index] = CRegionStat(
                int(region.id),
                int(region.enter),
                int(region.leave),
                int(region.gold_generated),
                int(region.gold_collected),
                int(region.gold_remaining),
                int(region.occupants),
            )
    return c_input


if __name__ == "__main__":
    main()
