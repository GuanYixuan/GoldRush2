"""Symmetry observation probe, layout A.

The strategy identifies the current public map from visible static obstacles,
moves each unit toward a fixed observation anchor, and buys 9x9 vision every
round. It intentionally ignores NPCs, gold, bombs, and enemy units.
"""

GRID_SIZE = 17
WALL = -1
FOG = -5

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

MAX_UNIT_STEPS = 3
K_SPLIT = 3
ORDER_UNIT0_FIRST = 0
VISION_9X9 = 2

LAYOUT_NAME = "A"

MAP_OBSTACLES = {
    1: frozenset(
        [
            (0, 3), (0, 13),
            (2, 1), (2, 2), (2, 14), (2, 15),
            (3, 0), (3, 3), (3, 13), (3, 16),
            (4, 4), (4, 12),
            (5, 5), (5, 11),
            (6, 3), (6, 13),
            (7, 7), (7, 9),
            (8, 4), (8, 6), (8, 10), (8, 12),
            (9, 7), (9, 9),
            (10, 3), (10, 13),
            (11, 5), (11, 11),
            (12, 4), (12, 12),
            (13, 0), (13, 3), (13, 13), (13, 16),
            (14, 1), (14, 2), (14, 14), (14, 15),
            (16, 3), (16, 13),
        ]
    ),
    2: frozenset(
        [
            (2, 2), (2, 6), (2, 10), (2, 14),
            (4, 4), (4, 8), (4, 12),
            (6, 2), (6, 6), (6, 10), (6, 14),
            (8, 4), (8, 12),
            (10, 2), (10, 6), (10, 10), (10, 14),
            (12, 4), (12, 8), (12, 12),
            (14, 2), (14, 6), (14, 10), (14, 14),
        ]
    ),
}

TARGETS = {
    "P1": ((4, 3), (12, 13)),
    "P2": ((4, 11), (13, 4)),
}

DIRECTIONS = (
    (UP, -1, 0),
    (DOWN, 1, 0),
    (LEFT, 0, -1),
    (RIGHT, 0, 1),
)


def _as_pos(value):
    return int(value[0]), int(value[1])


def _detect_side(my_units):
    unit0 = _as_pos(my_units[0])
    unit1 = _as_pos(my_units[1])
    return "P1" if unit0[1] < unit1[1] else "P2"


def _map_mismatch_count(grid, map_id):
    obstacles = MAP_OBSTACLES[map_id]
    mismatches = 0
    observed = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            value = int(grid[r][c])
            if value == FOG:
                continue
            observed += 1
            is_wall = value == WALL
            should_be_wall = (r, c) in obstacles
            if is_wall != should_be_wall:
                mismatches += 1
    return mismatches, observed


def _detect_map(grid):
    scores = {}
    for map_id in MAP_OBSTACLES:
        scores[map_id] = _map_mismatch_count(grid, map_id)
    best_id, best_score = min(scores.items(), key=lambda item: item[1])
    other_scores = [score for mid, score in scores.items() if mid != best_id]
    if best_score[1] == 0 or (other_scores and best_score[0] == other_scores[0][0]):
        return None
    return best_id


def _shortest_actions(start, target, obstacles):
    if start == target:
        return []
    queue = [start]
    head = 0
    previous = {start: (None, STAY)}
    while head < len(queue):
        row, col = queue[head]
        head += 1
        for action, dr, dc in DIRECTIONS:
            nxt = (row + dr, col + dc)
            nr, nc = nxt
            if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
                continue
            if nxt in obstacles or nxt in previous:
                continue
            previous[nxt] = ((row, col), action)
            if nxt == target:
                queue = []
                break
            queue.append(nxt)
    if target not in previous:
        return []
    actions = []
    cur = target
    while cur != start:
        cur, action = previous[cur]
        actions.append(action)
    actions.reverse()
    return actions


def _unit_actions(start, target, obstacles):
    actions = _shortest_actions(start, target, obstacles)[:MAX_UNIT_STEPS]
    while len(actions) < MAX_UNIT_STEPS:
        actions.append(STAY)
    return actions


class Player:
    def __init__(self):
        self.side = None
        self.map_id = None

    def MoveDecision(self, game_input):
        if self.side is None:
            self.side = _detect_side(game_input.my_units)
        if self.map_id is None:
            self.map_id = _detect_map(game_input.grid)

        actions = [STAY] * 6
        if self.side in TARGETS and self.map_id in MAP_OBSTACLES:
            obstacles = MAP_OBSTACLES[self.map_id]
            targets = TARGETS[self.side]
            unit0 = _as_pos(game_input.my_units[0])
            unit1 = _as_pos(game_input.my_units[1])
            actions = (
                _unit_actions(unit0, targets[0], obstacles)
                + _unit_actions(unit1, targets[1], obstacles)
            )

        return actions + [K_SPLIT, ORDER_UNIT0_FIRST, VISION_9X9]

