"""Symmetry observation probe for map 3, layout M3-B.

M3-B is the horizontal/vertical complement of M3-A. Alternating both layouts
covers every map 3 cell except the center point (8, 8) across batches.
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

LAYOUT_NAME = "M3B"

MAP_OBSTACLES = {
    3: frozenset(
        [
            (2, 2), (2, 3), (2, 4), (2, 12), (2, 13), (2, 14),
            (3, 2), (3, 3), (3, 4), (3, 12), (3, 13), (3, 14),
            (4, 4), (4, 5), (4, 6), (4, 7), (4, 9), (4, 10), (4, 11), (4, 12),
            (5, 4), (5, 5), (5, 6), (5, 7), (5, 9), (5, 10), (5, 11), (5, 12),
            (6, 4), (6, 5), (6, 6), (6, 7), (6, 9), (6, 10), (6, 11), (6, 12),
            (8, 4), (8, 5), (8, 6), (8, 10), (8, 11), (8, 12),
            (10, 4), (10, 5), (10, 6), (10, 7), (10, 9), (10, 10), (10, 11), (10, 12),
            (11, 4), (11, 5), (11, 6), (11, 7), (11, 9), (11, 10), (11, 11), (11, 12),
            (12, 4), (12, 5), (12, 6), (12, 7), (12, 9), (12, 10), (12, 11), (12, 12),
            (13, 2), (13, 3), (13, 4), (13, 12), (13, 13), (13, 14),
            (14, 2), (14, 3), (14, 4), (14, 12), (14, 13), (14, 14),
        ]
    ),
}

TARGETS = {
    "P1": ((3, 5), (13, 11)),
    "P2": ((4, 13), (12, 3)),
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
    if best_score[1] == 0 or best_score[0] != 0:
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
