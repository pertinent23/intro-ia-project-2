"""
Extraction des features et Dataset pour Pacman.

Toutes les features sont :
- normalisées
- informatives
- adaptées à l'imitation learning

Respecte la norme PEP-8.
"""

from collections import deque
import pickle
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from pacman_module.game import Directions, Actions
from pacman_module.pacman import GameState


ACTION_TO_INDEX = {
    Directions.NORTH: 0,
    Directions.SOUTH: 1,
    Directions.EAST: 2,
    Directions.WEST: 3,
    Directions.STOP: 4,
}

INDEX_TO_ACTION_MAP = {v: k for k, v in ACTION_TO_INDEX.items()}


# ------------------------------------------------------------------
# Utilitaires
# ------------------------------------------------------------------

def bfs_distance(
    start: Tuple[int, int],
    targets: List[Tuple[int, int]],
    walls,
) -> int:
    """Calcule la distance BFS vers l'une des cibles."""
    if not targets:
        return 999

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) in targets:
            return dist

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not walls[nx][ny] and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))

    return 999


def proximity_score(
    pos: Tuple[int, int],
    targets: List[Tuple[int, int]],
    walls,
) -> float:
    """Retourne un score de proximité normalisé."""
    dist = bfs_distance(pos, targets, walls)
    return 0.0 if dist >= 999 else 1.0 / (dist + 1.0)


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def state_to_tensor(state: GameState) -> torch.Tensor:
    """
    Convertit un GameState Pacman en vecteur de 35 features.
    """
    pac_x, pac_y = map(int, state.getPacmanPosition())
    pacman_pos = (pac_x, pac_y)

    walls = state.getWalls()
    food = state.getFood().asList()
    capsules = state.getCapsules()
    ghost_states = state.getGhostStates()
    maze_width, maze_height = walls.width, walls.height

    normal_ghosts = []
    scared_ghosts = []
    scared_timer_sum = 0

    for ghost in ghost_states:
        pos = tuple(map(int, ghost.getPosition()))
        pos = ghost.getPosition()
        g_info = {
            'pos': (int(pos[0]), int(pos[1])),
            'dir': ghost.getDirection(),
            'scared_timer': ghost.scaredTimer
        }
        if ghost.scaredTimer > 0:
            scared_ghosts.append(g_info)
            scared_timer_sum += ghost.scaredTimer
        else:
            normal_ghosts.append(g_info)

    avg_scared_timer = (
        scared_timer_sum / max(1, len(scared_ghosts)) / 40.0
    )

    features: List[int] = []

    # ---------------- Directionnelles (20) ----------------
    for action in [
        Directions.NORTH,
        Directions.SOUTH,
        Directions.EAST,
        Directions.WEST,
    ]:
        dx, dy = Actions.directionToVector(action)
        next_pos = (
            int(pac_x + dx),
            int(pac_y + dy)
        )

        if walls[next_pos[0]][next_pos[1]]:
            features.extend([1.0, 0.0, 0.0, 0.0, 0.0])
        else:
            features.append(0.0)
            features.append(proximity_score(next_pos, food, walls))
            features.append(proximity_score(next_pos, capsules, walls))
            features.append(proximity_score(next_pos, scared_ghosts, walls))
            features.append(proximity_score(next_pos, normal_ghosts, walls))

    # ---------------- Fantômes (10) ----------------
    def get_closest_ghost_features(ghost_list):
        if len(ghost_list) < 1:
            return [0.0] * 5

        closest_ghost = min(
            ghost_list,
            key=lambda g: bfs_distance(pacman_pos, g['pos'], walls)
        )

        dist = bfs_distance(pacman_pos, closest_ghost['pos'], walls)

        proximity = 1.0 / (dist + 1.0)
        dx = (closest_ghost['pos'][0] - pacman_pos[0]) / maze_width
        dy = (closest_ghost['pos'][1] - pacman_pos[1]) / maze_height

        vx, vy = Actions.directionToVector(closest_ghost['dir'])

        return [proximity, dx, dy, vx, vy]

    features.extend(get_closest_ghost_features(normal_ghosts))
    features.extend(get_closest_ghost_features(scared_ghosts))

    # ---------------- Globales (5) ----------------
    features.append(avg_scared_timer)
    features.append(len(scared_ghosts) / max(1, len(ghost_states)))
    features.append(proximity_score(pacman_pos, capsules, walls))
    features.append(
        len(food) / state.data.layout.totalFood
    )
    features.append(1.0)  # biais

    return torch.tensor(features, dtype=torch.float32)


class PacmanDataset(Dataset):
    """Dataset PyTorch pour l'imitation learning Pacman."""

    def __init__(self, path: str) -> None:
        with open(path, "rb") as file:
            data = pickle.load(file)

        self.inputs: List[torch.Tensor] = []
        self.labels: List[int] = []

        for state, action in data:
            if action != Directions.STOP:
                self.inputs.append(state_to_tensor(state))
                self.labels.append(ACTION_TO_INDEX[action])

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], torch.tensor(self.labels[index])
