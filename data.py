import pickle
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from pacman_module.pacman import GameState
from pacman_module.pacman import Directions


# Mapping des directions vers des indices entiers
ACTION_TO_IDX = {
    Directions.NORTH: 0,
    Directions.SOUTH: 1,
    Directions.EAST: 2,
    Directions.WEST: 3,
    Directions.STOP: 4,
}


def _nearest_position(origin: Tuple[int, int], positions: List[Tuple[int, int]]):
    """Retourne la position la plus proche (manhattan) et la distance."""
    if not positions:
        return None, None
    ox, oy = origin
    best = None
    best_d = None
    for (x, y) in positions:
        d = abs(x - ox) + abs(y - oy)
        if best_d is None or d < best_d:
            best_d = d
            best = (x, y)
    return best, best_d


def state_to_tensor(state: GameState) -> torch.FloatTensor:
    """
    Build the input of your network.
    We encourage you to do some clever feature engineering here!

    Returns:
        A tensor of features representing the state

    Arguments:
        state: a GameState object
    """
    # Position de Pacman
    pac_x, pac_y = state.getPacmanPosition()

    walls = state.getWalls()
    width = getattr(walls, "width", None)
    height = getattr(walls, "height", None)
    if width is None or height is None:
        # valeurs de repli raisonnables
        width, height = 1, 1

    # Normalisation des positions
    px = float(pac_x) / float(max(1, width - 1))
    py = float(pac_y) / float(max(1, height - 1))

    features: List[float] = [px, py]

    # Fantômes: on fixe un nombre max pour garder la dimension constante
    max_ghosts = 4
    ghosts = state.getGhostPositions()
    # Recueillir dx,dy et distance normalisée
    for i in range(max_ghosts):
        if i < len(ghosts):
            gx, gy = ghosts[i]
            dx = gx - pac_x
            dy = gy - pac_y
            # normalisation approximative par taille de la carte
            features.append(float(dx) / float(max(1, width)))
            features.append(float(dy) / float(max(1, height)))
            dist = abs(dx) + abs(dy)
            features.append(float(dist) / float(width + height))
        else:
            # pas de fantôme -> distance grande (1)
            features += [0.0, 0.0, 1.0]

    # Nourriture la plus proche
    food_grid = state.getFood()
    try:
        food_positions = food_grid.asList(True)
    except Exception:
        # asList peut ne pas être disponible, essayer autre moyen
        food_positions = []
    nearest_food, food_dist = _nearest_position((pac_x, pac_y), food_positions)
    if nearest_food is not None:
        fx, fy = nearest_food
        features.append(float(fx - pac_x) / float(max(1, width)))
        features.append(float(fy - pac_y) / float(max(1, height)))
        features.append(float(food_dist) / float(width + height))
    else:
        features += [0.0, 0.0, 1.0]

    # Capsule la plus proche
    capsules = state.getCapsules()
    nearest_caps, caps_dist = _nearest_position((pac_x, pac_y), capsules)
    if nearest_caps is not None:
        cx, cy = nearest_caps
        features.append(float(cx - pac_x) / float(max(1, width)))
        features.append(float(cy - pac_y) / float(max(1, height)))
        features.append(float(caps_dist) / float(width + height))
    else:
        features += [0.0, 0.0, 1.0]

    # Grille locale des murs 5x5 centrée sur Pacman
    half = 2
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            x = int(pac_x + dx)
            y = int(pac_y + dy)
            if x < 0 or y < 0 or x >= width or y >= height:
                features.append(1.0)
            else:
                features.append(1.0 if walls[x][y] else 0.0)

    # Masque des actions légales (ordre selon ACTION_TO_IDX)
    legal = state.getLegalPacmanActions()
    legal_mask = [0.0] * len(ACTION_TO_IDX)
    for a in legal:
        idx = ACTION_TO_IDX.get(a)
        if idx is not None:
            legal_mask[idx] = 1.0
    features += legal_mask

    # Score (simple normalisation)
    score = float(state.getScore())
    features.append(score / 100.0)

    return torch.tensor(features, dtype=torch.float32)


class PacmanDataset(Dataset):
    def __init__(self, path: str):
        """
        Load and transform the pickled dataset into a format suitable
        for training your architecture.

        Arguments:
            path: The file path to the pickled dataset.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.inputs: List[torch.FloatTensor] = []
        self.actions: List[torch.LongTensor] = []

        for s, a in data:
            x = state_to_tensor(s)
            if isinstance(a, str):
                label = ACTION_TO_IDX.get(a, ACTION_TO_IDX[Directions.STOP])
            else:
                label = ACTION_TO_IDX.get(a, ACTION_TO_IDX[Directions.STOP])

            self.inputs.append(x)
            self.actions.append(torch.tensor(label, dtype=torch.long))

        # Empiler les tenseurs pour un accès constant en temps
        if len(self.inputs) > 0:
            self.inputs = torch.stack(self.inputs)
            self.actions = torch.stack(self.actions)
        else:
            # cas vide — garder des tensors vides
            self.inputs = torch.empty((0,))
            self.actions = torch.empty((0,), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.actions[idx]
