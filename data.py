"""
Module de gestion des données pour le projet Pacman.
Optimisé pour l'apprentissage 'Survival & Combat' avec features séparées.
"""
from collections import deque

import pickle
import torch
from torch.utils.data import Dataset

from pacman_module.game import Directions, Actions
from pacman_module.pacman import GameState

# --- CONFIGURATION ---
ACTION_TO_IDX = {
    Directions.NORTH: 0,
    Directions.SOUTH: 1,
    Directions.EAST: 2,
    Directions.WEST: 3,
    Directions.STOP: 4
}

INDEX_TO_ACTION_MAP = {v: k for k, v in ACTION_TO_IDX.items()}

def get_maze_distance(start, target, walls):
    """
    Calcule la distance réelle (BFS). Retourne 999 si inaccessible.
    """
    if start == target: return 0
    w, h = walls.width, walls.height
    if not (0 <= target[0] < w and 0 <= target[1] < h) or walls[int(target[0])][int(target[1])]:
        return 999

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        (cx, cy), dist = queue.popleft()
        if (cx, cy) == target:
            return dist

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = int(cx + dx), int(cy + dy)
            if 0 <= nx < w and 0 <= ny < h:
                if not walls[nx][ny] and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
    return 999

def is_trap(start_pos, current_direction, walls, ghosts_pos):
    """
    Détecte les culs-de-sac mortels (Pièges).
    """
    dx, dy = Actions.directionToVector(current_direction)
    curr_x, curr_y = int(start_pos[0] + dx), int(start_pos[1] + dy)
    
    if walls[curr_x][curr_y]: return True

    # Vérification simplifiée de proximité immédiate d'un fantôme
    for g_pos in ghosts_pos:
        if abs(g_pos[0] - curr_x) + abs(g_pos[1] - curr_y) <= 1:
            return True 
    return False

def calculate_score(pos, targets, walls):
    """
    Trouve la cible la plus proche et retourne 1 / (Distance + 1).
    Optimisation: Utilise la distance Manhattan pour pré-sélectionner la cible
    avant de lancer le BFS coûteux.
    """
    if not targets:
        return 0.0
    
    # 1. Heuristique rapide : Trouver la cible la plus proche à vol d'oiseau
    closest_target = min(targets, key=lambda t: abs(t[0]-pos[0]) + abs(t[1]-pos[1]))
    
    # 2. Calcul précis : BFS seulement vers cette cible
    dist = get_maze_distance(pos, closest_target, walls)
    
    if dist >= 999:
        return 0.0
        
    return 1.0 / (dist + 1.0)

def state_to_tensor(state: GameState):
    """
    Extraction de Features Séparées.
    Input Size: 25 Floats
    Structure par direction : [Mur, Food, Capsule, ScaredGhost, Danger]
    """
    pacman_pos = state.getPacmanPosition()
    pacman_pos = (int(pacman_pos[0]), int(pacman_pos[1]))
    walls = state.getWalls()
    food = state.getFood().asList()
    capsules = state.getCapsules()
    ghost_states = state.getGhostStates()

    scared_ghosts_pos = []
    normal_ghosts_pos = []
    scared_timer_val = 0.0

    for g in ghost_states:
        pos = (int(g.getPosition()[0]), int(g.getPosition()[1]))
        if g.scaredTimer > 0:
            scared_ghosts_pos.append(pos)
            scared_timer_val = max(scared_timer_val, g.scaredTimer / 40.0)
        else:
            normal_ghosts_pos.append(pos)

    features = []
    directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

    # --- PARTIE A : ANALYSE PAR DIRECTION (4 dir * 5 features = 20) ---
    for action in directions:
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(pacman_pos[0] + dx), int(pacman_pos[1] + dy)
        next_pos = (next_x, next_y)

        # 1. MUR / PIÈGE
        is_blocked = False
        if walls[next_x][next_y] or is_trap(pacman_pos, action, walls, normal_ghosts_pos):
            is_blocked = True
        
        if is_blocked:
            # Ordre: [Mur, Food, Cap, Scared, Danger]
            # Si mur, tout est 0 sauf l'index mur
            features.extend([1.0, 0.0, 0.0, 0.0, 0.0])
            continue
        
        # Si voie libre : Mur = 0
        features.append(0.0)

        # 2. FOOD SCORE (On veut manger)
        features.append(calculate_score(next_pos, food, walls))

        # 3. CAPSULE SCORE (On veut des pouvoirs)
        features.append(calculate_score(next_pos, capsules, walls))

        # 4. SCARED GHOST SCORE (On veut chasser)
        features.append(calculate_score(next_pos, scared_ghosts_pos, walls))

        # 5. DANGER SCORE (On veut fuir - Fantôme Normal)
        features.append(calculate_score(next_pos, normal_ghosts_pos, walls))


    # --- PARTIE B : GLOBAL INFO (5 features) ---
    
    # 21. Timer Scared
    features.append(scared_timer_val)

    # 22. Ratio Capsules restantes
    features.append(len(capsules) / 4.0 if len(capsules) > 0 else 0.0)

    # 23. Danger immédiat global (Radar de panique)
    features.append(calculate_score(pacman_pos, normal_ghosts_pos, walls))

    # 24. Distance Capsule Sauvetage (Radar de sauvetage)
    features.append(calculate_score(pacman_pos, capsules, walls))
    
    # 25. Biais constant
    features.append(1.0)

    return torch.tensor(features, dtype=torch.float32)


class PacmanDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.inputs = []
        self.actions = []

        print(f"Chargement des données séparées depuis {path}...")
        dropped = 0
        
        for s, a in data:
            if a == Directions.STOP:
                dropped += 1
                continue
            
            self.inputs.append(state_to_tensor(s))
            self.actions.append(ACTION_TO_IDX[a])

        print(f"  - Données valides : {len(self.inputs)}")
        print(f"  - Dimension Input : {self.inputs[0].shape[0]} (Cible: 25)")
        print(f"  - Stops ignorés   : {dropped}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            torch.tensor(self.actions[idx], dtype=torch.long)
        )