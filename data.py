from pacman_module.game import Directions

import pickle
import torch

from torch.utils.data import Dataset
from collections import deque

# Mapping des actions textuelles vers des index numériques pour le réseau
ACTION_TO_IDX = {
    Directions.NORTH: 0,
    Directions.SOUTH: 1,
    Directions.EAST: 2,
    Directions.WEST: 3,
    Directions.STOP: 4
}

# Mapping inverse utile pour le débogage ou l'agent
INDEX_TO_ACTION_MAP = {v: k for k, v in ACTION_TO_IDX.items()}


# --- FONCTIONS UTILITAIRES (Feature Engineering) ---

def get_maze_distance(start, target, walls):
    """
    Calcule la distance réelle (nombre de pas) entre deux points
    en contournant les murs, utilisant un algorithme BFS (Breadth-First Search).
    
    Arguments:
        start (tuple): (x, y) point de départ
        target (tuple): (x, y) point d'arrivée
        walls (Grid): La grille des murs du jeu
        
    Returns:
        int: La distance en nombre de pas. Retourne une valeur arbitraire (ex: 100)
        si la cible est inaccessible.
    """
    if start == target:
        return 0
    
    # File d'attente pour le BFS: stocke ((x, y), distance_actuelle)
    queue = deque([(start, 0)])
    visited = set([start])
    w, h = walls.width, walls.height
    
    while queue:
        (curr_x, curr_y), dist = queue.popleft()
        
        if (curr_x, curr_y) == target:
            return dist
        
        # Explorer les voisins (Nord, Sud, Est, Ouest)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x, next_y = int(curr_x + dx), int(curr_y + dy)
            
            # Vérifier que l'on reste dans la carte et qu'on ne traverse pas un mur
            if 0 <= next_x < w and 0 <= next_y < h:
                if not walls[next_x][next_y] and (next_x, next_y) not in visited:
                    visited.add((next_x, next_y))
                    queue.append(((next_x, next_y), dist + 1))
    
    # Si on ne trouve pas de chemin (cible inatteignable)
    return 100 


def state_to_tensor(state):
    """
    Transforme un objet GameState complexe en un vecteur de nombres (Tensor)
    compréhensible par le réseau de neurones.
    
    Stratégie Hybride :
    - Action-Centric : Évaluation de chaque direction pour la nourriture/capsules.
    - Globale : État des fantômes, vision locale des murs, actions légales et score.
    """
    pacman_pos = state.getPacmanPosition()
    pacman_pos = (int(pacman_pos[0]), int(pacman_pos[1]))
    
    walls = state.getWalls()
    food = state.getFood()
    capsules = state.getCapsules()
    ghost_states = state.getGhostStates()
    
    features = []
    
    MAX_DIST_NORM = 50.0 

    # --- FEATURES "ACTION-CENTRIC" (Évaluation de chaque direction) ---
    scan_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for dx, dy in scan_directions:
        next_x = pacman_pos[0] + dx
        next_y = pacman_pos[1] + dy
        next_pos = (next_x, next_y)
        
        # Feature A : Mur
        is_wall = walls[next_x][next_y]
        features.append(1.0 if is_wall else 0.0)
        
        if is_wall:
            features.append(1.0) # Nourriture considérée "très loin"
            features.append(1.0) # Capsule considérée "très loin"
            continue
            
        # Feature B : Nourriture
        food_list = food.asList()
        if len(food_list) > 0:
            closest_food = min(food_list, key=lambda f: abs(f[0]-next_x) + abs(f[1]-next_y))
            dist_food = get_maze_distance(next_pos, closest_food, walls)
            features.append(min(dist_food, MAX_DIST_NORM) / MAX_DIST_NORM)
        else:
            features.append(0.0)

        # Feature C : Capsules
        if len(capsules) > 0:
            closest_capsule = min(capsules, key=lambda c: abs(c[0]-next_x) + abs(c[1]-next_y))
            dist_capsule = get_maze_distance(next_pos, closest_capsule, walls)
            features.append(min(dist_capsule, MAX_DIST_NORM) / MAX_DIST_NORM)
        else:
            features.append(0.0)

    # --- FEATURES GLOBALES : FANTÔMES ---
    sorted_ghosts = sorted(ghost_states, key=lambda g: get_maze_distance(pacman_pos, (int(g.getPosition()[0]), int(g.getPosition()[1])), walls))

    for i in range(4):
        if i < len(sorted_ghosts):
            ghost = sorted_ghosts[i]
            ghost_pos = (int(ghost.getPosition()[0]), int(ghost.getPosition()[1]))
            dist_ghost = get_maze_distance(pacman_pos, ghost_pos, walls)
            features.append(min(dist_ghost, MAX_DIST_NORM) / MAX_DIST_NORM)
            
            proximity_score = max(0, 1.0 - (dist_ghost / 15.0))
            if ghost.scaredTimer > 5:
                features.append(-proximity_score)
            else:
                features.append(proximity_score)
        else:
            features.append(1.0) 
            features.append(0.0)

    # --- FEATURE GLOBALE : GRILLE DE MURS 5x5 ---
    half = 2
    for dy_grid in range(-half, half + 1):
        for dx_grid in range(-half, half + 1):
            x = int(pacman_pos[0] + dx_grid)
            y = int(pacman_pos[1] + dy_grid)
            if x < 0 or y < 0 or x >= walls.width or y >= walls.height:
                features.append(1.0) # Hors de la carte = considéré comme un mur
            else:
                features.append(1.0 if walls[x][y] else 0.0)
    
    # --- FEATURE GLOBALE : MASQUE D'ACTIONS LÉGALES ---
    legal = state.getLegalPacmanActions()
    legal_mask = [0.0] * len(ACTION_TO_IDX)
    for a in legal:
        idx = ACTION_TO_IDX.get(a)
        if idx is not None:
            legal_mask[idx] = 1.0
    features += legal_mask

    # Conversion finale en Tensor Float32
    return torch.tensor(features, dtype=torch.float32)


# --- CLASSE DATASET ---

class PacmanDataset(Dataset):
    def __init__(self, path):
        """
        Charge les données brutes (pickles) et les convertit en tenseurs
        prêts pour l'entraînement PyTorch.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.inputs = []
        self.actions = []

        print(f"Traitement des données depuis {path}...")
        
        for s, a in data:
            # Feature Engineering : GameState -> Tensor
            x = state_to_tensor(s)
            
            # Label Encoding : 'North' -> 0
            y = ACTION_TO_IDX[a]
            
            self.inputs.append(x)
            self.actions.append(y)
            
        print(f"Chargement terminé : {len(self.inputs)} exemples prêts.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Retourne (features, label)
        # label doit être LongTensor pour CrossEntropyLoss
        return self.inputs[idx], torch.tensor(self.actions[idx], dtype=torch.long)