import pickle

import torch
import numpy as np
from torch.utils.data import Dataset

from pacman_module.pacman import GameState
from pacman_module.pacman import Directions

# Dictionnaire pour convertir les mots (North, South...) en chiffres (0, 1...)
ACTION_TO_INDEX = {
    Directions.NORTH: 0,
    Directions.SOUTH: 1,
    Directions.EAST: 2,
    Directions.WEST: 3,
    Directions.STOP: 4,
}

INDEX_TO_ACTION_MAP = {a: i for i, a in ACTION_TO_INDEX.items()}


def state_to_tensor(state: GameState):
    """
    Build the input of your network.
    We encourage you to do some clever feature engineering here!

    Returns:
        A tensor of features representing the state

    Arguments:
        state: a GameState object
    """
    
    pacman_pos = state.getPacmanPosition()
    ghost_states = state.getGhostStates()
    food_grid = state.getFood()
    walls = state.getWalls()
    
    features = []
    
    x, y = int(pacman_pos[0]), int(pacman_pos[1])
    
    # Pour chaque direction, 1 s'il y a un mur, 0 sinon.
    # Cela aide le réseau à apprendre les règles de physique (ne pas foncer dans un mur).
    # Note: walls[x][y] retourne True s'il y a un mur.
    
    features.append(1 if walls[x][y+1] else 0) # North
    features.append(1 if walls[x][y-1] else 0) # South
    features.append(1 if walls[x+1][y] else 0) # East
    features.append(1 if walls[x-1][y] else 0) # West
    
    # On cherche la nourriture active la plus proche.
    food_list = food_grid.asList()
    if len(food_list) > 0:
        # On calcule les distances vers tous les points de nourriture
        distances = [abs(fx - x) + abs(fy - y) for fx, fy in food_list]
        min_idx = np.argmin(distances)
        closest_food = food_list[min_idx]
        # On normalise les distances (diviser par la taille du board aide l'apprentissage)
        # Ici on simplifie en donnant la direction relative (dx, dy)
        features.append((closest_food[0] - x) / walls.width)  # Distance X relative
        features.append((closest_food[1] - y) / walls.height) # Distance Y relative
    else:
        features.append(0)
        features.append(0)
    
    # Fantôme le plus proche (3 valeurs) ---
    if len(ghost_states) > 0:
        ghost_positions = [g.getPosition() for g in ghost_states]
        # Distances Manhattan vers les fantômes
        dists = [abs(gx - x) + abs(gy - y) for gx, gy in ghost_positions]
        min_idx = np.argmin(dists)
        closest_ghost = ghost_states[min_idx]
        gx, gy = closest_ghost.getPosition()
        
        # Distance relative X et Y normalisée
        features.append((gx - x) / walls.width)
        features.append((gy - y) / walls.height)
        
        # Est-ce que le fantôme est effrayé (mangeable) ? 
        # 1 si oui, 0 si non. C'est crucial pour savoir si on fuit ou si on attaque.
        is_scared = 1 if closest_ghost.scaredTimer > 0 else 0
        features.append(is_scared)
    else:
        features.append(0)
        features.append(0)
        features.append(0)
    
    return torch.sensor(features, dtype=torch.float32)


class PacmanDataset(Dataset):
    def __init__(self, path):
        """
        Load and transform the pickled dataset into a format suitable
        for training your architecture.

        Arguments:
            path: The file path to the pickled dataset.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.inputs = []
        self.actions = []

        for s, a in data:
            # On transforme l'état en vecteur de features
            x = state_to_tensor(s)
            
            # On transforme l'action textuelle ('North') en index (0)
            y = ACTION_TO_INDEX[a]
            
            self.inputs.append(x)
            self.actions.append(y)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # On retourne l'input et l'action (convertie en tenseur Long pour les classes)
        return self.inputs[idx], torch.tensor(self.actions[idx], dtype=torch.long)
