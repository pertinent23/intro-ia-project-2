import pickle
from collections import deque

import torch
from torch.utils.data import Dataset

from pacman_module.pacman import GameState
from pacman_module.pacman import Directions
from pacman_module.util import manhattanDistance

def get_bfs_distance(start, target, walls):
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

def state_to_tensor(state: GameState):
    """
    Build the input of your network.
    We encourage you to do some clever feature engineering here!

    Returns:
        A tensor of features representing the state

    Arguments:
        state: a GameState object
    """
    pacman_position = state.getPacmanPosition()
    walls = state.getWalls()
    food_grid = state.getFood()
    capsules = state.getCapsules()
    ghosts = state.getGhostStates()

    DISTANCE_NORMALISATION = 50 #ou height? ou width?
    TIME_NORMALISATION = 40

    features = []

    possible_moves = [(-1,0),(0,1),(1,0),(0,-1)]

    # SPECIFICS FEATURES : FOR EACH POSSIBLE MOVE
    for x, y in possible_moves:
        next_x = pacman_position[0] + x
        next_y = pacman_position[1] + y

        # FEATURE 1 : Is wall
        # Verify if it's a wall
        is_wall = walls[next_x][next_y]

        if is_wall:
            # FOOD : Pas de nourriture proche
            features.append(1.0)
            # CAPSULE : Pas de capsule proche
            features.append(1.0)
            # NUMBER OF EXIT : Pas de sortie
            features.append(0.0)
            # GHOSTS STATES : Pas de fantômes
            features.append(0.0)

        else:
            # If it's wall 1.0, else 0.0
            features.append(1.0 if is_wall else 0.0)

            # FEATURE 2 : CLOSEST FOOD
            food_grid_width = len(food_grid)
            food_grid_height = len(food_grid[0])
            food_position_list = [(x, y) for x in range(food_grid_width) for y in range(food_grid_height) if food_grid[x][y]]
            closest_food = min(food_position_list, key=lambda pos: manhattanDistance(pacman_position, pos))

            distance_closest_food = get_bfs_distance((next_x, next_y), closest_food, walls)

            features.append(min(distance_closest_food, DISTANCE_NORMALISATION)/DISTANCE_NORMALISATION)

            # FEATURE 3 : CLOSEST CAPSULE
            if len(capsules) > 0:
                closest_capsule = min(capsules, key=lambda pos: manhattanDistance(pacman_position, pos))
                features.append(min(closest_capsule, DISTANCE_NORMALISATION)/DISTANCE_NORMALISATION)
            else:
                features.append(0.0)

            # FEATURE 4 : NUMBER OF EXIT
            # Pour éviter de se retrouver dans un cul-de-sac qui obligerait de revenir en arrière
            number_of_exit = 0
            for x, y in possible_moves:
                next_next_x = next_x + x
                next_next_y = next_y + y
                if walls[next_next_x][next_next_y]:
                    number_of_exit += 1
            # Normalisation par 4 car il y a maximum 4 sorties possibles (N,S,W,E)
            features.append(min(number_of_exit,4)/4)

            #FEATURE 5 : GHOST STATES
            if len(ghosts) > 0:
                for ghost in ghosts:
                    ghost_distance = get_bfs_distance(pacman_position, abs(ghost.getPosition()[0]-next_x) + abs(ghost.getPosition()[1]-next_y), walls)
                    features.append(min(ghost_distance,DISTANCE_NORMALISATION)/DISTANCE_NORMALISATION)
                    features.append(min(ghost.scaredTimer, TIME_NORMALISATION)/TIME_NORMALISATION)
            else:
                features.append(0.0)

    # GLOBALS FEATURES
    # FEATURE 6 : NUMBER OF LEFT CAPSULES
    features.append(1.0 if len(capsules) > 0 else 0.0)

    # FEATURE 7 : NUMBER OF LEFT FOOD
    # Si il reste peut de nourriture, ça indique une fin proche
    features.append(state.getNumFood())

    # FEATURE 8 : SUM OF ALL SCARED TIME
    total_scared_timer = sum([g.scaredTimer for g in ghosts])
    features.append(min(total_scared_timer, TIME_NORMALISATION) / TIME_NORMALISATION)

    return torch.tensor(features, dtype=torch.float32)

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
            x = state_to_tensor(s)
            self.inputs.append(x)
            self.actions.append(a)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.actions[idx]
