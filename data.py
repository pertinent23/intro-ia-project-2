import pickle
import torch

from torch.utils.data import Dataset
from collections import deque

from pacman_module.game import Directions
from pacman_module.pacman import GameState


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


def state_to_tensor(state: GameState):
    """
    Transforme un objet GameState complexe en un vecteur de nombres (Tensor)
    compréhensible par le réseau de neurones.
    
    Stratégie "Action-Centric" :
    Pour chaque direction (N, S, E, O), on regarde :
    1. Y a-t-il un mur ?
    2. À quelle distance est la nourriture si je vais par là ?
    3. À quelle distance est le danger si je vais par là ?
    """
    pacman_pos = state.getPacmanPosition()
    # On s'assure d'avoir des entiers pour les index de grille
    pacman_pos = (int(pacman_pos[0]), int(pacman_pos[1]))
    
    walls = state.getWalls()
    food = state.getFood()
    capsules = state.getCapsules()
    ghost_states = state.getGhostStates()
    
    grid_width = walls.width
    grid_height = walls.height
    
    features = []
    
    # Constante pour normaliser les distances (aide le réseau à converger)
    MAX_DIST_NORM = 50.0 
    
    # La distance au delà de laquelle le fantome n'est plus un danger
    MAX_DIST_GHOST_NORM = 15.0
        
    # Ordre fixe des directions à analyser : Nord, Sud, Est, Ouest
    # Cela correspond souvent aux index 0, 1, 2, 3 de sortie du réseau
    scan_directions = []
    
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue # On ignore la position de Pacman
            scan_directions.append((dx, dy))
    
    for dx, dy in scan_directions:
        # La case que l'on vise
        next_x = pacman_pos[0] + dx
        next_y = pacman_pos[1] + dy
        
        features.append(dx) # is_wall
        features.append(dy) # is_wall
        
        # --- VÉRIFICATION DES LIMITES ---
        if not (0 <= next_x < grid_width and 0 <= next_y < grid_height):
            # La case est hors de la carte, on la traite comme un mur avec des valeurs par défaut
            features.append(1.0) # is_wall
            features.append(1.0) # food_dist (très loin)
            features.append(0.0) # danger (fantôme)
            features.append(0.0) # capsule
            features.append(0.0) # capsule
            continue
        
        # --- FEATURE A : MUR (1 valeur) ---
        # Si c'est un mur, 1.0, sinon 0.0
        is_wall = walls[next_x][next_y]
        features.append(1.0 if is_wall else 0.0)
        
        # Si c'est un mur, inutile de calculer des chemins complexes, on met des valeurs par défaut
        if is_wall:
            features.append(1.0) # Nourriture considérée "très loin"
            features.append(0.0) # Danger (fantôme) considéré "nul"
            features.append(0.0) # Le fantôme ne peut pas être atteind dans son état éffrayé
            features.append(0.0) # Capsule considérée "nulle"
            continue
            
        # --- FEATURE B : NOURRITURE (1 valeur) ---
        food_list = food.asList()
        if len(food_list) > 0:
            # Optimisation: On trouve d'abord le candidat le plus proche en "vol d'oiseau" (Manhattan)
            # pour éviter de lancer 50 BFS gourmands en calculs.
            closest_food_manhattan = min(food_list, key=lambda f: abs(f[0]-next_x) + abs(f[1]-next_y))
            
            # On calcule la vraie distance labyrinthe vers ce candidat
            dist_food = get_maze_distance((next_x, next_y), closest_food_manhattan, walls)
            
            # Normalisation : 0.0 (sur place) à 1.0 (très loin)
            features.append(min(dist_food, MAX_DIST_NORM) / MAX_DIST_NORM)
        else:
            # Plus de nourriture sur le terrain (victoire proche)
            features.append(0.0)

        # --- FEATURE C : FANTÔMES (1 valeur) ---
        if len(ghost_states) > 0:
            # Trouver le fantôme le plus proche
            closest_ghost = min(ghost_states, key=lambda g: abs(g.getPosition()[0]-next_x) + abs(g.getPosition()[1]-next_y))
            ghost_pos_int = (int(closest_ghost.getPosition()[0]), int(closest_ghost.getPosition()[1]))
            
            dist_ghost = get_maze_distance((next_x, next_y), ghost_pos_int, walls)
            
            # On normalise la distance (ex: sur une base de 15 pas)
            # Si dist=0 (sur nous), norm=1. Si dist=15, norm=0.
            proximity_score = max(0, 1.0 - (dist_ghost / MAX_DIST_GHOST_NORM))
            
            if closest_ghost.scaredTimer > 0:
                # CAS 1 : Fantôme effrayé (Mangeable)
                # On veut être proche -> score négatif ou feature dédiée. 
                # Ici on inverse : une grande proximité devient une "invitation" (valeur négative)
                features.append(-proximity_score)
            else:
                # CAS 2 : Fantôme normal (Danger)
                # Une grande proximité est un grand danger (valeur positive forte)
                features.append(proximity_score)
                
            features.append(1.0 if (dist_ghost < closest_ghost.scaredTimer and closest_ghost.scaredTimer > 0) else 0.0)
                
        else:
            features.append(0.0)
            features.append(0.0)
            
        
        # --- FEATURE D : Capsules (1 valeur) ---
        if len(capsules) > 0:
            # Optimisation: On trouve d'abord le candidat le plus proche en "vol d'oiseau" (Manhattan)
            closest_capsule_manhattan = min(capsules, key=lambda c: abs(c[0]-next_x) + abs(c[1]-next_y))
            
            # On calcule la vraie distance labyrinthe vers ce candidat
            dist_capsule = get_maze_distance((next_x, next_y), closest_capsule_manhattan, walls)
            
            # Normalisation : 0.0 (sur place) à 1.0 (très loin)
            features.append(min(dist_capsule, MAX_DIST_NORM) / MAX_DIST_NORM)
        else:
            # Pas de capsule sur le terrain
            features.append(0.0)

    # --- FEATURES GLOBALES (Optionnel mais utile) ---
    
    # Timer d'effroi global (normalisé) : Indique si c'est le moment d'attaquer
    total_scared_timer = sum([g.scaredTimer for g in ghost_states])
    features.append(min(total_scared_timer, 40) / 40.0)
    
    # --- FEATURE F : FANTÔMES (40 valeur) ---
    sorted_ghosts = sorted(ghost_states, key=lambda g: get_maze_distance(pacman_pos, (int(g.getPosition()[0]), int(g.getPosition()[1])), walls))
    for ghost_index in range(4):
        if ghost_index < len(sorted_ghosts):
            gstate = sorted_ghosts[ghost_index]
            dist_ghost_pacman = get_maze_distance(pacman_pos, gstate.getPosition(), walls)
            dist_ghost = max(0, 1.0 - (dist_ghost_pacman / MAX_DIST_GHOST_NORM))
            features.append(dist_ghost) 
            if gstate.scaredTimer > dist_ghost_pacman:
                features.append(gstate.scaredTimer - dist_ghost_pacman)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
            features.append(0.0)

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