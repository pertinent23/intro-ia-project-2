import torch
import numpy as np

from pacman_module.game import Agent, Directions
from pacman_module.pacman import GameState
from data import state_to_tensor, INDEX_TO_ACTION_MAP

class PacmanAgent(Agent):
    def __init__(self, model):
        """
        Initialise l'agent avec un modèle entraîné.
        """
        super().__init__()
        self.model = model
        # On met le modèle en mode "évaluation" (fige les poids)
        self.model.eval() 

    def get_action(self, state: GameState):
        """
        Décide de la prochaine action.
        """
        # Récupération des actions légales (règles du jeu)
        legal_actions = state.getLegalActions()
        
        # Préparation de la donnée pour le réseau
        # On transforme l'état en tenseur
        x = state_to_tensor(state)
        # On ajoute une dimension "batch" de taille 1 car le réseau attend [Batch, Features]
        # x passe de [9] à [1, 9]
        x = x.unsqueeze(0)
        
        # 3. Prédiction du réseau
        with torch.no_grad(): # Pas de gradients nécessaires pour jouer
            logits = self.model(x) # Sortie brute du réseau (scores)
            
        # 4. Sélection de la meilleure action LÉGALE
        # Le réseau nous donne 5 scores. On ne garde que ceux des actions possibles.
        best_action = Directions.STOP
        best_score = -float('inf') # On part de -l'infini
        
        # On parcourt les 5 possibilités du réseau
        for idx in range(5):
            action_candidate = INDEX_TO_ACTION_MAP[idx]
            
            # Si cette action est autorisée par le jeu...
            if action_candidate in legal_actions:
                # On regarde le score donné par le réseau
                score = logits[0][idx].item()
                
                # Si c'est le meilleur score vu jusqu'ici, on garde l'action
                if score > best_score:
                    best_score = score
                    best_action = action_candidate
                    
        return best_action