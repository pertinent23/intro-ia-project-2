import torch.nn as nn

class PacmanNetwork(nn.Module):
    """
    Architecture neuronale pour Pacman (MLP).
    """
    def __init__(self, input_size, num_actions=5):
        """
        Initialise les couches du réseau.
        
        Arguments:
            input_size (int): La taille du vecteur de features (ex: 9).
            num_actions (int): Le nombre d'actions possibles (5).
        """
        super().__init__()
        
        # Définition du réseau séquentiel (couche après couche)
        self.net = nn.Sequential(
            # Couche d'entrée -> Couche cachée 1
            # On projette les features vers 128 neurones
            nn.Linear(input_size, 256),
            
            # Fonction d'activation Leaky ReLU
            # Elle introduit de la non-linéarité pour apprendre des fonctions complexes
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            
            # Couche cachée 1 -> Couche cachée 2
            # On réduit la dimensionnalité pour forcer le réseau à synthétiser l'info
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            # Couche cachée 2 -> Couche cachée 3
            # On réduit la dimensionnalité pour forcer le réseau à synthétiser l'info
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            
            # Couche cachée 5 -> Sortie
            # On projette vers les 5 actions possibles
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        """
        Passe les données (x) à travers le réseau pour obtenir les scores.
        """
        return self.net(x)