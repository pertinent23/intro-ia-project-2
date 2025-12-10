import os

import numpy as np
import random
import torch

from pacman_module.pacman import runGame
from pacman_module.ghostAgents import SmartyGhost

from architecture import PacmanNetwork
from pacmanagent import PacmanAgent


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

input_size = 65
pacman_model = "pacman_model.pth"
model = PacmanNetwork(input_size)
    
# Charger les poids appris
if os.path.exists(pacman_model):
    model.load_state_dict(torch.load(pacman_model, map_location="cpu"))
    print("Modèle chargé avec succès.")
else:
    print("Erreur : pacman_model.pth introuvable. Lancez train.py d'abord.")

model.eval()

pacman_agent = PacmanAgent(model)

score, elapsed_time, nodes = runGame(
    layout_name="test_layout",
    pacman=pacman_agent,
    ghosts=[SmartyGhost(1)],
    beliefstateagent=None,
    displayGraphics=True,
    expout=0.0,
    hiddenGhosts=False,
)

print(f"Score: {score}")
print(f"Computation time: {elapsed_time}")
