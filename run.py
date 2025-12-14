import os

import numpy as np
import random
import torch

from pacman_module.pacman import runGame
from pacman_module.ghostAgents import SmartyGhost

from architecture import PacmanNetwork
from pacmanagent import PacmanAgent


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(50)

# input_size = 35
pacman_model = "pacman_model.pth"
model = PacmanNetwork()

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
