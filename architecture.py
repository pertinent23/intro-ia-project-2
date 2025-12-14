"""
Réseau de neurones pour l'imitation d'un expert Pacman.

Architecture multi-branches (late fusion) :
- Branche directionnelle (analyse locale des actions)
- Branche fantômes (menaces / opportunités)
- Branche globale (état général de la partie)

Conçu pour des features tabulaires (pas d'images).
Respecte la norme PEP-8.
"""

import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    Réseau multi-branches pour Pacman.
    Chaque groupe de features est traité séparément avant fusion.
    """

    def __init__(self) -> None:
        """Initialise toutes les couches du réseau."""
        super().__init__()

        # ------------------------------------------------------------------
        # Branche A : Features directionnelles (4 directions × 5 = 20)
        # ------------------------------------------------------------------
        self.direction_branch = nn.Sequential(
            nn.Linear(20, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
        )

        # ------------------------------------------------------------------
        # Branche B : Features liées aux fantômes (10)
        # ------------------------------------------------------------------
        self.ghost_branch = nn.Sequential(
            nn.Linear(10, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
        )

        # ------------------------------------------------------------------
        # Branche C : Informations globales (5)
        # ------------------------------------------------------------------
        self.global_branch = nn.Sequential(
            nn.Linear(5, 32),
            nn.LeakyReLU(),
        )

        # ------------------------------------------------------------------
        # Tête de décision (fusion tardive)
        # ------------------------------------------------------------------
        self.decision_head = nn.Sequential(
            nn.Linear(64 + 32 + 32, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5),  # 5 actions possibles
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Optimisé pour LeakyReLU
            nn.init.kaiming_normal_(
                m.weight,
                mode='fan_out',
                nonlinearity='leaky_relu'
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du réseau.

        Args:
            x (Tensor): Batch de features de taille (N, 35)

        Returns:
            Tensor: Scores pour chaque action (N, 5)
        """
        # Séparation des features par branche
        # dir_features: (N, 20)
        dir_features = x[:, :20]

        # ghost_features: (N, 10)
        ghost_features = x[:, 20:30]

        # global_features: (N, 5)
        global_features = x[:, 30:]

        dir_out = self.direction_branch(dir_features)
        ghost_out = self.ghost_branch(ghost_features)
        global_out = self.global_branch(global_features)

        fused = torch.cat([dir_out, ghost_out, global_out], dim=1)
        return self.decision_head(fused)
