import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Bloc résiduel pour MLP : permet d'entraîner des réseaux plus profonds
    sans perdre le gradient.
    Structure : Input -> Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Add(Input) -> ReLU
    """
    def __init__(self, hidden_size, dropout_p=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection (magie du ResNet)
        return self.relu(out)

class PacmanNetwork(nn.Module):
    def __init__(self, input_size, num_actions=5):
        super(PacmanNetwork, self).__init__()
        
        # 1. Projection initiale (Expansion)
        # On passe de 25 features à 256 pour capturer toutes les interactions
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # 2. Cœur du réseau : Blocs Résiduels
        # Le réseau peut "réfléchir" plus longtemps sans oublier l'entrée
        self.res_blocks = nn.Sequential(
            ResidualBlock(256, dropout_p=0.3),
            ResidualBlock(256, dropout_p=0.15),
            ResidualBlock(256, dropout_p=0.2)
        )
        
        # 3. Tête de classification (Compression)
        self.output_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        logits = self.output_head(x)
        return logits