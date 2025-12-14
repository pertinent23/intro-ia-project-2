import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader, random_split
from architecture import PacmanNetwork
from data import PacmanDataset


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(50)


class Pipeline(nn.Module):
    def __init__(self, path, model_save_path="pacman_model.pth"):
        """
        Initialise le pipeline d'entraînement.
        """
        super().__init__()
        self.model_save_path = model_save_path
        # Chargement du Dataset
        full_dataset = PacmanDataset(path)
        # Séparation Train / Validation (80% / 20%)
        train_size = int(0.80 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        # Initialisation du modèle
        # On récupère dynamiquement la taille de l'input (nombre de features)
        # input_size = full_dataset[0][0].shape[0]
        self.model = PacmanNetwork()
        # Fonction de coût et Optimiseur
        # CrossEntropyLoss est standard pour la classification
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
        # Adam est un excellent optimiseur par défaut
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.003
        )
        # Scheduler pour ajuster le learning rate en fonction de la perte
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

    def train(self, epochs=300, batch_size=64, patience=15):
        print(f"Début de l'entraînement sur {len(self.train_dataset)} data")

        # DataLoader permet de créer des "batchs" (paquets) de données
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        best_loss = 2
        best_acc = 0.0
        patience_counter = 0
        for epoch in range(epochs):
            # Mode entraînement (active le dropout si présent)
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            # --- Boucle d'apprentissage ---
            for inputs, labels in train_loader:
                # 1. Remise à zéro des gradients
                self.optimizer.zero_grad()

                # 2. Forward pass : Calcul des prédictions
                outputs = self.model(inputs)

                # 3. Calcul de la perte (Loss)
                loss = self.criterion(outputs, labels)

                # 4. Backward pass : Rétropropagation des erreurs
                loss.backward()

                # 5. Mise à jour des poids
                self.optimizer.step()

                # Statistiques
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total
            avg_loss = total_loss / len(train_loader)
            val_acc, val_loss = self.evaluate(val_loader)

            info = f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f}"
            info += f" | Val Loss: {val_loss:.4f} "
            info += f" | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
            print(info)

            # Sauvegarde & Early Stopping
            if val_acc - best_acc >= 0.001 \
                    and abs(val_loss - best_loss) >= 0.0001:
                best_loss = val_loss
                best_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"Enrégistrement avec: {best_acc:.2f}% {best_loss:.4f}")
            else:
                patience_counter += 1

            self.scheduler.step(val_loss)

            if patience_counter >= patience:
                print(f"Early Stopping à l'époque {epoch+1}")
                break
        print(f"Entraînement terminé avec {best_acc:.2f}% {best_loss}")

    def evaluate(self, loader):
        """
        Évalue le modèle sans l'entraîner (pas de gradient).
        """
        # Mode évaluation
        self.model.eval()
        correct = 0
        total_loss = 0
        total = 0
        # Désactive le calcul des gradients pour aller plus vite
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total, total_loss / len(loader)


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
