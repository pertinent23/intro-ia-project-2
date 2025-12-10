import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from architecture import PacmanNetwork
from data import PacmanDataset

class Pipeline(nn.Module):
    def __init__(self, path):
        """
        Initialise le pipeline d'entraînement.
        """
        super().__init__()

        # Chargement du Dataset
        full_dataset = PacmanDataset(path)
        
        # Séparation Train / Validation (80% / 20%)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # Initialisation du modèle
        # On récupère dynamiquement la taille de l'input (nombre de features)
        input_size = full_dataset[0][0].shape[0]
        self.model = PacmanNetwork(input_size)

        # Fonction de coût et Optimiseur
        # CrossEntropyLoss est standard pour la classification
        self.criterion = nn.CrossEntropyLoss()
        # Adam est un excellent optimiseur par défaut
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Scheduler pour ajuster le learning rate en fonction de la perte
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    def train(self, epochs=120, batch_size=64):
        print(f"Début de l'entraînement sur {len(self.train_dataset)} exemples...")
        
        # DataLoader permet de créer des "batchs" (paquets) de données
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            self.model.train() # Mode entraînement (active le dropout si présent)
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

            # --- Validation (Test sur données non vues) ---
            val_acc = self.evaluate(val_loader)

            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

            # Mise à jour du scheduler avec la perte moyenne de l'époque
            self.scheduler.step(avg_loss)

        # Sauvegarde du modèle final
        torch.save(self.model.state_dict(), "pacman_model.pth")
        print("Modèle sauvegardé dans pacman_model.pth !")

    def evaluate(self, loader):
        """
        Évalue le modèle sans l'entraîner (pas de gradient).
        """
        self.model.eval() # Mode évaluation
        correct = 0
        total = 0
        with torch.no_grad(): # Désactive le calcul des gradients pour aller plus vite
            for inputs, labels in loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()