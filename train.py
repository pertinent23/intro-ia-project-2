import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import random_split

from architecture import PacmanNetwork
from data import PacmanDataset


class Pipeline(nn.Module):
    def __init__(self, path):
        """
        Initialize your training pipeline.

        Arguments:
            path: The file path to the pickled dataset.
        """
        super().__init__()

        self.path = path
        self.dataset = PacmanDataset(self.path)
        self.model = PacmanNetwork(self.dataset[0][0].shape[0])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self):
        print("Beginning of the training of your network...")

        epochs_number = 200
        batch_size = 254
        losses = []
        training_accuracies = []
        verification_accuracies = []

        training_dataset_size = int(0.9 * len(self.dataset))
        verification_dataset_size = len(self.dataset) - training_dataset_size
        training_dataset, verification_dataset = random_split(self.dataset, [training_dataset_size, verification_dataset_size])
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        verification_loader = torch.utils.data.DataLoader(verification_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs_number):
            epcoh_loss = 0
            number_of_examples = 0
            number_of_correct_prediction = 0

            # TRAINING
            self.model.train()
            for inputs, expected_result in training_loader:
                result = self.model(inputs)

                loss = self.criterion(result, expected_result)

                loss.backward()

                self.optimizer.step()

                self.optimizer.zero_grad()

                epcoh_loss += loss.item()

                score, predicted = torch.max(result.data, 1)
                number_of_examples += expected_result.size(0)
                number_of_correct_prediction += (predicted == expected_result).sum().item()

            #VERIFICATION
            number_of_examples_verif = 0
            number_of_correct_prediction_verif = 0

            self.model.eval()
            with torch.no_grad():
                for inputs, expected_result in verification_loader:
                    result = self.model(inputs)
                    score, predicted = torch.max(result.data, 1)
                    number_of_examples_verif += expected_result.size(0)
                    number_of_correct_prediction_verif += (predicted == expected_result).sum().item()

            # STATISTIC
            average_loss = epcoh_loss / len(training_loader)
            training_accuracy = number_of_correct_prediction / number_of_examples * 100
            verification_accuracy = number_of_correct_prediction_verif / number_of_examples_verif * 100
            losses.append(average_loss)
            training_accuracies.append(training_accuracy)
            verification_accuracies.append(verification_accuracy)
            print("[EPOCH]: %i, [LOSS]: %.6f, [TRAINING_ACCURACY]: %.3f, [VERIFICATION_ACCURACY]: %.3f" % (epoch, average_loss, training_accuracy, verification_accuracy))

            if epoch == epochs_number - 1:
                self.plot_metrics(losses, training_accuracies, verification_accuracies, epoch + 1)

        torch.save(self.model.state_dict(), "pacman_model.pth")
        print("Model saved !")

    def plot_metrics(self, losses, training_accuracy, verification_accuracy, epoch):
        plt.figure(figsize=(12, 4))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(losses, label="Train Loss")
        plt.title(f"Loss jusqu'à l'époque {epoch}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(training_accuracy, label="Train Accuracy")
        plt.plot(verification_accuracy, label="Validation Accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
