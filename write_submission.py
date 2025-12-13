import pickle
import torch
import pandas as pd

from data import state_to_tensor, INDEX_TO_ACTION_MAP
from architecture import PacmanNetwork


class SubmissionWriter:
    def __init__(self, test_set_path, model_path):
        """
        Initialize the writing of your submission.
        Pay attention that the test set only contains GameState objects,
        it's no longer (GameState, action) pairs.
        """
        with open(test_set_path, "rb") as f:
            self.test_set = pickle.load(f)

        self.model = PacmanNetwork(input_size=25)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict_on_testset(self):
        """
        Generate predictions for the test set.
        The order MUST match the order of the test set.
        """
        actions = []

        with torch.no_grad():
            for state in self.test_set:
                x = state_to_tensor(state).unsqueeze(0)

                logits = self.model(x)

                best_idx = torch.argmax(logits, dim=1).item()

                action = INDEX_TO_ACTION_MAP[best_idx]

                actions.append(action)

        return actions

    def write_csv(self, actions, file_name="submission"):
        """
        ! Do not modify !
        """
        submission = pd.DataFrame(
            data={'ACTION': actions},
            columns=["ACTION"]
        )

        submission.to_csv(file_name + ".csv", index=False)


if __name__ == "__main__":
    writer = SubmissionWriter(
        test_set_path="pacman_test.pkl",
        model_path="pacman_model.pth"
    )
    predictions = writer.predict_on_testset()
    writer.write_csv(predictions)
