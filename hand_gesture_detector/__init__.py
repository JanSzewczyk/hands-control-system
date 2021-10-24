import pickle
import pandas as pd
import numpy as np

from typing import Optional

from models import Hand, GestureClassificationResult, ActionType
import utils.hand_utils as hu


class HandGestureDetector:

    def __init__(self, min_classification_confidence=0.5):
        """
        Constructor.

        Args:
            min_classification_confidence: Minimum Classification Confidence Threshold.
        """

        self._min_classification_confidence = min_classification_confidence
        self.model = None

        # Load model from file
        self.load_classification_model('hand-gestures-model.pkl')

    def load_classification_model(self, file_location_path: str) -> None:
        """
        Load classification model from file.

        Args:
            file_location_path: Path to file with classification model.
        """

        with open(file_location_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, hand: Hand) -> Optional[GestureClassificationResult]:
        """

        Args:
            hand: Hand information.

        Returns:
            Gesture Classification Result while predicting successfully.
        """

        gesture_classification_result = GestureClassificationResult()

        # Prepare data to prediction
        predicted_data = self.prepare_predicted_data(hand)

        hand_gesture_class = self.model.predict(predicted_data)[0]
        hand_gesture_prob = self.model.predict_proba(predicted_data)[0]

        gesture_classification_result.gesture_type = ActionType(hand_gesture_class)
        gesture_classification_result.score = round(hand_gesture_prob[np.argmax(hand_gesture_prob)], 2)

        if gesture_classification_result.score < self._min_classification_confidence:
            return None
        else:
            return gesture_classification_result

    @staticmethod
    def prepare_predicted_data(hand: Hand) -> pd.DataFrame:
        """
        Preparing a list of landmarks to predict hand gesture.

        Args:
            hand: Hand information.

        Returns:
            DataFrame object with data ready to predict hand gesture.
        """

        # Scales Hand landmarks to ranges [0, 1].
        landmarks = hu.prepare_hand_data(hand)

        # Flattening the list of hand landmarks
        row = list(np.array(landmarks).flatten())

        return pd.DataFrame([row])
