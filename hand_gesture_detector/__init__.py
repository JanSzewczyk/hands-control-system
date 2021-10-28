import pickle
import pandas as pd
import numpy as np

from typing import Optional, Any

from models import Hand, GestureClassificationResult, ActionType
import utils.hand_utils as hu


class HandGestureDetector:
    """
    Classification of hand gestures using a previously prepared classification model.
    Classification model is from scikit-learn library. The model is loaded from the file.

    Attributes:
        _min_classification_confidence (float): Minimal certainty of classification to be considered
            as the correct choice of the classifier.
        _model (Optional[Any]): Classification model.
    """

    def __init__(self, min_classification_confidence: float = 0.5):
        """
        Constructor.

        Args:
            min_classification_confidence (float): Defaults to 0.5. Minimum Classification Confidence Threshold.
        """

        self._min_classification_confidence: float = min_classification_confidence
        self._model: Optional[Any] = None

        # Load model from file
        self.__load_classification_model('hand-gestures-model.pkl')

    def __load_classification_model(self, file_location_path: str) -> None:
        """
        Load classification model from file.

        Args:
            file_location_path: Path to file with classification model.
        """

        with open(file_location_path, 'rb') as f:
            self._model = pickle.load(f)

    def predict(self, hand: Hand) -> Optional[GestureClassificationResult]:
        """
        Predict hand gesture using classification model.
        When predicted gesture probability is smallest than self._min_classification_confidence
        return None, in other cases return information about predicted hand gesture.

        Args:
            hand (model.Hand): Hand information.

        Returns:
            Optional[model.GestureClassificationResult]: Gesture Classification Result while predicting successfully.
        """

        gesture_classification_result = GestureClassificationResult()

        # Prepare data to prediction
        predicted_data = self.prepare_predicted_data(hand)

        hand_gesture_class = self._model.predict(predicted_data)[0]
        hand_gesture_prob = self._model.predict_proba(predicted_data)[0]

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
            hand (model.Hand): Hand information.

        Returns:
            pandas.core.frame.DataFrame: DataFrame object with data ready to predict hand gesture.
        """

        # Scales Hand landmarks to ranges [0, 1].
        landmarks = hu.prepare_hand_data(hand)

        # Flattening the list of hand landmarks
        row = np.array(landmarks).flatten()

        return pd.DataFrame([row])
