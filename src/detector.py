"""
Emotion Detection Module using DeepFace and OpenCV
Detects facial emotions from images/webcam frames
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from deepface import DeepFace


class EmotionDetector:
    """
    Wrapper class for DeepFace emotion detection
    """

    def __init__(self, enforce_detection: bool = False):
        """
        Initialize the emotion detector

        Args:
            enforce_detection: If True, raise exception if no face detected
        """
        self.enforce_detection = enforce_detection
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def detect_emotion(self, image: np.ndarray) -> Dict:
        """
        Detect emotion from an image frame

        Args:
            image: numpy array representing an image (BGR format from OpenCV)

        Returns:
            Dict containing:
                - dominant_emotion: str (primary emotion detected)
                - emotions: Dict[str, float] (all emotion probabilities)
                - region: Dict (face bounding box coordinates)
                - confidence: float (confidence of dominant emotion)
        """
        try:
            # DeepFace.analyze returns a list in newer versions
            result = DeepFace.analyze(
                img_path=image,
                actions=['emotion'],
                enforce_detection=self.enforce_detection,
                detector_backend='opencv',
                silent=True
            )

            # Handle both list and dict returns
            if isinstance(result, list):
                result = result[0]

            # Extract emotion data
            emotions = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', 'neutral')
            region = result.get('region', {})

            # Calculate confidence as the probability of dominant emotion
            confidence = emotions.get(dominant_emotion, 0.0) if emotions else 0.0

            return {
                'dominant_emotion': dominant_emotion,
                'emotions': emotions,
                'region': region,
                'confidence': confidence / 100.0  # Convert percentage to decimal
            }

        except Exception as e:
            # Return neutral emotion if detection fails
            return {
                'dominant_emotion': 'neutral',
                'emotions': {'neutral': 100.0},
                'region': {},
                'confidence': 0.0,
                'error': str(e)
            }

    def get_blended_emotions(self, emotions: Dict[str, float], threshold: float = 10.0) -> Dict[str, float]:
        """
        Get emotions above a certain threshold for multi-emotion blending

        Args:
            emotions: Dict of emotion probabilities
            threshold: Minimum probability threshold (percentage)

        Returns:
            Dict of emotions above threshold with normalized probabilities
        """
        filtered = {k: v for k, v in emotions.items() if v >= threshold}

        if not filtered:
            return emotions

        # Normalize to sum to 1.0
        total = sum(filtered.values())
        normalized = {k: v / total for k, v in filtered.items()}

        return normalized

    def draw_emotion_overlay(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw emotion detection results on the image

        Args:
            image: Original image
            result: Detection result from detect_emotion()

        Returns:
            Image with overlay
        """
        img_copy = image.copy()

        # Draw bounding box if face region available
        region = result.get('region', {})
        if region:
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add emotion text
        dominant = result.get('dominant_emotion', 'unknown')
        confidence = result.get('confidence', 0.0)

        text = f"{dominant.upper()} ({confidence:.2%})"
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        return img_copy

    @staticmethod
    def capture_from_webcam() -> Optional[np.ndarray]:
        """
        Capture a single frame from webcam

        Returns:
            Image as numpy array or None if capture fails
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            return None

        ret, frame = cap.read()
        cap.release()

        return frame if ret else None


def test_emotion_detection():
    """
    Test function for emotion detection
    """
    detector = EmotionDetector()

    # Try to capture from webcam
    frame = EmotionDetector.capture_from_webcam()

    if frame is None:
        print("Could not capture from webcam")
        return

    # Detect emotion
    result = detector.detect_emotion(frame)

    print(f"Dominant Emotion: {result['dominant_emotion']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"All emotions: {result['emotions']}")

    # Show image with overlay
    overlay = detector.draw_emotion_overlay(frame, result)
    cv2.imshow('Emotion Detection', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_emotion_detection()
