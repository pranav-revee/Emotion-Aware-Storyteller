"""
Emotion-Aware Storyteller Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .detector import EmotionDetector
from .generator import StoryGenerator
from .tts import TextToSpeech

__all__ = [
    'EmotionDetector',
    'StoryGenerator',
    'TextToSpeech'
]
