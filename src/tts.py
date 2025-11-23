"""
Text-to-Speech Module
Converts generated text to speech using various TTS backends
"""

import os
import tempfile
from typing import Optional, Tuple
from enum import Enum
import numpy as np


class TTSBackend(Enum):
    """Supported TTS backends"""
    GTTS = "gtts"
    EDGE_TTS = "edge-tts"
    PYTTSX3 = "pyttsx3"  # Offline alternative


class TextToSpeech:
    """
    Text-to-speech converter with multiple backend support
    """

    def __init__(
        self,
        backend: str = "gtts",
        language: str = "en",
        slow: bool = False,
        voice: Optional[str] = None
    ):
        """
        Initialize TTS engine

        Args:
            backend: TTS backend ('gtts', 'edge-tts', 'pyttsx3')
            language: Language code (e.g., 'en', 'es', 'fr')
            slow: Whether to use slow speech (gTTS only)
            voice: Specific voice to use (backend-dependent)
        """
        self.backend = backend.lower()
        self.language = language
        self.slow = slow
        self.voice = voice
        self.engine = None

        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the selected TTS backend"""
        if self.backend == "pyttsx3":
            self._init_pyttsx3()

    def _init_pyttsx3(self):
        """Initialize pyttsx3 offline TTS engine"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()

            # Set properties
            if self.voice:
                self.engine.setProperty('voice', self.voice)

            print(f"✓ Initialized pyttsx3 TTS engine")

        except Exception as e:
            print(f"Warning: Could not initialize pyttsx3: {e}")
            self.backend = "gtts"

    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> Tuple[Optional[str], Optional[int]]:
        """
        Convert text to speech and save as audio file

        Args:
            text: Text to convert
            output_path: Optional output path (generated if not provided)

        Returns:
            Tuple of (audio_file_path, sample_rate) or (None, None) on error
        """
        if not text or not text.strip():
            return None, None

        # Generate output path if not provided
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "emotion_story.mp3")

        try:
            if self.backend == "gtts":
                return self._tts_gtts(text, output_path)
            elif self.backend == "edge-tts":
                return self._tts_edge(text, output_path)
            elif self.backend == "pyttsx3":
                return self._tts_pyttsx3(text, output_path)
            else:
                print(f"Unknown backend: {self.backend}, falling back to gTTS")
                return self._tts_gtts(text, output_path)

        except Exception as e:
            print(f"Error in TTS conversion: {e}")
            return None, None

    def _tts_gtts(self, text: str, output_path: str) -> Tuple[str, int]:
        """
        Generate speech using gTTS (Google Text-to-Speech)
        """
        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang=self.language, slow=self.slow)
            tts.save(output_path)

            print(f"✓ Generated speech with gTTS: {output_path}")
            return output_path, 24000  # gTTS typical sample rate

        except Exception as e:
            print(f"Error in gTTS: {e}")
            raise

    def _tts_edge(self, text: str, output_path: str) -> Tuple[str, int]:
        """
        Generate speech using Edge TTS
        """
        try:
            import edge_tts
            import asyncio

            # Default voice for English
            voice = self.voice or "en-US-AriaNeural"

            async def generate():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(output_path)

            # Run async function
            asyncio.run(generate())

            print(f"✓ Generated speech with Edge TTS: {output_path}")
            return output_path, 24000

        except Exception as e:
            print(f"Error in Edge TTS: {e}")
            # Fallback to gTTS
            return self._tts_gtts(text, output_path)

    def _tts_pyttsx3(self, text: str, output_path: str) -> Tuple[str, int]:
        """
        Generate speech using pyttsx3 (offline)
        """
        try:
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()

            print(f"✓ Generated speech with pyttsx3: {output_path}")
            return output_path, 22050

        except Exception as e:
            print(f"Error in pyttsx3: {e}")
            raise

    def get_emotion_voice_mapping(self, emotion: str) -> Optional[str]:
        """
        Get appropriate voice based on emotion (for Edge TTS)

        Args:
            emotion: Detected emotion

        Returns:
            Voice name string for Edge TTS
        """
        # Edge TTS emotion-appropriate voices
        emotion_voices = {
            "happy": "en-US-AriaNeural",  # Cheerful female
            "sad": "en-US-GuyNeural",  # Calm male
            "angry": "en-US-DavisNeural",  # Strong male
            "fear": "en-US-JennyNeural",  # Gentle female
            "surprise": "en-US-AriaNeural",  # Expressive
            "neutral": "en-US-JennyNeural",  # Neutral
            "disgust": "en-US-DavisNeural",  # Firm
        }

        return emotion_voices.get(emotion.lower(), "en-US-JennyNeural")

    def text_to_speech_emotion_aware(
        self,
        text: str,
        emotion: str,
        output_path: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Convert text to speech with emotion-appropriate voice

        Args:
            text: Text to convert
            emotion: Detected emotion
            output_path: Optional output path

        Returns:
            Tuple of (audio_file_path, sample_rate)
        """
        # Use emotion-specific voice for Edge TTS
        if self.backend == "edge-tts":
            original_voice = self.voice
            self.voice = self.get_emotion_voice_mapping(emotion)
            result = self.text_to_speech(text, output_path)
            self.voice = original_voice
            return result
        else:
            return self.text_to_speech(text, output_path)

    @staticmethod
    def list_available_voices_edge() -> list:
        """
        List available Edge TTS voices

        Returns:
            List of voice names
        """
        try:
            import edge_tts
            import asyncio

            async def get_voices():
                voices = await edge_tts.list_voices()
                return voices

            voices = asyncio.run(get_voices())
            return [v["Name"] for v in voices]

        except Exception as e:
            print(f"Error listing voices: {e}")
            return []

    @staticmethod
    def list_available_voices_pyttsx3() -> list:
        """
        List available pyttsx3 voices

        Returns:
            List of voice objects
        """
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            return [(v.id, v.name) for v in voices]

        except Exception as e:
            print(f"Error listing voices: {e}")
            return []


def test_tts():
    """Test TTS functionality"""
    tts = TextToSpeech(backend="gtts")

    test_text = "Hello! This is a test of the emotion-aware storytelling system."

    print("Generating speech...")
    audio_path, sample_rate = tts.text_to_speech(test_text)

    if audio_path:
        print(f"✓ Audio saved to: {audio_path}")
        print(f"  Sample rate: {sample_rate} Hz")

        # Test emotion-aware version
        print("\nTesting emotion-aware TTS...")
        audio_path_2, _ = tts.text_to_speech_emotion_aware(
            "You're doing great! Keep up the positive energy!",
            emotion="happy"
        )
        print(f"✓ Emotion-aware audio saved to: {audio_path_2}")
    else:
        print("✗ Failed to generate audio")


if __name__ == "__main__":
    test_tts()
