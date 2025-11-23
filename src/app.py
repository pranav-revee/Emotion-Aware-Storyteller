"""
Emotion-Aware Storyteller - Gradio Web Application
Main application integrating emotion detection, story generation, and TTS
"""

import gradio as gr
import numpy as np
import os
import tempfile
from typing import Tuple, Optional
import cv2

from detector import EmotionDetector
from generator import StoryGenerator
from tts import TextToSpeech


class EmotionAwareStoryteller:
    """
    Main application class combining all components
    """

    def __init__(
        self,
        llm_backend: str = "ollama",
        tts_backend: str = "gtts",
        model_name: Optional[str] = None
    ):
        """
        Initialize the storyteller application

        Args:
            llm_backend: LLM backend ('ollama', 'huggingface', 'openai', 'template')
            tts_backend: TTS backend ('gtts', 'edge-tts', 'pyttsx3')
            model_name: Optional model name for LLM
        """
        print("🎭 Initializing Emotion-Aware Storyteller...")

        # Initialize components
        self.detector = EmotionDetector(enforce_detection=False)
        print("✓ Emotion detector ready")

        self.generator = StoryGenerator(
            backend=llm_backend,
            model_name=model_name,
            max_length=150,
            temperature=0.7
        )
        print("✓ Story generator ready")

        self.tts = TextToSpeech(backend=tts_backend)
        print("✓ Text-to-speech ready")

        print("🎉 System ready!\n")

    def process_image(
        self,
        image: np.ndarray,
        prompt_type: str = "Story",
        enable_tts: bool = True,
        use_blended_emotions: bool = False
    ) -> Tuple[str, str, str, Optional[str]]:
        """
        Process image and generate emotion-aware content

        Args:
            image: Input image from webcam
            prompt_type: Type of content to generate
            enable_tts: Whether to generate audio
            use_blended_emotions: Whether to use multi-emotion blending

        Returns:
            Tuple of (emotion_info, generated_text, emotion_chart, audio_path)
        """
        if image is None:
            return "❌ No image provided", "", "", None

        try:
            # Detect emotion
            result = self.detector.detect_emotion(image)

            if 'error' in result and result['confidence'] == 0.0:
                return f"⚠️  No face detected. Please ensure your face is visible in the camera.", "", "", None

            dominant_emotion = result['dominant_emotion']
            confidence = result['confidence']
            emotions = result['emotions']

            # Create emotion info display
            emotion_info = f"**Detected Emotion:** {dominant_emotion.upper()}\n"
            emotion_info += f"**Confidence:** {confidence:.1%}\n\n"
            emotion_info += "**All Emotions:**\n"

            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            for emotion, score in sorted_emotions:
                bar = "█" * int(score / 5)
                emotion_info += f"- {emotion.capitalize()}: {score:.1f}% {bar}\n"

            # Get blended emotions if enabled
            blended = None
            if use_blended_emotions:
                blended = self.detector.get_blended_emotions(emotions, threshold=15.0)

            # Generate story/content
            prompt_type_map = {
                "Story": "story",
                "Caption": "caption",
                "Advice": "advice",
                "Poem": "poem",
                "Affirmation": "affirmation"
            }

            actual_prompt_type = prompt_type_map.get(prompt_type, "story")

            generated_text = self.generator.generate(
                emotion=dominant_emotion,
                prompt_type=actual_prompt_type,
                blended_emotions=blended if use_blended_emotions else None
            )

            # Generate TTS if enabled
            audio_path = None
            if enable_tts and generated_text:
                audio_path, _ = self.tts.text_to_speech_emotion_aware(
                    text=generated_text,
                    emotion=dominant_emotion
                )

            # Create emotion visualization
            emotion_chart = self._create_emotion_chart(sorted_emotions)

            return emotion_info, generated_text, emotion_chart, audio_path

        except Exception as e:
            error_msg = f"❌ Error processing image: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, "", "", None

    def _create_emotion_chart(self, sorted_emotions: list) -> str:
        """
        Create a text-based emotion chart

        Args:
            sorted_emotions: List of (emotion, score) tuples

        Returns:
            Formatted chart string
        """
        chart = "### Emotion Breakdown\n\n"

        for emotion, score in sorted_emotions:
            # Create progress bar
            bar_length = 20
            filled = int((score / 100) * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)

            chart += f"**{emotion.capitalize()}**\n"
            chart += f"`{bar}` {score:.1f}%\n\n"

        return chart


def create_gradio_interface(storyteller: EmotionAwareStoryteller):
    """
    Create and configure the Gradio interface

    Args:
        storyteller: EmotionAwareStoryteller instance

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(
        title="🎭 Emotion-Aware Storyteller",
        theme=gr.themes.Soft()
    ) as interface:

        gr.Markdown(
            """
            # 🎭 Emotion-Aware Storyteller

            **Real-time facial emotion detection powered by AI storytelling**

            📸 Capture your emotion → 🤖 AI generates personalized content → 🔊 Optional text-to-speech

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📷 Camera Input")
                image_input = gr.Image(
                    source="webcam",
                    type="numpy",
                    label="Webcam Feed"
                )

                # Controls
                gr.Markdown("### ⚙️ Settings")

                prompt_type = gr.Radio(
                    choices=["Story", "Caption", "Advice", "Poem", "Affirmation"],
                    value="Story",
                    label="Content Type"
                )

                enable_tts = gr.Checkbox(
                    value=True,
                    label="🔊 Enable Text-to-Speech"
                )

                use_blended = gr.Checkbox(
                    value=False,
                    label="🎨 Use Multi-Emotion Blending"
                )

                analyze_btn = gr.Button(
                    "✨ Analyze & Generate",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📊 Emotion Analysis")

                emotion_output = gr.Markdown(
                    label="Detected Emotion"
                )

                emotion_chart = gr.Markdown(
                    label="Emotion Breakdown"
                )

                gr.Markdown("### 📝 Generated Content")

                text_output = gr.Textbox(
                    label="AI-Generated Content",
                    lines=5,
                    show_copy_button=True
                )

                audio_output = gr.Audio(
                    label="🔊 Audio Output",
                    type="filepath"
                )

        # Examples section
        gr.Markdown(
            """
            ---
            ### 💡 Tips
            - Ensure your face is clearly visible and well-lit
            - Try different content types to see varied responses
            - Multi-emotion blending creates richer, more nuanced content
            - The AI adapts its tone based on your detected emotion

            ### 🏗️ Built With
            - **DeepFace** for emotion detection
            - **Llama 3** for story generation (via Ollama)
            - **gTTS** for text-to-speech
            - **Gradio** for the web interface
            """
        )

        # Connect the button to the processing function
        analyze_btn.click(
            fn=storyteller.process_image,
            inputs=[image_input, prompt_type, enable_tts, use_blended],
            outputs=[emotion_output, text_output, emotion_chart, audio_output]
        )

    return interface


def main():
    """
    Main entry point for the application
    """
    import argparse

    parser = argparse.ArgumentParser(description="Emotion-Aware Storyteller")

    parser.add_argument(
        "--llm-backend",
        type=str,
        default="ollama",
        choices=["ollama", "huggingface", "openai", "template"],
        help="LLM backend to use"
    )

    parser.add_argument(
        "--tts-backend",
        type=str,
        default="gtts",
        choices=["gtts", "edge-tts", "pyttsx3"],
        help="TTS backend to use"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for LLM (optional)"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )

    args = parser.parse_args()

    # Initialize the storyteller
    storyteller = EmotionAwareStoryteller(
        llm_backend=args.llm_backend,
        tts_backend=args.tts_backend,
        model_name=args.model_name
    )

    # Create and launch the interface
    interface = create_gradio_interface(storyteller)

    print(f"\n🚀 Launching Gradio interface on port {args.port}...")
    print(f"   LLM Backend: {args.llm_backend}")
    print(f"   TTS Backend: {args.tts_backend}")

    interface.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
