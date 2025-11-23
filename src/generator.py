"""
Story Generation Module using LLMs
Generates context-aware captions and stories based on detected emotions
"""

import os
from typing import Dict, Optional, List
from enum import Enum


class LLMBackend(Enum):
    """Supported LLM backends"""
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    OPENAI = "openai"  # Optional for API-based generation


class StoryGenerator:
    """
    Generates emotion-aware stories and captions using various LLM backends
    """

    def __init__(
        self,
        backend: str = "ollama",
        model_name: Optional[str] = None,
        max_length: int = 150,
        temperature: float = 0.7
    ):
        """
        Initialize the story generator

        Args:
            backend: LLM backend to use ('ollama', 'huggingface', 'openai')
            model_name: Model name/path (default varies by backend)
            max_length: Maximum generation length
            temperature: Sampling temperature for generation
        """
        self.backend = backend.lower()
        self.max_length = max_length
        self.temperature = temperature

        # Set default model names
        if model_name is None:
            if self.backend == "ollama":
                self.model_name = "llama3"
            elif self.backend == "huggingface":
                self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
            else:
                self.model_name = "gpt-3.5-turbo"
        else:
            self.model_name = model_name

        # Initialize the model
        self.model = None
        self.tokenizer = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the selected backend"""
        if self.backend == "huggingface":
            self._init_huggingface()
        elif self.backend == "ollama":
            self._init_ollama()
        elif self.backend == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _init_huggingface(self):
        """Initialize Hugging Face transformers pipeline"""
        try:
            from transformers import pipeline, AutoTokenizer
            import torch

            device = 0 if torch.cuda.is_available() else -1

            self.model = pipeline(
                "text-generation",
                model=self.model_name,
                device=device,
                max_new_tokens=self.max_length,
                temperature=self.temperature
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            print(f"✓ Loaded Hugging Face model: {self.model_name}")

        except Exception as e:
            print(f"Warning: Could not initialize Hugging Face backend: {e}")
            print("Falling back to simple template-based generation")
            self.backend = "template"

    def _init_ollama(self):
        """Initialize Ollama client"""
        try:
            import ollama
            # Test connection
            ollama.list()
            print(f"✓ Connected to Ollama, will use model: {self.model_name}")

        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Falling back to template-based generation")
            self.backend = "template"

    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self.model = openai.OpenAI(api_key=api_key)
            print(f"✓ Initialized OpenAI client with model: {self.model_name}")

        except Exception as e:
            print(f"Warning: Could not initialize OpenAI backend: {e}")
            print("Falling back to template-based generation")
            self.backend = "template"

    def create_prompt(
        self,
        emotion: str,
        prompt_type: str = "story",
        context: Optional[str] = None,
        blended_emotions: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Create a prompt for the LLM based on emotion and type

        Args:
            emotion: Dominant emotion detected
            prompt_type: Type of generation ('story', 'caption', 'advice', 'poem')
            context: Optional additional context
            blended_emotions: Optional dict of multiple emotions with weights

        Returns:
            Formatted prompt string
        """
        emotion_context = emotion

        # If blended emotions provided, create richer context
        if blended_emotions and len(blended_emotions) > 1:
            sorted_emotions = sorted(blended_emotions.items(), key=lambda x: x[1], reverse=True)
            top_emotions = [e[0] for e in sorted_emotions[:3]]
            emotion_context = f"{emotion} (with hints of {', '.join(top_emotions[1:])})"

        prompts = {
            "story": f"Write a short, uplifting two-sentence story for someone who feels {emotion_context}. Make it warm, encouraging, and relatable.",
            "caption": f"Generate a single inspirational caption (one sentence) for someone feeling {emotion_context}. Make it positive and motivating.",
            "advice": f"Give brief, compassionate advice (2-3 sentences) for someone experiencing {emotion_context}. Be understanding and supportive.",
            "poem": f"Write a short 4-line poem that acknowledges and uplifts someone feeling {emotion_context}. Make it gentle and hopeful.",
            "affirmation": f"Create a powerful affirmation statement for someone feeling {emotion_context}. Make it personal and empowering."
        }

        base_prompt = prompts.get(prompt_type, prompts["story"])

        if context:
            base_prompt += f" Context: {context}"

        return base_prompt

    def generate(
        self,
        emotion: str,
        prompt_type: str = "story",
        context: Optional[str] = None,
        blended_emotions: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate text based on emotion

        Args:
            emotion: Dominant emotion
            prompt_type: Type of generation
            context: Optional context
            blended_emotions: Optional blended emotions

        Returns:
            Generated text
        """
        prompt = self.create_prompt(emotion, prompt_type, context, blended_emotions)

        if self.backend == "huggingface":
            return self._generate_huggingface(prompt)
        elif self.backend == "ollama":
            return self._generate_ollama(prompt)
        elif self.backend == "openai":
            return self._generate_openai(prompt)
        else:
            return self._generate_template(emotion, prompt_type)

    def _generate_huggingface(self, prompt: str) -> str:
        """Generate using Hugging Face transformers"""
        try:
            messages = [
                {"role": "system", "content": "You are a compassionate storyteller who creates uplifting content."},
                {"role": "user", "content": prompt}
            ]

            outputs = self.model(
                messages,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True
            )

            generated_text = outputs[0]['generated_text']

            # Extract assistant response
            if isinstance(generated_text, list):
                return generated_text[-1]['content']
            return generated_text

        except Exception as e:
            print(f"Error in HuggingFace generation: {e}")
            return self._generate_template(prompt.split("feeling ")[1].split()[0] if "feeling " in prompt else "neutral", "story")

    def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama"""
        try:
            import ollama

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a compassionate storyteller who creates uplifting, concise content. Keep responses short and warm."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_length
                }
            )

            return response['message']['content'].strip()

        except Exception as e:
            print(f"Error in Ollama generation: {e}")
            return self._generate_template(prompt.split("feeling ")[1].split()[0] if "feeling " in prompt else "neutral", "story")

    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API"""
        try:
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a compassionate storyteller who creates uplifting content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_length,
                temperature=self.temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error in OpenAI generation: {e}")
            return self._generate_template(prompt.split("feeling ")[1].split()[0] if "feeling " in prompt else "neutral", "story")

    def _generate_template(self, emotion: str, prompt_type: str = "story") -> str:
        """
        Fallback template-based generation (no LLM required)
        """
        templates = {
            "happy": {
                "story": "Your joy is contagious and lights up the world around you. Keep sharing that beautiful smile and positive energy with everyone you meet.",
                "caption": "Happiness looks amazing on you - keep shining! ✨",
                "advice": "Your happiness is a gift. Take a moment to appreciate this feeling and maybe share it with someone who needs a smile today.",
            },
            "sad": {
                "story": "It's okay to feel sad sometimes - it means you care deeply. Remember, storms pass and sunshine always returns. You're stronger than you know.",
                "caption": "This too shall pass. Be gentle with yourself. 💙",
                "advice": "Allow yourself to feel these emotions. Reach out to someone you trust, take care of yourself, and remember that brighter days are ahead.",
            },
            "angry": {
                "story": "Your feelings are valid. Take a deep breath and remember that you have the power to choose how you respond. You're in control.",
                "caption": "Feel the feeling, then let it go. You've got this. 💪",
                "advice": "Take a moment to breathe deeply. Channel this energy into something constructive, and remember that staying calm gives you power.",
            },
            "surprise": {
                "story": "Life is full of unexpected moments. Embrace the surprise and see where this new path takes you. Adventure awaits!",
                "caption": "Expect the unexpected - that's where the magic happens! ✨",
                "advice": "Stay curious and open. Surprises can lead to wonderful new experiences and opportunities.",
            },
            "fear": {
                "story": "Courage isn't the absence of fear - it's moving forward despite it. You're braver than you believe. Take it one step at a time.",
                "caption": "Feel the fear and do it anyway. You're stronger than you know. 🦁",
                "advice": "Acknowledge your fear without letting it control you. Break challenges into smaller steps, and remember past times you've overcome difficulty.",
            },
            "neutral": {
                "story": "In this moment of calm, find your center. Sometimes the quiet moments are when we discover our greatest clarity and peace.",
                "caption": "Peaceful moments are gifts. Embrace the stillness. 🌊",
                "advice": "Use this balanced state to reflect, plan, or simply be present. Calm is a strength.",
            },
            "disgust": {
                "story": "It's okay to have boundaries and standards. Trust your instincts and know that it's fine to walk away from what doesn't serve you.",
                "caption": "Honor your boundaries. You deserve better. 🛡️",
                "advice": "Your feelings are telling you something important. Listen to them and make choices that align with your values.",
            }
        }

        emotion = emotion.lower()
        return templates.get(emotion, templates["neutral"]).get(prompt_type, templates[emotion]["story"])


def test_generator():
    """Test the story generator"""
    # Try Ollama first, fall back to template
    generator = StoryGenerator(backend="ollama")

    emotions = ["happy", "sad", "angry", "surprise", "fear", "neutral"]

    for emotion in emotions:
        print(f"\n=== {emotion.upper()} ===")
        story = generator.generate(emotion, prompt_type="story")
        print(f"Story: {story}")


if __name__ == "__main__":
    test_generator()
