"""
Emotion to natural language instruction mapping for Qwen3-TTS.

Maps Claude's emotion categories and intensities to natural language
'instruct' parameters that Qwen3-TTS understands.
"""

# Emotion mappings with intensity thresholds
# Each emotion maps to three intensity levels: low, medium, high
EMOTION_INSTRUCTIONS = {
    "neutral": {
        "low": "",
        "medium": "",
        "high": ""
    },
    "sarcastic": {
        "low": "Speak with a hint of sarcasm",
        "medium": "Speak with clear sarcasm and a mocking undertone",
        "high": "Speak with heavy, dripping sarcasm"
    },
    "excited": {
        "low": "Speak with mild enthusiasm",
        "medium": "Speak with clear excitement and energy",
        "high": "Speak with intense excitement and jubilation!"
    },
    "disappointed": {
        "low": "Speak with slight disappointment",
        "medium": "Speak with clear disappointment and sadness",
        "high": "Speak with deep disappointment and frustration"
    },
    "angry": {
        "low": "Speak with slight irritation",
        "medium": "Speak with clear anger",
        "high": "Speak with intense outrage!"
    },
    "laughing": {
        "low": "Speak with an amused tone",
        "medium": "Speak with light laughter in your voice",
        "high": "Speak with hearty laughter"
    },
    "mocking": {
        "low": "Speak with a teasing tone",
        "medium": "Speak as if making fun of something",
        "high": "Speak with heavy mockery"
    },
    "happy": {
        "low": "Speak with a pleasant, warm tone",
        "medium": "Speak with clear happiness",
        "high": "Speak with overwhelming joy!"
    },
    "sad": {
        "low": "Speak with a slightly subdued tone",
        "medium": "Speak with clear sadness",
        "high": "Speak with deep sorrow"
    },
    "fearful": {
        "low": "Speak with slight nervousness",
        "medium": "Speak with clear fear in your voice",
        "high": "Speak with terror and panic!"
    },
    "surprised": {
        "low": "Speak with mild surprise",
        "medium": "Speak with clear surprise and wonder",
        "high": "Speak with complete shock and amazement!"
    },
    "disgusted": {
        "low": "Speak with slight distaste",
        "medium": "Speak with clear disgust",
        "high": "Speak with intense revulsion"
    },
    "curious": {
        "low": "Speak with mild curiosity",
        "medium": "Speak with engaged interest",
        "high": "Speak with intense fascination"
    },
    "confident": {
        "low": "Speak with quiet confidence",
        "medium": "Speak with clear self-assurance",
        "high": "Speak with bold, commanding confidence"
    },
    "worried": {
        "low": "Speak with slight concern",
        "medium": "Speak with clear worry",
        "high": "Speak with intense anxiety"
    },
    "playful": {
        "low": "Speak with a light, playful hint",
        "medium": "Speak with a clearly playful, teasing tone",
        "high": "Speak with intense playfulness and mischief"
    }
}


def get_intensity_level(intensity: float) -> str:
    """Convert intensity float (0.0-1.0) to level name."""
    if intensity <= 0.35:
        return "low"
    elif intensity <= 0.7:
        return "medium"
    else:
        return "high"


def map_emotion_to_instruct(emotion: str, intensity: float = 0.5) -> str:
    """
    Map an emotion and intensity to a natural language instruction.

    Args:
        emotion: Emotion name (e.g., "excited", "sarcastic", "neutral")
        intensity: Float from 0.0 to 1.0 indicating emotion strength

    Returns:
        Natural language instruction string for Qwen3-TTS
    """
    # Normalize emotion to lowercase
    emotion_lower = emotion.lower().strip()

    # Get intensity level
    level = get_intensity_level(intensity)

    # Look up instruction
    if emotion_lower in EMOTION_INSTRUCTIONS:
        return EMOTION_INSTRUCTIONS[emotion_lower][level]

    # Fallback: construct a generic instruction
    if emotion_lower and emotion_lower != "neutral":
        intensity_adj = {
            "low": "mildly",
            "medium": "clearly",
            "high": "very"
        }[level]
        return f"Speak in a {intensity_adj} {emotion_lower} manner"

    return ""
