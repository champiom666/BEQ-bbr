"""Behaviour names and descriptions used to initialise behaviour queries.

Each behaviour query in the BEQ decoder is initialised from the Qwen token
embeddings of a short textual description that contains both the category name
and discriminative visual cues (e.g. "Shrug: shoulders are raised and dropped").
"""

from __future__ import annotations

from .constants import LABEL_COLUMNS


BEHAVIOUR_DESCRIPTIONS = {
    "Settle": "The participant settles into a sitting posture or adjusts body position after movement.",
    "Legs crossed": "The participant has one leg crossed over the other or moves into a crossed-leg posture.",
    "Groom": "The participant grooms themselves, such as adjusting hair, touching face for grooming, or tidying appearance.",
    "Hand-mouth": "The participant touches the mouth, covers the mouth, or moves a hand very near the mouth.",
    "Fold arms": "The participant folds or crosses their arms in front of the torso.",
    "Leg movement": "The participant moves, shakes, lifts, or repositions one or both legs.",
    "Scratch": "The participant scratches the face, head, arm, body, or another visible body part.",
    "Gesture": "The participant makes expressive hand or arm movements while communicating.",
    "Hand-face": "The participant touches the face or keeps a hand near the face.",
    "Adjusting clothing": "The participant adjusts clothing, sleeves, collar, glasses, or worn accessories.",
    "Fumble": "The participant fumbles with hands, fingers, objects, or small repetitive hand movements.",
    "Shrug": "The participant raises shoulders or makes a shrugging body gesture.",
    "Stretching": "The participant stretches arms, shoulders, neck, back, or upper body.",
    "Smearing hands": "The participant rubs, smears, or wipes the hands together.",
}


DISCRIMINATIVE_BEHAVIOUR_DESCRIPTIONS = {
    "Settle": "The participant makes a broad posture adjustment or settles the torso after movement.",
    "Legs crossed": "One leg is visibly crossed over the other or the participant moves into a crossed-leg posture.",
    "Groom": "The participant tidies appearance, hair, face, glasses, or accessories with purposeful grooming motion.",
    "Hand-mouth": "A hand touches, covers, or hovers directly at the mouth or lips.",
    "Fold arms": "Both arms are folded or crossed in front of the torso and held there.",
    "Leg movement": "One or both legs shake, lift, swing, tap, or reposition while seated.",
    "Scratch": "A hand or fingers repeatedly scratch a localized face, head, arm, or body area.",
    "Gesture": "Hands or arms move expressively outward or through space while communicating.",
    "Hand-face": "A hand touches or rests near the cheek, chin, forehead, eyes, or general face area.",
    "Adjusting clothing": "Hands manipulate sleeves, collar, shirt, jacket, glasses, or worn items.",
    "Fumble": "Hands or fingers make small restless movements with each other or with an object.",
    "Shrug": "Shoulders briefly lift upward or the upper body makes a shrugging motion.",
    "Stretching": "Arms, shoulders, neck, back, or upper body extend into a stretch.",
    "Smearing hands": "Both hands rub, wipe, or smear against each other in a repeated motion.",
}


def build_behaviour_texts(style: str = "name_description") -> list[str]:
    """Return behaviour texts in the official label order."""
    style = style.lower()
    texts: list[str] = []
    for name in LABEL_COLUMNS:
        description = BEHAVIOUR_DESCRIPTIONS[name]
        discriminative_description = DISCRIMINATIVE_BEHAVIOUR_DESCRIPTIONS[name]
        if style == "name":
            texts.append(name)
        elif style == "description":
            texts.append(description)
        elif style == "name_description":
            texts.append(f"{name}: {description}")
        elif style == "name_description_discriminative":
            texts.append(f"{name}: {discriminative_description}")
        else:
            raise ValueError(f"Unknown behaviour text style: {style}")
    return texts
