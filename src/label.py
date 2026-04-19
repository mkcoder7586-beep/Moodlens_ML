import pandas as pd
import ast

# =========================
# HF Mapping
# =========================
HF_MAP = {
    "sadness": ("sad", "sad"),
    "joy": ("happy", "joyful"),
    "love": ("happy", "love"),
    "anger": ("angry", "angry"),
    "fear": ("fear", "fear"),
    "surprise": ("surprise", "surprised")
}

# =========================
# GoEmotions Mapping
# =========================
GOEMOTION_MAP = {
    "admiration": ("happy", "proud"),
    "amusement": ("happy", "joyful"),
    "anger": ("angry", "angry"),
    "annoyance": ("angry", "annoyed"),
    "disappointment": ("sad", "disappointed"),
    "grief": ("sad", "hopeless"),
    "fear": ("fear", "fear"),
    "nervousness": ("fear", "nervous"),
    "joy": ("happy", "joyful"),
    "love": ("happy", "love"),
    "optimism": ("happy", "excited"),
    "sadness": ("sad", "sad"),
    "surprise": ("surprise", "surprised"),
    "neutral": ("neutral", "neutral")
}

# =========================
# Utility: Safe label conversion
# =========================
def safe_convert(x):
    if isinstance(x, list):
        return [int(i) for i in x]
    elif isinstance(x, str):
        return [int(i) for i in ast.literal_eval(x)]
    else:
        return []

# =========================
# Process HF Dataset (single → multi-label format)
# =========================
def process_hf(df, label_names):
    processed = []

    for _, row in df.iterrows():
        text = row['text']
        emotion = label_names[row['label']]

        if emotion in HF_MAP:
            primary, sub = HF_MAP[emotion]

            processed.append({
                "text": text,
                "primary_emotion": primary,
                "sub_emotions": [sub]   # convert to list
            })

    return pd.DataFrame(processed)


# =========================
# Process GoEmotions Dataset (multi-label)
# =========================
def process_goemotions(df, label_names):
    df['labels'] = df['labels'].apply(safe_convert)

    processed = []

    for _, row in df.iterrows():
        text = row['text']
        label_ids = row['labels']

        emotions = [label_names[i] for i in label_ids]

        sub_emotions = []
        primary_emotion = None

        for emotion in emotions:
            if emotion in GOEMOTION_MAP:
                primary, sub = GOEMOTION_MAP[emotion]

                sub_emotions.append(sub)

                # Set first valid primary emotion
                if primary_emotion is None:
                    primary_emotion = primary

        if sub_emotions:
            processed.append({
                "text": text,
                "primary_emotion": primary_emotion,
                "sub_emotions": list(set(sub_emotions))  # remove duplicates
            })

    return pd.DataFrame(processed)