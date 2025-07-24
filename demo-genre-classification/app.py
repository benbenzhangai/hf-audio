import gradio as gr
from transformers import pipeline

MODEL_ID = "BenbenbenZZZ/distilhubert-finetuned-gtzan"
pipe = pipeline("audio-classification", model=MODEL_ID)

def classify_audio(file):
    if file is None:
        return {}
    preds = pipe(file)
    return {pred["label"]: float(pred["score"]) for pred in preds}

iface = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(
        sources=["microphone", "upload"],  # This enables browser recording by default
        type="filepath",
        label="Upload or record audio (music or voice)"
    ),
    outputs=gr.Label(num_top_classes=5, label="Predicted Genres"),
    title="Music Genre Classification Demo",
    description=(
        "Upload a music clip or record audio using your microphone. "
        "The model will classify the audio into one of the GTZAN music genres. "
        "Note: If you record your voice, the model will still try to assign a music genre."
    ),
    examples=[
        ["metal-sample-1.wav"],
        ["metal-sample-2.wav"]
    ]
)

if __name__ == "__main__":
    iface.launch()