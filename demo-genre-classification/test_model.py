from transformers import pipeline
import os

# Replace with your actual model repo name (e.g., 'your-username/distilhubert-finetuned-gtzan')
MODEL_ID = "BenbenbenZZZ/distilhubert-finetuned-gtzan"

def test_audio_files():
    pipe = pipeline("audio-classification", model=MODEL_ID)
    audio_files = [
        "metal-sample-1.wav",
        "metal-sample-2.wav"
    ]
    for audio_file in audio_files:
        file_path = os.path.join(os.path.dirname(__file__), audio_file)
        print(f"\nTesting {audio_file}...")
        preds = pipe(file_path)
        for pred in preds:
            print(f"Label: {pred['label']}, Score: {pred['score']:.4f}")

if __name__ == "__main__":
    test_audio_files() 