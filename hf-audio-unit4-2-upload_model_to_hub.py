# Script to upload the fine-tuned model to Hugging Face Hub

import os
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from huggingface_hub import login, HfFolder

# Define paths
CHECKPOINT_PATH = "/teamspace/studios/this_studio/hf-audio/distilhubert-finetuned-gtzan/checkpoint-1130"

# Define genre labels
id2label = {
    '0': 'blues',
    '1': 'classical',
    '2': 'country',
    '3': 'disco',
    '4': 'hiphop',
    '5': 'jazz',
    '6': 'metal',
    '7': 'pop',
    '8': 'reggae',
    '9': 'rock'
}
label2id = {v: k for k, v in id2label.items()}

# Login to Hugging Face Hub
def login_to_hub(token=None):
    if token is None:
        # Allow manual token input
        token = input("Enter your Hugging Face token (from https://huggingface.co/settings/tokens): ")
    
    print("Logging in to Hugging Face Hub...")
    login(token=token, add_to_git_credential=True)
    
    # Check if login was successful
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        print("Not logged in or unable to retrieve user info")
        print(f"Error: {e}")
        return False

# Load the model and feature extractor from checkpoint
def load_model_from_checkpoint():
    print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
    model = AutoModelForAudioClassification.from_pretrained(
        CHECKPOINT_PATH,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
    )
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(CHECKPOINT_PATH)
    
    return model, feature_extractor

# Push model to hub
def push_model_to_hub(model, feature_extractor, repo_name):
    # Create model card content
    model_card = f"""
    ---
    language: en
    license: apache-2.0
    tags:
    - audio-classification
    - transformers
    - gtzan
    - music-genre-classification
    datasets:
    - marsyas/gtzan
    metrics:
    - accuracy
    ---
    
    # DistilHuBERT Fine-tuned on GTZAN
    
    This model is a fine-tuned version of [ntu-spml/distilhubert](https://huggingface.co/ntu-spml/distilhubert) on the [GTZAN](https://huggingface.co/datasets/marsyas/gtzan) dataset for music genre classification.
    
    ## Model description
    
    The model is based on DistilHuBERT, a distilled version of the HuBERT model, fine-tuned for audio classification tasks. It has been trained to classify music into 10 different genres.
    
    ## Training procedure
    
    The model was fine-tuned on the GTZAN dataset, which contains 1000 audio tracks each 30 seconds long, with 100 tracks per genre across 10 genres.
    
    ## Evaluation results
    
    The model achieves an accuracy of approximately 82% on the test set.
    
    ## Usage
    
    ```python
    from transformers import pipeline
    
    classifier = pipeline("audio-classification", model="{repo_name}")
    
    # Classify an audio file
    result = classifier("path/to/audio.wav")
    print(result)
    ```
    
    ## Limitations and bias
    
    This model was trained on a limited dataset and may not generalize well to all music genres or to music from different cultural contexts.
    
    ## Training
    
    The model was trained using the Hugging Face Transformers library.
    """
    
    # Save model card
    with open("README.md", "w") as f:
        f.write(model_card)
    
    print(f"Pushing model to {repo_name}...")
    
    # Define training arguments with push_to_hub enabled
    training_args = TrainingArguments(
        output_dir="./results",
        push_to_hub=True,
        hub_model_id=repo_name,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
    )
    
    # Push to hub
    trainer.push_to_hub()
    print("Model pushed successfully!")

# Main function
def main():
    # Set your Hugging Face username
    username = input("Enter your Hugging Face username: ")
    repo_name = f"{username}/distilhubert-finetuned-gtzan"
    
    # Login to Hugging Face Hub
    token = input("Enter your Hugging Face token (from https://huggingface.co/settings/tokens): ")
    if not login_to_hub(token):
        print("Failed to login. Please check your token and try again.")
        return
    
    # Load model and feature extractor from checkpoint
    model, feature_extractor = load_model_from_checkpoint()
    
    # Push model to hub
    push_model_to_hub(model, feature_extractor, repo_name)
    
    print(f"\nYour model is now available at: https://huggingface.co/{repo_name}")
    print("You can use it with the transformers pipeline:")
    print(f"\nfrom transformers import pipeline")
    print(f"classifier = pipeline('audio-classification', model='{repo_name}')")
    print(f"result = classifier('path/to/audio.wav')")
    print(f"print(result)")

if __name__ == "__main__":
    main()