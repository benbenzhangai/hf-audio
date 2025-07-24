# Music Genre Classification Demo

This demo uses a DistilHuBERT model fine-tuned on the GTZAN dataset to classify music genres from audio files or microphone recordings.

## Features
- Upload or record audio in the browser
- Classifies audio into one of 10 music genres
- Example files included for testing

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Install ffmpeg if not already installed:
   ```bash
   sudo apt-get update && sudo apt-get install ffmpeg
   ```

## Usage
Run the Gradio app:
```bash
python app.py
```

Open the provided link in your browser to use the demo.

## Files
- `app.py`: Gradio web app
- `test_model.py`: Script to test the model on sample files
- `metal-sample-1.wav`, `metal-sample-2.wav`: Example audio files

## Notes
- The model is trained for music genre classification. Voice recordings will still be assigned a genre, but results may not be meaningful.
- Do **not** commit the `.gradio/` directory or other temporary files. 