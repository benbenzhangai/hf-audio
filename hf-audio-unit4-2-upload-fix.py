## Re-upload the preprocessor_config.json
from transformers import AutoFeatureExtractor

CHECKPOINT_PATH = "/teamspace/studios/this_studio/hf-audio/distilhubert-finetuned-gtzan/checkpoint-1130"
feature_extractor = AutoFeatureExtractor.from_pretrained(CHECKPOINT_PATH)
feature_extractor.push_to_hub("BenbenbenZZZ/distilhubert-finetuned-gtzan")