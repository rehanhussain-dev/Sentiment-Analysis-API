import json

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from .sentiment_classifier import SentimentClassifier

with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(config["BERT_MODEL"])

        classifier = SentimentClassifier(len(config["CLASS_NAMES"]))
        classifier.load_state_dict(
            torch.load(config["PRE_TRAINED_MODEL"], map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        # Using explicit .encode() to completely isolate and avoid 'encode_plus' routing
        input_ids_list = self.tokenizer.encode(
            text,
            max_length=config["MAX_SEQUENCE_LEN"],
            add_special_tokens=True,
            padding="max_length",
            truncation=True
        )
        
        # Manually construct the attention mask (1 for real tokens, 0 for padded 0s)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        attention_mask_list = [1 if tid != pad_token_id else 0 for tid in input_ids_list]

        # Convert directly to PyTorch Tensors and send to your device
        input_ids = torch.tensor([input_ids_list]).to(self.device)
        attention_mask = torch.tensor([attention_mask_list]).to(self.device)

        with torch.no_grad():
            # Get the raw model predictions
            outputs = self.classifier(input_ids, attention_mask)
            probabilities = F.softmax(outputs, dim=1)
        
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        # Extract native types safely
        confidence_val = confidence.cpu().item() 
        predicted_class = predicted_class.cpu().item()
        probabilities_list = probabilities.flatten().cpu().numpy().tolist()
        
        return (
            config["CLASS_NAMES"][predicted_class],
            confidence_val,
            dict(zip(config["CLASS_NAMES"], probabilities_list)),
        )


model = Model()


def get_model():
    return model
