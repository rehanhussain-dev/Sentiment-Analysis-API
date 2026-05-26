import json

from torch import nn
from transformers import BertModel

with open("config.json") as json_file:
    config = json.load(json_file)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config["BERT_MODEL"])
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # 1. Get the modern output object from BERT
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 2. Explicitly grab the pooled tensor by its attribute name
        pooled_output = bert_outputs.pooler_output
        
        # 3. Pass the actual tensor to your dropout and linear layers
        output = self.drop(pooled_output)
        return self.out(output)