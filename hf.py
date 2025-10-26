from transformers import AutoTokenizer, AutoModel
import torch

class emerge_text():
    def __init__(
        self, 
        model_id="answerdotai/ModernBERT-base"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[0, 0, :]
        return cls_embedding
    
    def forward(self, text):
        return self.get_embeddings(text)
