from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

class emerge_text(nn.Module):
    def __init__(
        self, 
        model_id="answerdotai/ModernBERT-base"
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)

    def get_embeddings(self, input_ids, attention_mask):
        """
        Get embeddings from pre-tokenized inputs.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
            
        Returns:
            cls_embeddings: Tensor of shape (batch_size, hidden_size)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Get CLS token embeddings (first token) for each item in the batch
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings
    
    def forward(self, input_ids, attention_mask):
        return self.get_embeddings(input_ids, attention_mask)

if __name__ == "__main__":
    # Test the emerge_text model
    model = emerge_text(model_id="answerdotai/ModernBERT-base")
    sample_texts = ["Hello, how are you?", "This is a test sentence."]
    encoded_inputs = model.tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        embeddings = model(
            input_ids=encoded_inputs['input_ids'],
            attention_mask=encoded_inputs['attention_mask']
        )
    print("Embeddings shape:", embeddings.shape)  # Should be (batch_size, hidden_size)