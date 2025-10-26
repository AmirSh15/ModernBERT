from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    """
    SwiGLU activation function: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.up = nn.Linear(input_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, output_dim)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        gate = self.silu(self.gate(x))
        up = self.up(x)
        return self.down(gate * up)

class emerge_text(nn.Module):
    def __init__(
        self, 
        model_id="answerdotai/ModernBERT-base",
        use_projection=False,
        output_dim=768
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        
        # Get the input dimension from the model config
        self.input_dim = self.model.config.hidden_size
        
        # Optional projection layer with SwiGLU activation
        self.use_projection = use_projection
        if use_projection:
            self.projection = SwiGLU(
                input_dim=self.input_dim,
                hidden_dim=self.input_dim*4,
                output_dim=output_dim
            )

    def get_embeddings(self, input_ids, attention_mask):
        """
        Get embeddings from pre-tokenized inputs.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
            
        Returns:
            cls_embeddings: Tensor of shape (batch_size, output_dim)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Get CLS token embeddings (first token) for each item in the batch
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Apply projection if enabled
        if self.use_projection:
            cls_embeddings = self.projection(cls_embeddings)
        
        return cls_embeddings
    
    def forward(self, input_ids, attention_mask):
        return self.get_embeddings(input_ids, attention_mask)

if __name__ == "__main__":
    # Test the emerge_text model without projection
    print("Testing without projection:")
    model = emerge_text(model_id="answerdotai/ModernBERT-base", use_projection=False)
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
    print("Embeddings shape:", embeddings.shape)  # Should be (batch_size, 768)
    
    # Test the emerge_text model with projection
    print("\nTesting with SwiGLU projection:")
    model_proj = emerge_text(
        model_id="answerdotai/ModernBERT-base", 
        use_projection=True,
        projection_hidden_dim=3072,
        output_dim=768
    )
    with torch.no_grad():
        embeddings_proj = model_proj(
            input_ids=encoded_inputs['input_ids'],
            attention_mask=encoded_inputs['attention_mask']
        )
    print("Projected embeddings shape:", embeddings_proj.shape)  # Should be (batch_size, 768)