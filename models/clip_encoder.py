import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP text encoder to encode text into embeddings.
    The model is frozen (all parameters require_grad=False).
    """
    def __init__(self, version="openai/clip-vit-base-patch32", device="cuda", max_length=77, return_sequence=True):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.return_sequence = return_sequence
        self.freeze()

    def freeze(self):
        """Freeze all parameters"""
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        """
        Args:
            text: list of strings or a single string

        Returns:
            If return_sequence=True: dict with:
                'tokens': (B, seq_len, 512)
                'pool': (B, 512)
                'token_ids': (B, seq_len)

            If return_sequence=False:
                pooled embeddings only (B, 512)
        """
        if isinstance(text, str):
            text = [text]

        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt"
        )

        tokens = batch_encoding["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.transformer(input_ids=tokens)

        if self.return_sequence:
            return {
                'tokens': outputs.last_hidden_state,   # (B, seq_len, 512)
                'pool': outputs.pooler_output,         # (B, 512)
                'token_ids': tokens                    # (B, seq_len)
            }
        else:
            return outputs.pooler_output

    def encode(self, text):
        """Alias for forward"""
        return self.forward(text)
