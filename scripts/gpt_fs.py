import torch
import torch.nn as nn
from transformers import AutoTokenizer


# things we need
# 1. Embedding
# 2. Transformer Block (Norm, MHSA, FFN)
# 3. Unembedding
# do llama2 implementation

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
print(tokenizer)