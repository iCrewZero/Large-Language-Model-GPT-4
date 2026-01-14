# GPT-4 Style Large Language Model

Welcome to my custom **GPT-4 inspired LLM** repository! This project includes everything from **training** to **efficient inference**, with advanced features like **paged KV caching**, **hybrid INT8-FP16 keys and values**, **YaRN RoPE embeddings**, and **speculative decoding** for faster generation.  

Think of this as a full-stack LLM repo, from dataset preprocessing to inference-ready deployment.

---

## **Repo Structure**

/config
model.yaml # Model hyperparameters
train.yaml # Training parameters

/deepspeed
ds_zero2.json # DeepSpeed configuration for distributed training

/eval
gsm8.py # GSM8K evaluation script
perplexity.py # Perplexity evaluation script

/inference
paged_kv.py # Paged KV allocator + helpers
attention.py # Inference attention layer (hybrid KV, RoPE)
block.py # Transformer block (inference)
transformer.py # Full transformer model (inference)
speculative.py # Speculative decoding logic

/model
attention.py # Training attention layer
blocks.py # Transformer block for training
gpt.py # Model wrapper for training
moe.py # MoE layers (optional)
rmsnorm.py # RMSNorm implementation
rope.py # RoPE helpers

/scripts
eval.py # Evaluate trained model
launch_ddp.sh # Launch DDP training
lauch_deepspeed.sh # Launch DeepSpeed training
preprocess_data.py # Dataset preprocessing

/tokenizer
tokenizer.py # Tokenizer for training and inference
train_tokenizer.py # Tokenizer training script

/training
checkpoint.py # Save/load checkpoints
dataset.py # Dataset loader
distributed.py # Distributed training helpers
loss_mask.py # Loss masking
optimizer.py # Optimizer wrapper
scheduler.py # Learning rate scheduler
train.py # Training loop

yaml
Copy code

---

## **Installation**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch deepspeed numpy transformers
GPU is highly recommended for both training and inference (CUDA >= 11.7).

Training
Configure model and training parameters in /config/model.yaml and /config/train.yaml.

Prepare your dataset using /scripts/preprocess_data.py.

Launch distributed training:

bash
Copy code
bash scripts/lauch_deepspeed.sh
# or
python -m torch.distributed.run --nproc_per_node=8 scripts/train.py
Model checkpoints are saved in /training/checkpoints/.

Inference
This repository separates training and inference pipelines for efficiency.

Initialize KV pools:

python
Copy code
from inference.paged_kv import PageAllocator, KVState, PAGE_SIZE, MAX_PAGES
import torch

device = "cuda"
K_pool = torch.zeros(MAX_PAGES,16,PAGE_SIZE,128,device=device)
V_pool = torch.zeros(MAX_PAGES,16,PAGE_SIZE,128,device=device)
allocator = PageAllocator(MAX_PAGES)
state = KVState()
Load the transformer model:

python
Copy code
from inference.transformer import Transformer
model = Transformer().to(device)
Feed tokenized input through the model:

python
Copy code
logits = model(input_ids, state, K_pool, V_pool, allocator)
Optional: use speculative decoding for faster generation:

python
Copy code
from inference.speculative import speculative_decode
output_ids = speculative_decode(target_model, draft_model, input_ids, max_new_tokens=50)
Features
Paged KV Cache: Efficient memory management for long-context sequences (~32k tokens)

Hybrid KV: Keys quantized to INT8, Values stored in FP16

YaRN RoPE: Frequency-scaled Rotary embeddings for long sequences

Speculative Decoding: Uses draft and target models to speed up generation

FlashAttention Support: If available, boosts attention speed

Training vs Inference Split: /model contains training modules; /inference optimized for deployment

Tips for Developers
Always match the tokenizer with your model configuration.

For long-context generation, use 32k block size.

Use GPU memory monitoring when increasing BLOCK_SIZE or sequence length.

Hybrid KV cache reduces memory usage ~50% without major quality loss.

Contributing
Fork the repository

Add features (datasets, attention variants, inference optimizations)

Test thoroughly and submit PRs with documentation
