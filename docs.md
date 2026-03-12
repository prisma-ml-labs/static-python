# Static (python bindings)
## Installation

```bash
git clone --recurse-submodules https://github.com/prisma-ml-labs/static-python.git
cd static-python

pip install -e .
```

Or install using uv
```bash
uv add git+https://github.com/prisma-ml-labs/static-python.git
```

## Quick Start

```python
from _pipeline import EmbeddingPipeline
import numpy as np

p = EmbeddingPipeline(
    embeddings_path="embeddings.bin",
    max_token_id=200000
)

emb = p.encode("hello world")
print(emb.shape)  # (1536,)

texts = ["hello", "world", "test"]
embeddings = p.encode_batch(texts)
print(embeddings.shape)  # (3, 1536)

tokens = p._tokenizer.encode("hello")
decoded = p.decode(tokens)
```

## API

### EmbeddingPipeline

```python
EmbeddingPipeline(
    embeddings_path: str = "",    # Path to binary embeddings file
    max_token_id: int = 0,        # Maximum token ID
    n_vocab: int = 200000,        # Vocabulary size
    tokenizer_path: str = "",      # Custom tokenizer JSON (optional)
    model: TokenizerModel = TokenizerModel.O200K_BASE
)
```

**Properties:**
- `embedding_dim` - Dimension of embeddings
- `vocab_size` - Vocabulary size

**Methods:**
- `encode(text: str) -> np.ndarray` - Get embedding for single text
- `encode_batch(texts: list[str]) -> np.ndarray` - Get embeddings for batch
- `decode(tokens) -> str` - Decode tokens back to text

### BatchedEmbeddingPipeline

Batched version that processes in chunks for memory efficiency:

```python
BatchedEmbeddingPipeline(
    embeddings_path: str = "",
    max_token_id: int = 0,
    n_vocab: int = 200000,
    tokenizer_path: str = "",
    model: TokenizerModel = TokenizerModel.O200K_BASE,
    batch_size: int = 32
)
```

### Low-level API

For finer control, use the raw C++ bindings directly:

```python
from _static import Tokenizer, BatchTokenizer, Embedder, TokenizerModel

# Tokenizer
t = Tokenizer(TokenizerModel.O200K_BASE)
tokens = t.encode("hello world")
decoded = t.decode(tokens)

# Batch tokenizer
bt = BatchTokenizer()
batches = bt.encode(["hello", "world"])

# Embedder
e = Embedder()
e.set_tokenizer(t)
emb = e.get_embedding_from_tokens(tokens)
```

## Building from Source

```bash
# Install build dependencies
pip install pybind11 numpy

# Build bindings
python3 build_bindings.py

# Or install in development mode
pip install -e .
```

## Tokenizer Models

Available models:
- `TokenizerModel.R50K_BASE` - GPT-1/2 tokenizer
- `TokenizerModel.P50K_BASE` - GPT-3 (base)
- `TokenizerModel.P50K_EDIT` - GPT-3 (edit)
- `TokenizerModel.CL100K_BASE` - Code and text
- `TokenizerModel.O200K_BASE` - GPT-4 (default)
- `TokenizerModel.O200K_HARMONY` - GPT-4o

## File Formats

### Embeddings Binary

The `embeddings.bin` file format:
- Header: 4 bytes (uint32) = embedding dimension
- Body: int16 values, shape (n_vocab, embedding_dim)

Convert from `.emb` file:
```python
e = Embedder()
e.load_embeddings("embeddings.emb")
e.save_binary("embeddings.bin")
```
