import numpy as np
from _static import Tokenizer, BatchTokenizer, Embedder, TokenizerModel


class EmbeddingPipeline:
    def __init__(
        self,
        embeddings_path: str = "",
        max_token_id: int = 0,
        n_vocab: int = 200000,
        tokenizer_path: str = "",
        model: TokenizerModel = TokenizerModel.O200K_BASE,
    ):
        if tokenizer_path:
            self._tokenizer = Tokenizer(tokenizer_path)
        else:
            self._tokenizer = Tokenizer(model)

        self._embedder = Embedder(n_vocab)
        self._embedder.set_tokenizer(self._tokenizer)

        if embeddings_path:
            self._embedder.load_binary(embeddings_path, max_token_id)

    @property
    def embedding_dim(self) -> int:
        return self._embedder.embedding_dim

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def encode(self, text: str) -> np.ndarray:
        tokens = self._tokenizer.encode(text)
        return self._embedder.get_embedding_from_tokens(tokens)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        token_batches = [self._tokenizer.encode(t) for t in texts]
        embeddings = self._embedder.get_embeddings_from_token_batches(token_batches)
        return np.array(embeddings)

    def decode(self, tokens) -> str:
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return self._tokenizer.decode(tokens)


class BatchedEmbeddingPipeline:
    def __init__(
        self,
        embeddings_path: str = "",
        max_token_id: int = 0,
        n_vocab: int = 200000,
        tokenizer_path: str = "",
        model: TokenizerModel = TokenizerModel.O200K_BASE,
        batch_size: int = 32,
    ):
        if tokenizer_path:
            self._tokenizer = Tokenizer(tokenizer_path)
        else:
            self._tokenizer = Tokenizer(model)

        self._batch_tokenizer = BatchTokenizer(
            model if not tokenizer_path else tokenizer_path
        )
        self._embedder = Embedder(n_vocab)
        self._embedder.set_tokenizer(self._tokenizer)

        if embeddings_path:
            self._embedder.load_binary(embeddings_path, max_token_id)

        self._batch_size = batch_size

    @property
    def embedding_dim(self) -> int:
        return self._embedder.embedding_dim

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            token_batches = self._batch_tokenizer.encode(batch)
            embeddings = self._embedder.get_embeddings_from_token_batches(token_batches)
            all_embeddings.extend(embeddings)
        return [np.array(e) for e in all_embeddings]

    def decode(self, tokens) -> str:
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return self._tokenizer.decode(tokens)
