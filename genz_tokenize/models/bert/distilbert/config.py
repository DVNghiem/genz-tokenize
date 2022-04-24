from genz_tokenize.models.bert.model_utils import Config


class DistilBertConfig(Config):
    def __init__(
        self,
        vocab_size: int = 33333,
        hidden_dim: int = 256,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 1024,
        num_labels: int = 2,
        initializer_range: float = 0.02,
        dropout: float = 0.1,
        attention_dropout: float = 0.2,
        epsilon: float = 1e-6,
        dim: int = 1024,
        initial_range: float = 0.02,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_labels = num_labels
        self.initializer_range = initializer_range
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.epsilon = epsilon
        self.dim = dim
        self.initial_range = initial_range

    def __str__(self) -> str:
        return 'DistilBert Config'
