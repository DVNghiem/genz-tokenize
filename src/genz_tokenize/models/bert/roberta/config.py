class Config:
    def __init__(
        self,
        vocab_size=33333,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        max_position_embedding=1024,
        num_labels=2
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embedding = max_position_embedding
        self.num_labels = num_labels

    def __str__(self) -> str:
        return 'RoBerta Config'
