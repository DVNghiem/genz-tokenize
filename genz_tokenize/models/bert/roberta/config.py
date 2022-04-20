from genz_tokenize.models.bert.model_utils import Config


class RobertaConfig(Config):
    def __init__(
        self,
        vocab_size=33333,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embedding=1024,
        num_labels=2,
        type_vocab_size=1,
        initial_range=0.002,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-6,
        attention_probs_dropout_prob=0.1,
        is_decoder=False
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embedding = max_position_embedding
        self.num_labels = num_labels
        self.type_vocab_size = type_vocab_size
        self.initial_range = initial_range
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.is_decoder = is_decoder
