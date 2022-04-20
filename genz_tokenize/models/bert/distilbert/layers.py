import tensorflow as tf
from .config import DistilBertConfig
from ...base_model.utils import shape_list, get_initial_params


class Embeddings(tf.keras.layers.Layer):
    def __init__(self, config: DistilBertConfig, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.config = config
        self.dim = config.dim
        self.initializer_range = config
        self.max_position_embeddings = config.max_position_embeddings
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.epsilon, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.dropout)

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.vocab_size, self.dim],
                initializer=get_initial_params(
                    config=self.config),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.dim],
                initializer=get_initial_params(
                    config=self.config),
            )

        super().build(input_shape)

    def call(self, input_ids=None, training=False):

        if input_ids is not None:
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        position_ids = tf.expand_dims(
            tf.range(start=0, limit=input_shape[-1]), axis=0)

        position_embeds = tf.gather(
            params=self.position_embeddings, indices=position_ids)
        final_embeddings = inputs_embeds + position_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(
            inputs=final_embeddings, training=training)

        return final_embeddings


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: DistilBertConfig, is_look_mask=False, ** kwargs) -> None:
        super().__init__(**kwargs)

        self.num_attention_heads = config.num_attention_heads
        self.dim = config.dim
        self.is_look_mask = is_look_mask
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout)

        assert self.dim % self.num_attention_heads == 0, f"Hidden size {self.dim} not dividable by number of heads {self.num_attention_heads}"

        self.q_lin = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initial_params(config), name="q_lin"
        )
        self.k_lin = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initial_params(config), name="k_lin"
        )
        self.v_lin = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initial_params(config), name="v_lin"
        )
        self.out_lin = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initial_params(config), name="out_lin"
        )

    def call(self, query, key, value, mask, training=False):
        """
        Parameters:
            query: tf.Tensor(bs, seq_length, dim)
            key: tf.Tensor(bs, seq_length, dim)
            value: tf.Tensor(bs, seq_length, dim)
            mask: tf.Tensor(bs, seq_length)
        Returns:
           context (bs, q_length, dim)
        """
        bs, q_length, dim = shape_list(query)
        dim_per_head = int(self.dim / self.num_attention_heads)
        dim_per_head = tf.cast(dim_per_head, dtype=tf.int32)

        def create_look_ahead_mask(size):
            mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
            return tf.cast(mask, tf.int32)

        def shape(x):
            """separate heads"""
            return tf.transpose(tf.reshape(x, (bs, -1, self.num_attention_heads, dim_per_head)), perm=(0, 2, 1, 3))

        def unshape(x):
            """group heads"""
            return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.num_attention_heads * dim_per_head))

        # (bs, num_attention_heads, q_length, dim_per_head)
        q = shape(self.q_lin(query))
        # (bs, num_attention_heads, k_length, dim_per_head)
        k = shape(self.k_lin(key))
        # (bs, num_attention_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))
        q = tf.cast(q, dtype=tf.float32)
        q = tf.multiply(q, tf.math.rsqrt(
            tf.cast(dim_per_head, dtype=tf.float32)))
        k = tf.cast(k, dtype=q.dtype)
        # (bs, num_attention_heads, q_length, k_length)
        scores = tf.matmul(q, k, transpose_b=True)

        mask = tf.reshape(mask, shape=(bs, 1, 1, -1))
        if self.is_look_mask:
            look_mask = create_look_ahead_mask(q_length)
            mask = tf.maximum(mask, look_mask)

        # scores.masked_fill_(mask, -float('inf'))            # (bs, num_attention_heads, q_length, k_length)
        mask = tf.cast(mask, dtype=scores.dtype)
        scores = scores - 1e30 * (1.0 - mask)
        # (bs, num_attention_heads, qlen, klen)
        weights = tf.nn.softmax(scores, axis=-1)
        # (bs, num_attention_heads, qlen, klen)
        weights = self.dropout(weights, training=training)

        # (bs, num_attention_heads, qlen, dim_per_head)
        context = tf.matmul(weights, v)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        return context


class FFN(tf.keras.layers.Layer):
    def __init__(self, config: DistilBertConfig, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.linear1 = tf.keras.layers.Dense(
            config.hidden_dim, kernel_initializer=get_initial_params(config), name="linear1"
        )
        self.linear2 = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initial_params(config), name="linear2"
        )
        self.activation = tf.nn.gelu

    def call(self, input, training=False):
        x = self.linear1(input)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x, training=training)
        return x


class DistilEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: DistilBertConfig, **kwargs):
        super().__init__(**kwargs)

        self.num_attention_heads = config.num_attention_heads
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dropout = tf.keras.layers.Dropout(config.dropout)

        assert (
            config.dim % config.num_attention_heads == 0
        ), f"Hidden size {config.dim} not dividable by number of heads {config.num_attention_heads}"

        self.attention = MultiHeadSelfAttention(config, name="attention")
        self.layerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.epsilon, name="layerNorm")

        self.ffn = FFN(config, name="ffn")
        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.epsilon, name="output_layer_norm")

    def call(self, x, attn_mask, training=False):
        """
        Parameters:
            x: tf.Tensor(bs, seq_length, dim)
            attn_mask: tf.Tensor(bs, seq_length)
        Outputs: hidden state tf.Tensor(bs, seq_length, dim)
        """
        # Self-Attention
        sa_output = self.attention(x, x, x, attn_mask, training=training)
        sa_output = self.layerNorm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        # (bs, seq_length, dim)
        ffn_output = self.ffn(sa_output, training=training)
        ffn_output = self.output_layer_norm(
            ffn_output + sa_output)  # (bs, seq_length, dim)

        return ffn_output


class DistilDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: DistilBertConfig, **kwargs):
        super().__init__(**kwargs)

        self.num_attention_heads = config.num_attention_heads
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dropout = tf.keras.layers.Dropout(config.dropout)

        assert (
            config.dim % config.num_attention_heads == 0
        ), f"Hidden size {config.dim} not dividable by number of heads {config.num_attention_heads}"

        self.looking_attention = MultiHeadSelfAttention(
            config, is_look_mask=True, name="looking_attention")

        self.attention = MultiHeadSelfAttention(config, name='attention')

        self.layerNorm1 = tf.keras.layers.LayerNormalization(
            epsilon=config.epsilon, name="layerNorm1")

        self.layerNorm2 = tf.keras.layers.LayerNormalization(
            epsilon=config.epsilon, name="layerNorm2")

        self.layerNorm3 = tf.keras.layers.LayerNormalization(
            epsilon=config.epsilon, name="layerNorm3")

        self.ffn = FFN(config, name="ffn")
        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.epsilon, name="output_layer_norm")

    def call(self, x, attn_mask, encoder_hidden_state, training=False):
        """
        Parameters:
            x: tf.Tensor(bs, seq_length, dim)
            attn_mask: tf.Tensor(bs, seq_length)
            encoder_hidden_state: tf.Tensor(bs, seq_length, dim)
        Outputs: hidden state 
        """

        # look-Attention
        look = self.looking_attention(
            x, x, x, mask=attn_mask, training=training)

        norm = self.layerNorm1(look+x)

        attention = self.attention(
            norm, encoder_hidden_state, encoder_hidden_state, mask=attn_mask, training=training)

        norm = self.layerNorm2(attention+norm)

        ffn = self.ffn(norm)
        return self.layerNorm3(ffn+norm)


class DistilEncoder(tf.keras.layers.Layer):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers

        self.layers = [DistilEncoderLayer(config, name=f'distil_encoder_{_}')
                       for _ in range(self.num_hidden_layers)]

    def call(self, x, attention_mask, training):
        out = x
        for i in range(self.num_hidden_layers):
            out = self.layers[i](out, attention_mask, training)
        return out


class DistilDecoder(tf.keras.layers.Layer):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers

        self.layers = [DistilDecoderLayer(config)
                       for _ in range(self.num_hidden_layers)]

    def call(self, x, attention_mask, encoder_hidden_state, training):
        out = x
        for i in range(self.num_hidden_layers):
            out = self.layers[i](out, attention_mask,
                                 encoder_hidden_state, training)
        return out
