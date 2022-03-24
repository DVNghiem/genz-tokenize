import tensorflow as tf
import numpy as np
from .utils import Config, get_initial_params, scaled_dot_product_attention


class EncoderSeq2Seq(tf.keras.layers.Layer):
    def __init__(self, config: Config):
        super(EncoderSeq2Seq, self).__init__()
        self.enc_units = config.units
        self.embedding = tf.keras.layers.Embedding(
            config.vocab_size, config.hidden_size)

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        # x shape bs, maxlen
        x = self.embedding(x)  # bs, maxlen, d_model
        output, state = self.gru(x, initial_state=hidden)  # bs, maxlen, units
        return output, state


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, config: Config):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(config.units)
        self.W2 = tf.keras.layers.Dense(config.units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector


class DecoderSeq2Seq(tf.keras.layers.Layer):
    def __init__(self, config: Config):
        super(DecoderSeq2Seq, self).__init__()
        """
            target_vocab_size: 2 different languages
        """
        self.dec_units = config.units
        self.embedding = tf.keras.layers.Embedding(
            config.target_vocab_size, config.hidden_size)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(
            config.target_vocab_size, kernel_initializer=get_initial_params(config))

        self.attention = BahdanauAttention(config)

    def call(self, x, hidden, enc_output):
        context_vector = self.attention(hidden, enc_output)
        x = self.embedding(x)  # bs, 1, d_model
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)   # bs, 1, units
        output = tf.reshape(output, (-1, output.shape[2]))  # bs, units
        x = self.fc(output)
        return x, state


class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size, maxlen) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)
        self.pos_emb = tf.keras.layers.Embedding(maxlen, hidden_size)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        position = tf.range(start=0, limit=maxlen, delta=1)
        position = self.pos_emb(position)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.hidden_size, tf.float32))
        return x+position


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output


def point_wise_feed_forward_network(d_model, dff, activation):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=activation),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: Config):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(
            config.hidden_size, config.num_heads)
        self.ffn = point_wise_feed_forward_network(
            config.hidden_size, config.dff, config.hidden_activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=config.layerNorm_epsilon)
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=config.layerNorm_epsilon)

        self.dropout1 = tf.keras.layers.Dropout(config.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(config.dropout_rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: Config):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(
            config.hidden_size, config.num_heads)
        self.mha2 = MultiHeadAttention(
            config.hidden_size, config.num_heads)

        self.ffn = point_wise_feed_forward_network(
            config.hidden_size, config.dff, config.hidden_activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=config.layerNorm_epsilon)
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=config.layerNorm_epsilon)
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=config.layerNorm_epsilon)

        self.dropout1 = tf.keras.layers.Dropout(config.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(config.dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(config.dropout_rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config: Config):
        super(Encoder, self).__init__()

        self.d_model = config.hidden_size
        self.num_layers = config.num_hidden_layers

        self.embedding = PositionEmbedding(
            config.vocab_size, config.hidden_size, config.maxlen)

        self.enc_layers = [EncoderLayer(config)
                           for _ in range(config.num_hidden_layers)]

        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(self, x, training, mask):

        x = self.embedding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config: Config):
        super(Decoder, self).__init__()

        self.num_layers = config.num_hidden_layers

        self.embedding = PositionEmbedding(
            config.target_vocab_size, config.hidden_size, config.maxlen)

        self.dec_layers = [DecoderLayer(config)
                           for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):

        x = self.embedding(x)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training,
                                   look_ahead_mask, padding_mask)

        return x
