import tensorflow as tf
import numpy
from .config import RobertaConfig
from ...base_model.utils import shape_list, get_initial_params
import math


class Embedding(tf.keras.layers.Layer):

    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        self.padding_idx = 0
        self.vocab_size = config.vocab_size
        self.type_vocab_size = config.type_vocab_size
        self.hidden_size = config.hidden_size
        self.max_position_embedding = config.max_position_embedding
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initial_params(self.config),
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="token_embeddings",
                shape=[self.type_vocab_size, self.hidden_size],
                initializer=get_initial_params(self.config),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="position_embeddings",
                shape=[self.max_position_embedding, self.hidden_size],
                initializer=get_initial_params(self.config),
            )

        super().build(input_shape)

    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        mask = tf.cast(tf.math.not_equal(
            input_ids, self.padding_idx), dtype=input_ids.dtype)
        incremental_indices = (tf.math.cumsum(
            mask, axis=1) + past_key_values_length) * mask

        return incremental_indices + self.padding_idx

    def call(
        self,
        input_ids=None,
        token_type_ids=None,
        training=False,
    ):

        inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        position_ids = self.create_position_ids_from_input_ids(
            input_ids=input_ids, past_key_values_length=self.max_position_embedding
        )

        position_embeds = tf.gather(
            params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(
            params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(
            inputs=final_embeddings, training=training)

        return final_embeddings


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initial_params(config), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initial_params(config), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initial_params(config), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(
            rate=config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(
            batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_value,
        training=False,
    ):
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)

        is_cross_attention = encoder_hidden_states is not None  # using for decoder
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(inputs=encoder_hidden_states), batch_size)
            value_layer = self.transpose_for_scores(
                self.value(inputs=encoder_hidden_states), batch_size)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(
                self.key(inputs=hidden_states), batch_size)
            value_layer = self.transpose_for_scores(
                self.value(inputs=hidden_states), batch_size)
            key_layer = tf.concat([past_key_value[0], key_layer], axis=2)
            value_layer = tf.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(
                self.key(inputs=hidden_states), batch_size)
            value_layer = self.transpose_for_scores(
                self.value(inputs=hidden_states), batch_size)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFRobertaModel call() function)
            attention_scores = tf.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(
            inputs=attention_probs, training=training)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(
            tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output,)

        outputs = outputs + (past_key_value,)
        return outputs


class SelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initial_params(config), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class Attention(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = SelfAttention(config, name="self")
        self.dense_output = SelfOutput(config, name="output")

    def call(
        self,
        hidden_state,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_value,
        training=False,
    ):
        self_outputs = self.self_attention(
            hidden_states=hidden_state,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            training=training,
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=hidden_state, training=training
        )
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class Intermediate(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initial_params(config), name="dense"
        )

        self.intermediate_act_fn = tf.nn.gelu

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class RobertaOutput(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initial_params(config), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.bert_output = RobertaOutput(config)

    def call(
        self,
        hidden_state: tf.Tensor,
        attention_mask: tf.Tensor,
        past_key_value,
        training=False
    ):
        past_key_value = past_key_value[:2] if past_key_value is not None else None

        attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        attention_mask = tf.cast(attention_mask, dtype=hidden_state.dtype)
        one_cst = tf.constant(1.0, dtype=attention_mask.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=attention_mask.dtype)
        attention_mask = tf.multiply(tf.subtract(
            one_cst, attention_mask), ten_thousand_cst)
        attention_output = self.attention(
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            training=training
        )

        present_key_value = attention_output[-1]
        attention_output = attention_output[0]
        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        return layer_output, present_key_value


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = [EncoderLayer(config)
                       for _ in range(config.num_hidden_layers)]

    def call(
        self,
            input_ids=None,
            attention_mask=None,
            training=False

    ):
        hidden_state = input_ids
        past_key_value = None
        for layer in self.layers:
            out = layer(hidden_state,
                        attention_mask,
                        past_key_value=past_key_value,
                        training=training)
            hidden_state, past_key_value = out

        return hidden_state


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = Attention(config, name="attention")
        config.is_decoder = True
        self.crossattention = Attention(config, name="crossattention")
        self.intermediate = Intermediate(config, name="intermediate")
        self.bert_output = RobertaOutput(config, name="output")

    def call(
        self,
        hidden_state=None,
        attention_mask=None,
        encoder_output=None,
        encoder_attention_mask=None,
        past_key_value=None,
        training=False
    ):
        past_key_value = past_key_value[:2] if past_key_value is not None else None

        attention_mask = tf.cast(
            attention_mask, dtype=hidden_state.dtype)
        one_cst = tf.constant(1.0, dtype=attention_mask.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=attention_mask.dtype)
        attention_mask = tf.multiply(tf.subtract(
            one_cst, attention_mask), ten_thousand_cst)
        attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            past_key_value=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            training=training
        )
        attention_output = attention_output[0]
        encoder_attention_mask = tf.cast(
            encoder_attention_mask, dtype=hidden_state.dtype)
        encoder_attention_mask = encoder_attention_mask[:,
                                                        tf.newaxis, tf.newaxis, :]
        encoder_attention_mask = (1.0 - encoder_attention_mask) * -100000.0

        cross_attention_outputs = self.crossattention(
            hidden_state=attention_output,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            training=training,
        )

        hidden_state, past_key_value = cross_attention_outputs
        return hidden_state, past_key_value


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = [DecoderLayer(config)
                       for _ in range(config.num_hidden_layers)]

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_output=None,
        encoder_attention_mask=None,
        past_key_value=None,
        training=False
    ):
        hidden_state = input_ids
        for layer in self.layers:
            out = layer(
                hidden_state=hidden_state,
                attention_mask=attention_mask,
                encoder_output=encoder_output,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                training=training
            )
            hidden_state, past_key_value = out

        return hidden_state
