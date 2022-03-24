import tensorflow as tf
import os


class Config:
    def __init__(self,
                 vocab_size=48000,
                 target_vocab_size=48000,
                 hidden_size=512,
                 units=512,
                 dropout_rate=0.1,
                 initial_range=0.02,
                 hidden_activation='relu',
                 num_hidden_layers=8,
                 num_heads=8,
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 maxlen=128,
                 dff=1024,
                 layerNorm_epsilon=1e-12,
                 num_class=2,
                 ** kwargs
                 ) -> None:
        self.vocab_size = vocab_size
        self.target_vocab_size = target_vocab_size
        self.hidden_size = hidden_size
        self.units = units
        self.dropout_rate = dropout_rate
        self.initial_range = initial_range
        self.hidden_activation = hidden_activation
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.maxlen = maxlen
        self.dff = dff
        self.layerNorm_epsilon = layerNorm_epsilon
        self.num_class = num_class


class CallbackSave(tf.keras.callbacks.Callback):
    def __init__(self, model_dir: str, save_per_epochs: int = 1) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.save_per_epochs = save_per_epochs

    def on_epoch_end(self, epochs, logs=None):
        if epochs % self.save_per_epochs == 0:
            self.model.save_weights(os.path.join(
                self.model_dir, 'epochs_%s' % epochs))


def get_initial_params(config: Config):
    initial = tf.keras.initializers.TruncatedNormal(
        stddev=config.initial_range)
    return initial


def loss_seq2seq(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def loss_transformer(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # depth
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
