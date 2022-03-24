import tensorflow as tf
import numpy as np

from .layers import EncoderSeq2Seq, DecoderSeq2Seq, Encoder, Decoder
from .utils import Config, create_look_ahead_mask, create_padding_mask


class Seq2Seq(tf.keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = EncoderSeq2Seq(config)
        self.decoder = DecoderSeq2Seq(config)

        self.enc_unit = config.units
        self.dec_unit = config.units
        self.start_id = config.bos_token_id
        self.end_id = config.eos_token_id
        self.maxlen = config.maxlen

    def compile(self, loss, optimizer):
        super().compile()
        self.loss = loss
        self.optimizer = optimizer
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name='acc', dtype=tf.float32
        )
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name='val_acc', dtype=tf.float32
        )

    @property
    def metrics(self):
        return [self.acc, self.val_acc]

    @tf.function
    def train_step(self, data):
        x, y = data
        loss = 0
        hidden = self.initialize_hidden_state(self.enc_unit, self.batch_size)

        with tf.GradientTape() as tape:
            enc_output, hidden = self.encoder(x, hidden)
            dec_input = tf.expand_dims([self.start_id] * self.batch_size, 1)
            for t in range(1, y.shape[1]):
                predictions, hidden = self.decoder(
                    dec_input, hidden, enc_output)
                loss += self.loss(y[:, t], predictions)
                dec_input = tf.expand_dims(y[:, t], 1)
                self.acc.update_state(y[:, t], predictions)
        batch_loss = (loss / int(y.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return {'loss': batch_loss, 'accuracy': self.acc.result()}

    @tf.function
    def test_step(self, data):
        x, y = data
        loss = 0
        hidden = self.initialize_hidden_state(self.enc_unit, self.batch_size)
        enc_output, hidden = self.encoder(x, hidden)
        dec_input = tf.expand_dims([self.start_id] * self.batch_size, 1)
        for t in range(1, y.shape[1]):
            predictions, hidden = self.decoder(
                dec_input, hidden, enc_output)
            loss += self.loss(y[:, t], predictions)
            dec_input = tf.expand_dims(y[:, t], 1)
            self.val_acc.update_state(y[:, t], predictions)
        return {'val_loss': loss, 'val_acc': self.val_acc.result()}

    def predict(self, x):
        bs = x.shape[0]
        result = np.zeros(shape=(bs, self.maxlen))
        hidden = self.initialize_hidden_state(self.enc_unit, bs)
        enc_out, hidden = self.encoder(x, hidden)
        dec_input = tf.expand_dims([self.start_id]*bs, -1)
        for i in range(self.maxlen):
            predictions, hidden = self.decoder(dec_input,
                                               hidden,
                                               enc_out)

            predicted_id = tf.argmax(predictions, axis=-1).numpy()
            result[:, i] = predicted_id
            dec_input = tf.expand_dims(predicted_id, -1)
        return result

    def initialize_hidden_state(self, hidden, batch_size):
        return tf.zeros(shape=(batch_size, hidden))

    def __str__(self) -> str:
        return 'seq2seq'


class Transformer(tf.keras.Model):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.final_layer = tf.keras.layers.Dense(config.target_vocab_size)

    def call(self, x):
        inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask = x
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)

        return final_output

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        dec_padding_mask = create_padding_mask(inp)

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask

    def compile(self, loss, optimizer):
        super().compile()
        self.loss = loss
        self.optimizer = optimizer
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name='acc', dtype=tf.float32
        )
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name='val_acc', dtype=tf.float32
        )

    @property
    def metrics(self):
        return [self.acc, self.val_acc]

    @tf.function
    def train_step(self, data):
        inp, tar = data
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
            inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions = self((inp, tar_inp,
                               True,
                               enc_padding_mask,
                               combined_mask,
                               dec_padding_mask))
            loss = self.loss(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        self.acc.update_state(tar_real, predictions)
        return {'loss': loss, 'accuracy': self.acc.result()}

    @tf.function
    def test_step(self, data):
        inp, tar = data
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
            inp, tar_inp)
        predictions = self((inp, tar_inp,
                            False,
                            enc_padding_mask,
                            combined_mask,
                            dec_padding_mask))
        loss = self.loss(tar_real, predictions)

        self.val_acc.update_state(tar_real, predictions)
        return {'val_loss': loss, 'val_acc': self.val_acc.result()}

    def predict(self, x):
        bs = x.shape[0]
        result = np.zeros(shape=(bs, self.config.maxlen))
        output = tf.expand_dims([self.config.bos_token_id]*bs, axis=-1)
        for i in range(self.config.maxlen):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
                x, output)
            predictions = self((x,
                                output,
                                False,
                                enc_padding_mask,
                                combined_mask,
                                dec_padding_mask))
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(
                tf.argmax(predictions, axis=-1), dtype=tf.int32)
            output = tf.concat([output, predicted_id], axis=-1)
            result[:, i] = predicted_id[:, 0]
        return result

    def __str__(self) -> str:
        return 'transformer'


class TransformerClassification(tf.keras.Model):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.encoder = Encoder(config)
        self.global_conv = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.fc = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(
            config.num_class, activation='softmax')

    def call(self, x):
        inp, training, enc_padding_mask = x
        enc_output = self.encoder(inp, training, enc_padding_mask)
        x = self.global_conv(enc_output)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.out(x)
        return x

    def compile(self, loss, optimizer):
        super().compile()
        self.loss = loss
        self.optimizer = optimizer
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name="acc", dtype=tf.float32
        )
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_acc", dtype=tf.float32
        )

    @property
    def metrics(self):
        return [self.acc, self.val_acc]

    @tf.function
    def train_step(self, data):
        x, y = data
        mask = create_padding_mask(x)
        with tf.GradientTape() as tape:
            predict = self((x, True, mask))
            loss = self.loss(y, predict)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        self.acc.update_state(y, predict)
        return {'loss': loss, 'accuracy': self.acc.result()}

    @tf.function
    def test_step(self, data):
        x, y = data
        predict = self.predict(x)
        loss = self.loss(y, predict)
        self.val_acc.update_state(y, predict)
        return {'val_loss': loss, 'val_acc': self.val_acc.result()}

    def predict(self, x):
        mask = create_padding_mask(x)
        predict = self((x, False, mask))
        return predict

    def __str__(self) -> str:
        return 'transformer_cls'
