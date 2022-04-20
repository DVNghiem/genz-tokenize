from genz_tokenize.models.bert.model_utils import PretrainModel, QAMetricAccuracy
from genz_tokenize.models.bert.roberta import RobertaConfig, Embedding, Encoder, Decoder
import tensorflow as tf


class RoBertaClassification(PretrainModel):
    def __init__(self, config: RobertaConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.type = 'cls'
        self.embedding = Embedding(config)
        self.encoder = Encoder(config)

        self.pool = tf.keras.layers.Dense(
            config.hidden_size, activation='tanh')
        self.logits = tf.keras.layers.Dense(config.num_labels)

    def compile(self, loss, optimizer, metrics=None, **kwargs):
        super().compile(loss, optimizer, metrics, **kwargs)
        self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        self.val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    def call(
            self,
            input_ids: tf.Tensor = None,
            attention_mask: tf.Tensor = None,
            token_type_ids: tf.Tensor = None,
            dec_input_ids: tf.Tensor = None,
            dec_attention_mask: tf.Tensor = None,
            dec_token_type_ids: tf.Tensor = None,
            training: bool = False):
        embedding = self.embedding(
            input_ids=input_ids, token_type_ids=token_type_ids, training=training)
        hidden_state = self.encoder(
            input_ids=embedding, attention_mask=attention_mask, training=training)

        pool = self.pool(hidden_state[:, 0])
        return self.logits(pool)


class RoBertaQAPair(PretrainModel):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()

        self.type = 'QA'

        self.embedding = Embedding(config)
        self.encoder = Encoder(config)
        self.split = tf.keras.layers.Dense(2)
        self.start = tf.keras.layers.Dense(config.max_position_embedding)
        self.end = tf.keras.layers.Dense(config.max_position_embedding)

    def compile(self, loss, optimizer, metrics=None, **kwargs):
        self.train_acc_metric = QAMetricAccuracy(logits=True)
        self.val_acc_metric = QAMetricAccuracy(logits=True)
        return super().compile(loss, optimizer, metrics, **kwargs)

    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        dec_input_ids: tf.Tensor = None,
        dec_attention_mask: tf.Tensor = None,
        dec_token_type_ids: tf.Tensor = None,
        training: bool = False
    ):
        embedding = self.embedding(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            training=training
        )

        hidden_state = self.encoder(
            input_ids=embedding,
            attention_mask=attention_mask,
            training=training
        )

        split = self.split(hidden_state)
        start_logits, end_logits = tf.split(
            value=split, num_or_size_splits=2, axis=-1)
        start_logits = self.start(
            tf.squeeze(input=start_logits, axis=-1))
        end_logits = self.end(tf.squeeze(input=end_logits, axis=-1))
        return start_logits, end_logits


class RoBertaQAEncoderDecoder(RoBertaQAPair):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__(config)
        self.type = 'QA_ed'
        self.dec_embedding = Embedding(config)
        self.decoder = Decoder(config)

    def call(
            self,
            input_ids: tf.Tensor = None,
            attention_mask: tf.Tensor = None,
            token_type_ids: tf.Tensor = None,
            dec_input_ids: tf.Tensor = None,
            dec_attention_mask: tf.Tensor = None,
            dec_token_type_ids: tf.Tensor = None,
            training: bool = False
    ):

        enc_embedding = self.embedding(
            input_ids=input_ids, token_type_ids=token_type_ids)
        hidden_state = self.encoder(
            input_ids=enc_embedding,
            attention_mask=attention_mask,
            training=training
        )
        dec_embedding = self.dec_embedding(
            input_ids=dec_input_ids,
            token_type_ids=dec_token_type_ids,
            training=training
        )
        hidden_state = self.decoder(
            input_ids=dec_embedding,
            attention_mask=dec_attention_mask,
            encoder_output=hidden_state,
            encoder_attention_mask=attention_mask,
            training=training
        )

        split = self.split(hidden_state)
        start_logits, end_logits = tf.split(
            value=split, num_or_size_splits=2, axis=-1)
        start_logits = self.start(
            tf.squeeze(input=start_logits, axis=-1))
        end_logits = self.end(tf.squeeze(input=end_logits, axis=-1))
        return start_logits, end_logits

    def compile(self, loss, optimizer, metrics=None, **kwargs):
        return super().compile(loss, optimizer, metrics, **kwargs)
