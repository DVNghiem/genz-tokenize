import tensorflow as tf
from .layers import Embeddings, DistilEncoder, DistilDecoder
from .config import DistilBertConfig
from ..model_utils import PretrainModel, QAMetricAccuracy


class DistilBertSeqClassification(PretrainModel):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.type = 'cls'

        self.embedding = Embeddings(config, name='embedding')
        self.distil = DistilEncoder(config, name='distil_encoder')

        self.pool_output = tf.keras.layers.Dense(
            config.hidden_dim, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.logits = tf.keras.layers.Dense(
            config.num_labels, activation='softmax')

    def compile(self, loss, optimizer, metrics=None, **kwargs):
        super().compile(loss, optimizer, metrics, **kwargs)
        self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        self.val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        dec_input_ids=None,
        dec_attention_mask=None,
        dec_token_type_ids=None,
        training=False
    ):
        embedding = self.embedding(input_ids=input_ids)
        bert_output = self.distil(
            x=embedding,
            attention_mask=attention_mask,
            training=training
        )
        pool = self.pool_output(bert_output[:, 0])
        out = self.dropout(pool)
        logits = self.logits(out)
        return logits


class DistilBertQAPair(PretrainModel):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.type = 'QA'
        self.embedding = Embeddings(config)
        self.encoder = DistilEncoder(config, name='Encoder')
        self.split = tf.keras.layers.Dense(2)
        self.start = tf.keras.layers.Dense(config.max_position_embeddings)
        self.end = tf.keras.layers.Dense(config.max_position_embeddings)

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        dec_input_ids=None,
        dec_attention_mask=None,
        dec_token_type_ids=None,
        training=False
    ):
        hidden_state = self.embedding(input_ids=input_ids)
        bert_output = self.encoder(
            x=hidden_state,
            attention_mask=attention_mask,
            training=training
        )
        logits = self.split(bert_output)
        start_logits, end_logits = tf.split(
            value=logits, num_or_size_splits=2, axis=-1)
        start_logits = self.start(
            tf.squeeze(input=start_logits, axis=-1))
        end_logits = self.end(tf.squeeze(input=end_logits, axis=-1))
        return start_logits, end_logits

    def compile(self, loss, optimizer, metrics=None, **kwargs):
        super().compile(loss, optimizer, metrics, **kwargs)
        self.train_acc_metric = QAMetricAccuracy(logits=True)
        self.val_acc_metric = QAMetricAccuracy(logits=True)


class DistilBertQAEncoderDecoder(DistilBertQAPair):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.type = 'QA_ed'
        self.dec_embedding = Embeddings(config)
        self.decoder = DistilDecoder(config)

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        dec_input_ids=None,
        dec_attention_mask=None,
        dec_token_type_ids=None,
        training=False
    ):
        embedding = self.embedding(input_ids)
        enc_output = self.encoder(
            x=embedding, attention_mask=attention_mask, training=training)
        dec_embedding = self.dec_embedding(dec_input_ids)
        dec_output = self.decoder(x=dec_embedding,
                                  attention_mask=dec_attention_mask,
                                  encoder_hidden_state=enc_output,
                                  training=training)
        logits = self.split(dec_output)
        start_logits, end_logits = tf.split(
            value=logits, num_or_size_splits=2, axis=-1)
        start_logits = self.start(
            tf.squeeze(input=start_logits, axis=-1))
        end_logits = self.end(tf.squeeze(input=end_logits, axis=-1))
        return start_logits, end_logits

    def compile(self, loss, optimizer, metrics=None, **kwargs):
        return super().compile(loss, optimizer, metrics, **kwargs)
