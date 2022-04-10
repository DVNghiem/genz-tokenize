import tensorflow as tf
from .layers import Embeddings, DistilDecoderLayer, DistilEncoderLayer, DistilEncoder, DistilDecoder
from .config import DistilBertConfig
from ..model_utils import PretrainModel


class DistilBertSeqClassification(PretrainModel):
    def __init__(self, config: DistilBertConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embedding = Embeddings(config, name='embedding')
        self.distil = DistilEncoder(config, name='distil_encoder')

        self.pool_output = tf.keras.layers.Dense(
            config.hidden_dim, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.logits = tf.keras.layers.Dense(
            config.num_labels, activation='softmax')

    def compile(self, loss, optimizer, metrics=None, **kwargs):
        super().compile(loss, optimizer, metrics, **kwargs)
        self.train_loss_metric = tf.keras.metrics.Mean()
        self.val_loss_metric = tf.keras.metrics.Mean()
        self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        self.val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    def call(self, input_ids=None, attention_mask=None, token_type_ids=None, training=False):
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

    def __str__(self) -> str:
        return 'cls'
