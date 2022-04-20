import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
import os
import json


class Config:
    def saveJson(self, path):
        if not os.path.exists(path=path):
            os.mkdir(path=path)
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.__dict__, f)

    @classmethod
    def fromJson(cls, path):
        '''
        path: Folder contain config.json\n
        path:\n
            |__....\n
            |__ config.json\n
            |__....\n           
        '''
        if not os.path.exists(path):
            raise Exception(f'{os.path.join(path, "config.json")} not found')
        with open(os.path.join(path, 'config.json'), 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            setattr(cls, k, v)
        return cls


def save_checkpoint(model, optimizer: tf.keras.optimizers.Optimizer = None, checkpoint_dir: str = None):
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=5)
    ckpt_manager.save()


def load_checkpoint(model, optimizer: tf.keras.optimizers.Optimizer = None, checkpoint_dir: str = None):
    if optimizer:
        checkpoint = tf.train.Checkpoint(
            model=model,  optimizer=optimizer)
    else:
        checkpoint = tf.train.Checkpoint(
            model=model,  optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print('\nLatest checkpoint restored!!!\n')


class PretrainModel(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def fromPretrain(cls, config: Config, checkpoint_dir):
        model = cls(config)
        load_checkpoint(model, optimizer=None, checkpoint_dir=checkpoint_dir)
        return model

    def compile(self, loss, optimizer, metrics=None, **kwargs):
        super().compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            **kwargs
        )
        self.train_loss_metric = tf.keras.metrics.Mean()
        self.val_loss_metric = tf.keras.metrics.Mean()

    @property
    def metrics(self):
        return [
            self.train_loss_metric,
            self.train_acc_metric,
            self.val_loss_metric,
            self.val_acc_metric
        ]

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
        raise NotImplementedError

    def train_step(self, data):
        inputs, y, _ = data_adapter.unpack_x_y_sample_weight(data)
        inputs['training'] = True
        with tf.GradientTape() as tape:
            predicts = self(**inputs)
            loss = self.loss(y, predicts)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.train_loss_metric(loss)
        self.train_acc_metric.update_state(y, predicts)
        return {
            'loss': self.train_loss_metric.result(),
            'accuracy': self.train_acc_metric.result()
        }

    def test_step(self, data):
        inputs, y, _ = data_adapter.unpack_x_y_sample_weight(data)
        inputs['training'] = False
        predicts = self(**inputs)
        loss = self.loss(y, predicts)
        self.val_loss_metric(loss)
        self.val_acc_metric.update_state(y, predicts)
        return {
            'loss': self.val_loss_metric.result(),
            'accuracy': self.val_acc_metric()
        }

    def predict(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        dec_input_ids=None,
        dec_attention_mask=None,
        dec_token_type_ids=None
    ):
        pred = self(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    dec_input_ids=dec_input_ids,
                    dec_attention_mask=dec_attention_mask,
                    dec_token_type_ids=dec_token_type_ids)
        return pred


class LossQA(tf.keras.losses.Loss):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, y, predict):
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none',
        )
        loss_start = loss_obj(y[:, 0:1], predict[0])
        loss_end = loss_obj(y[:, 1:], predict[1])
        return (loss_start+loss_end)/2


class LossSeq2Seq(tf.keras.losses.Loss):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, y, predict):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        mask = tf.math.logical_not(tf.math.equal(y, 0))
        loss_ = loss_object(y, predict)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


class LossClassification(tf.keras.losses.Loss):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, y, predict):
        loss_obj = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE,
        )
        loss = loss_obj(y, predict)
        return loss


class QAMetricAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='qa_metric', logits=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.acc = self.add_weight(name='acc', initializer='zeros')
        self.logits = logits

    def update_state(self, y_true, y_pred):
        start = y_pred[0]
        end = y_pred[1]
        if self.logits:
            start = tf.nn.softmax(start, axis=1)
            end = tf.nn.softmax(end, axis=1)
        start = tf.argmax(start, axis=1)
        end = tf.argmax(end, axis=1)
        y_true = tf.cast(y_true, dtype=start.dtype)
        acc = (tf.cast(tf.equal(
            y_true[:, 0], start), dtype=tf.float32)
            +
            tf.cast(
                tf.equal(y_true[:, 1], end), dtype=tf.float32))/2
        append_prev = tf.convert_to_tensor([tf.reduce_mean(acc), self.acc])
        self.acc.assign(tf.reduce_mean(append_prev))

    def result(self):
        return self.acc

    def reset_state(self):
        self.acc.assign(0.)
