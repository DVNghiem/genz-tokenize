import tensorflow as tf
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

    @property
    def metrics(self):
        return [
            self.train_loss_metric,
            self.train_acc_metric,
            self.val_loss_metric,
            self.val_acc_metric
        ]

    def call(self, input_ids=None, attention_mask=None, token_type_ids=None, training=False):
        raise NotImplementedError

    def train_step(self, data):
        inputs = {
            'input_ids': data[0],
            'attention_mask': data[1],
            'token_type_ids': None if len(data) == 3 else data[2],
            'training': True
        }

        y = data[-1]
        with tf.GradientTape() as tape:
            predicts = self(**inputs)
            loss = self.loss.compute_loss(y, predicts)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.train_loss_metric(loss)
        self.train_acc_metric.update_state(y, predicts)
        return {
            'loss': self.train_loss_metric.result(),
            'accuracy': self.train_acc_metric.result()
        }

    def test_step(self, data):
        x, y = data
        predicts = self(x, training=False)
        loss = self.loss.compute_loss(y, predicts)
        self.val_loss_metric(loss)
        self.val_acc_metric.update_state(y, predicts)
        return {
            'loss': self.val_loss_metric.result(),
            'accuracy': self.val_acc_metric()
        }

    def predict(self, x):
        raise NotImplementedError


class LossQA:

    def compute_loss(self, y, predict):
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none',
            name='loss QA'
        )
        loss_start = loss_obj(y[:, 0:1], predict[0])
        loss_end = loss_obj(y[:, 1:], predict[1])
        return (loss_start+loss_end)/2


class LossSeq2Seq:

    def compute_loss(self, y, predict):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        mask = tf.math.logical_not(tf.math.equal(y, 0))
        loss_ = loss_object(y, predict)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


class LossClassification:

    def compute_loss(self, y, predict):
        loss_obj = tf.keras.losses.CategoricalCrossentropy()
        loss = loss_obj(y, predict)
        return loss
