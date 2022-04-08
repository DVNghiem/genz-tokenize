import tensorflow as tf
import os
import pickle


class PretrainModel(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()

    def save_checkpoint(self, checkpoint_dir, epoch):
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        weight_path = os.path.join(checkpoint_dir, 'weights.h5')
        self.save_weight(weight_path)
        extract_data = {'epoch': epoch,
                        'optimizer_state': self.optimizer.get_weights()}
        extract_data_path = os.path.join(checkpoint_dir, 'extract_data.pkl')
        with open(extract_data_path, 'wb') as f:
            pickle.dump(extract_data, f)

    def load_checkpoint(self, checkpoint_dir):
        if getattr(self, 'optimizer', None) is None:
            raise RuntimeError(
                'Checkpoint load fail because model not compile'
            )
        weight_dir = os.path.join(checkpoint_dir, 'weights.h5')
        extract_data = os.path.join(checkpoint_dir, 'extract_data.pkl')

        self.load_weights(weight_dir)
        with open(extract_data, 'rb') as f:
            extract_data = pickle.load(f)

        self.optimizer.set_weights(extract_data['optimizer_state'])

    @classmethod
    def fromPretrain(cls, config, checkpoint_dir):
        model = cls(config)
        model.load_checkpoint(checkpoint_dir)
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

    def train_step(self, data):
        x = data[:-1]
        y = data[-1]
        with tf.GradientTape() as tape:
            predicts = self(x, training=True)
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
