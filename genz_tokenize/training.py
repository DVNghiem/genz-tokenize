from wsgiref import validate
import tensorflow as tf
import os

from .utils import loss_seq2seq, loss_transformer, CustomSchedule, CallbackSave


class TrainArgument:
    def __init__(self,
                 model_dir: str = 'model',
                 epochs: int = 10,
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 save_per_epochs: int = 1,
                 ) -> None:
        self.model_dir = model_dir
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_per_epochs = save_per_epochs


class Trainer:
    '''
    This Trainer class help training easier

    ```python
    >>> from genz_tokenize.utils import Config
    >>> from genz_tokenize.models import Seq2Seq
    >>> from genz_tokenize.training import TrainArgument, Trainer
    # create config hyper parameter
    >>> config = Config()
    >>> config.vocab_size = 100
    >>> config.target_vocab_size = 120
    >>> config.units = 16
    >>> config.maxlen = 20
    # initial model
    >>> model = Seq2Seq(config)
    >>> x = tf.zeros(shape=(10, config.maxlen))
    >>> y = tf.zeros(shape=(10, config.maxlen))
    # create dataset
    >>> BUFFER_SIZE = len(x)
    >>> dataset_train = tf.data.Dataset.from_tensor_slices((x, y))
    >>> dataset_train = dataset_train.shuffle(BUFFER_SIZE)
    >>> dataset_train = dataset_train.batch(2)
    >>> dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

    >>> args = TrainArgument(batch_size=2, epochs=2)
    >>> trainer = Trainer(model=model, args=args, data_train=dataset_train)
    >>> trainer.train()
    ```

    '''

    def __init__(self,
                 model: tf.keras.Model,
                 args: TrainArgument,
                 data_train: tf.data.Dataset = None,
                 data_eval: tf.data.Dataset = None,
                 ) -> None:
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate)

        self.model = model
        self.data_train = data_train
        self.data_eval = data_eval
        self.args = args
        self.model.batch_size = args.batch_size
        model_type = self.model.__str__()
        if model_type == 'seq2seq':
            self.loss_fn = loss_seq2seq
        elif model_type == 'transformer':
            self.loss_fn = loss_transformer
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=CustomSchedule(model.config.hidden_size))
        elif model_type == 'transformer_cls':
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            raise Exception(
                'Model type must be seq2seq, transformer or transformer_cls')

    def train(self, model_dir: str = '') -> None:
        '''
            compile and train model
        '''
        self.model.compile(loss=self.loss_fn, optimizer=self.optimizer)
        if model_dir != '':
            self.model.load_weights(model_dir)

        self.model.fit(self.data_train,
                       epochs=self.args.epochs,
                       validation_data=self.data_eval,
                       callbacks=[
                           CallbackSave(self.args.model_dir,
                                        self.args.save_per_epochs)
                       ])

    def save(self, model_dir: str = '') -> None:
        if model_dir == '':
            raise Exception('model directory not empty')
        self.model.save_weights(os.path.join(model_dir, 'final'))
