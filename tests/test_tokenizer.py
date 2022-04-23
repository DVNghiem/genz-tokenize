from genz_tokenize import Tokenize
from genz_tokenize import Tokenize, TokenizeForBert
from genz_tokenize.preprocess import remove_emoji
from genz_tokenize.models.base_model.utils import Config
from genz_tokenize.models.base_model.models import Seq2Seq, Transformer, TransformerClassification
from genz_tokenize.models.base_model.training import TrainArgument, Trainer

import tensorflow as tf

tokenize = TokenizeForBert()

config = Config()
config.vocab_size = 10
config.target_vocab_size = 12
config.units = 16
config.maxlen = 20
config.num_heads = 2
config.num_hidden_layers = 2
config.seq2seq_attention = 'bahdanau'
config.num_lang = 2


model = Transformer(config)
model.loadCheckpoint('checkpoint')

x = tf.zeros(shape=(10, 5))
y = tf.zeros(shape=(10, config.maxlen))

BUFFER_SIZE = len(x)
dataset_train = tf.data.Dataset.from_tensor_slices((x, y))
dataset_train = dataset_train.cache()
dataset_train = dataset_train.batch(2)
dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

BUFFER_SIZE = len(x)
dataset_eval = tf.data.Dataset.from_tensor_slices((x, y))
dataset_eval = dataset_eval.cache()
dataset_eval = dataset_eval.batch(2)
dataset_eval = dataset_eval.prefetch(tf.data.experimental.AUTOTUNE)

args = TrainArgument(batch_size=2, epochs=2, model_dir='checkpoint')
trainer = Trainer(model=model, args=args,
                  data_train=dataset_train, data_eval=dataset_eval)
trainer.train()
trainer.save()
print(model.predict(x))
