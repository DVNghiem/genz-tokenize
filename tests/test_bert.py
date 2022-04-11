from genz_tokenize.models.bert.distilbert import DistilBertConfig, DistilBertSeqClassification
from genz_tokenize.models.bert.training import TrainArg, Trainner
import tensorflow as tf

x = tf.zeros(shape=(10, 10), dtype=tf.int32)
mask = tf.zeros(shape=(10, 10), dtype=tf.int32)
y = tf.zeros(shape=(10, 5), dtype=tf.int32)

dataset_train = tf.data.Dataset.from_tensor_slices((x, mask, y))
dataset_train = dataset_train.batch(2)
dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

dataset_eval = tf.data.Dataset.from_tensor_slices((x, mask, y))
dataset_eval = dataset_eval.batch(2)
dataset_eval = dataset_eval.prefetch(tf.data.experimental.AUTOTUNE)

config = DistilBertConfig.fromJson('tests')
config.num_attention_heads = 2
config.num_hidden_layers = 2
config.vocab_size = 5
config.dim = 10
config.hidden_dim = 5
config.num_labels = 5

model = DistilBertSeqClassification(config)
arg = TrainArg(epochs=2, batch_size=2, learning_rate=1e-2)
trainer = Trainner(model, arg, dataset_train)
trainer.train()

print('oke')
