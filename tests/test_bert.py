from genz_tokenize.models.bert import DataCollection
from genz_tokenize.models.bert.distilbert import DistilBertConfig, DistilBertSeqClassification, DistilBertQAPair, DistilBertQAEncoderDecoder
from genz_tokenize.models.bert.training import TrainArg, Trainner
from genz_tokenize.models.bert.roberta import RoBertaClassification, RobertaConfig, RoBertaQAPair, RoBertaQAEncoderDecoder
import tensorflow as tf

x = tf.zeros(shape=(10, 10), dtype=tf.int32)
mask = tf.zeros(shape=(10, 10), dtype=tf.int32)
y = tf.zeros(shape=(10, 2), dtype=tf.int32)

config = RobertaConfig()
config.num_attention_heads = 2
config.num_hidden_layers = 2
config.vocab_size = 5
config.dim = 10
config.hidden_dim = 5
config.num_labels = 2

dataset = DataCollection(
    input_ids=x,
    attention_mask=mask,
    token_type_ids=mask,
    dec_input_ids=x,
    dec_attention_mask=mask,
    dec_token_type_ids=mask,
    y=y
)

tf_dataset = dataset.to_tf_dataset(batch_size=2)

model = RoBertaQAEncoderDecoder(config)
arg = TrainArg(epochs=2, batch_size=2, learning_rate=1e-2)
trainer = Trainner(model, arg, tf_dataset)
trainer.train()

print(model.predict(input_ids=x, attention_mask=mask,
      dec_input_ids=x, dec_attention_mask=mask))
