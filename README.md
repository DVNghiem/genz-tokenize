# Genz Tokenize

## Installation:

    pip install genz-tokenize

## Using for tokenize basic

```python
    >>> from genz_tokenize import Tokenize
    # using vocab from lib
    >>> tokenize = Tokenize()
    >>> print(tokenize('sinh_viên công_nghệ', 'hello', max_len = 10, padding = True, truncation = True))
    # {'input_ids': [1, 770, 1444, 2, 2, 30469, 2, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 'sequence_id': [None, 0, 0, None, None, 1, None]}

    >>> print(tokenize.decode([1, 770, 2]))
    # <s> sinh_viên </s>

    # from your vocab
    >>> tokenize = Tokenize.fromFile('vocab.txt','bpe.codes')
```

## Using bert tokenize inheritance from PreTrainedTokenizer [transformers](https://github.com/huggingface/transformers)

```python
    >>> from genz_tokenize import TokenizeForBert
    # Using vocab from lib
    >>> tokenize = TokenizeForBert()
    >>> print(tokenize(['sinh_viên công_nghệ', 'hello'], max_length=5, padding='max_length',truncation=True))
    # {'input_ids': [[1, 770, 1444, 2, 0], [1, 30469, 2, 0, 0]], 'token_type_ids': [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]}

    # Using your vocab
    >>> tokenize = TokenizeForBert.fromFile('vocab.txt','bpe.codes')
```

## Embedding matrix from fasttext

```python
    >>> from genz_tokenize import get_embedding_matrix
    >>> embedding_matrix = get_embedding_matrix()
```

## Model

    1. Seq2Seq with Bahdanau Attention
    2. Transformer classification
    3. Transformer

## Trainer

```python
    >>> from genz_tokenize.models.utils import Config
    >>> from genz_tokenize.models import Seq2Seq, Transformer, TransformerClassification
    >>> from genz_tokenize.models.training import TrainArgument, Trainer
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

### [Create your vocab](https://github.com/rsennrich/subword-nmt)
