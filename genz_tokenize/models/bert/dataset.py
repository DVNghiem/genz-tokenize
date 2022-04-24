import tensorflow as tf
from typing import Union
import numpy as np


class DataCollection:
    def __init__(
        self,
        input_ids: Union[tf.Tensor, np.ndarray] = None,
        attention_mask: Union[tf.Tensor, np.ndarray] = None,
        token_type_ids: Union[tf.Tensor, np.ndarray] = None,
        dec_input_ids: Union[tf.Tensor, np.ndarray] = None,
        dec_attention_mask: Union[tf.Tensor, np.ndarray] = None,
        dec_token_type_ids: Union[tf.Tensor, np.ndarray] = None,
        y=None
    ) -> None:
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.dec_input_ids = dec_input_ids
        self.dec_attention_mask = dec_attention_mask
        self.dec_token_type_ids = dec_token_type_ids
        self.y = y
        if y is None:
            raise Exception('y (label) is required')

    def to_tf_dataset(
        self, batch_size: int = 32
    ) -> tf.data.Dataset:
        """
        Covert data to Dataset to feed to Trainer
        """
        lables = []
        values = []
        for k, v in self.__dict__.items():
            if v is not None:
                lables.append(k)
                values.append(v)
        values = tuple(values)

        def to_dict(*args):
            out = {}
            for k, v in zip(lables, args):
                out[k] = v
            y = out['y']
            del out['y']
            return out, y

        dataset = tf.data.Dataset.from_tensor_slices(values)
        dataset = dataset.shuffle(len(self.input_ids))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.map(to_dict)
        return dataset
