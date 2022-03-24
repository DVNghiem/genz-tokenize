import re
from typing import Any, Dict, List, Optional
import os
from transformers import PreTrainedTokenizer
import numpy as np
import pickle


class Tokenize(object):
    def __init__(self,
                 pad_token='<pad>',
                 bos_token='<s>',
                 eos_token='</s>',
                 mask_token='<mask>',
                 unk_token='<unk>') -> None:
        super().__init__()

        this_dir, this_filename = os.path.split(__file__)
        try:
            self.vocab_file
        except:
            self.vocab_file = os.path.join(this_dir, 'data', 'vocab.txt')
        try:
            self.bpe_file
        except:
            self.bpe_file = os.path.join(this_dir, 'data', 'bpe.codes')

        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.unk_token = unk_token

        self.encoder = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.mask_token: 3,
            self.unk_token: 4
        }

        self.add_vocab_file(self.vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.add_bpe_file(self.bpe_file)

    def add_vocab_file(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            idx = line.rfind(' ')
            word = line[:idx]
            self.encoder[word] = len(self.encoder)

    def add_bpe_file(self, bpe_file):
        with open(bpe_file, 'r', encoding='utf-8') as f:
            merges = f.read().split('\n')[:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        word = tuple(token)
        word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
        pairs = get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)  # i: start
                except ValueError:  # khong tim thay
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j
                # nêu index tìm thấy là firts và từ kế tiếp là second thì gộp lại
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = "@@ ".join(word)
        word = word[:-4]
        return word

    def __tokenize(self, text, return_offset):
        split_tokens = []
        offset = [(0, 0)]
        words = re.findall(r"\S+\n?", text)  # tách từ ra

        for token in words:
            # kết nối các ký tự với nhau
            tokens = [t for t in self.bpe(token).split(" ")]
            if return_offset:
                offset.append(
                    (len(split_tokens)+1, len(split_tokens)+len(tokens)))
            split_tokens.extend(tokens)
        if return_offset:
            offset.append((len(split_tokens)+1, len(split_tokens)+1))
            return split_tokens, offset
        return split_tokens  # tách từ ra

    def __convert_token_to_ids(self, word):
        return self.encoder.get(word, self.encoder.get(self.unk_token))

    def __convert_token_to_string(self, token):
        return self.decoder.get(token, self.unk_token)

    def encode(self, sentence, return_offset) -> List[int]:
        tokens = self.__tokenize(sentence, return_offset)
        if return_offset:
            tokens, offset = tokens
        result = []
        for i in tokens:
            result.append(self.__convert_token_to_ids(i))
        if return_offset:
            return [self.encoder[self.bos_token]]+result+[self.encoder[self.eos_token]], offset
        return [self.encoder[self.bos_token]]+result+[self.encoder[self.eos_token]]

    def decode(self, token):
        sentence = [self.__convert_token_to_string(i) for i in token]
        return ' '.join(sentence).replace('@@ ', '')

    def __padding(self, token, max_len, truncation):
        if len(token) < max_len:
            return token+[self.encoder[self.pad_token]]*(max_len-len(token))
        if truncation:
            return token[:max_len-1]+[self.encoder[self.eos_token]]
        return token

    def get_atttention_mask(self, token):
        mask = []
        for i in token:
            mask.append(1 if i != self.encoder[self.pad_token] else 0)
        return mask

    def get_token_type(self, token):
        return [0]*len(token)

    def get_sequence_id(self, token):
        eos = self.encoder[self.eos_token]
        bos = self.encoder[self.bos_token]

        seq_id = []
        for i in token:
            if i == eos:
                seq_id.append(None)
                break
            else:
                seq_id.append(None if i == bos else 0)

        for i in range(len(seq_id), len(token), 1):
            if token[i] == eos:
                seq_id.append(None)
                if seq_id[i-1] == 1:
                    break
            else:
                seq_id.append(1)
        return seq_id

    def __call__(self,
                 text: str,
                 pair_text: str = None,
                 max_len: int = None,
                 padding: bool = None,
                 truncation: bool = False,
                 return_offset: bool = False) -> Dict:
        result = {}
        text = self.encode(text, return_offset)
        tokens = []
        if pair_text is not None:
            if return_offset:
                offset = []
                pair_text = self.encode(pair_text, return_offset=True)
                tokens.extend(text[0])
                tokens.extend([self.encoder[self.eos_token]
                               ]+pair_text[0][1:])
                offset.extend(text[1])
                offset.extend([(i+len(text[1]), j+len(text[1]))
                              for i, j in pair_text[1]])
                result['offset'] = offset

            else:
                tokens.extend(text)
                tokens.append(self.encoder[self.eos_token])
                tokens.extend(self.encode(pair_text, return_offset=False)[1:])
        else:
            if return_offset:
                tokens = text[0]
                offset = text[1]
                result['offset'] = offset
            else:
                tokens = text
        if max_len is not None and padding:
            tokens = self.__padding(
                tokens, max_len=max_len, truncation=truncation)
        result['input_ids'] = tokens
        result['attention_mask'] = self.get_atttention_mask(tokens)
        if pair_text is not None:
            result['sequence_id'] = self.get_sequence_id(tokens)
        return result

    @ classmethod
    def fromFile(cls, vocab_file, bpe_file):
        tokenize = cls()
        tokenize.vocab_file = vocab_file
        tokenize.bpe_file = bpe_file
        tokenize.__init__()
        return tokenize


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    pairs = set(pairs)
    return pairs


class TokenizeForBert(PreTrainedTokenizer):

    def __init__(
        self,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        this_dir, this_filename = os.path.split(__file__)
        try:
            self.vocab_file
        except:
            self.vocab_file = os.path.join(this_dir, 'data', 'vocab.txt')
        try:
            self.bpe_file
        except:
            self.bpe_file = os.path.join(this_dir, 'data', 'bpe.codes')

        self.encoder = {}
        self.encoder[self.pad_token] = 0
        self.encoder[self.bos_token] = 1
        self.encoder[self.eos_token] = 2
        self.encoder[self.mask_token] = 3
        self.encoder[self.unk_token] = 4

        self.add_from_file(self.vocab_file)

        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(self.bpe_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A PhoBERT sequence has the following format:
        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. PhoBERT does not
        make use of token type ids, therefore a list of zeros is returned.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @ property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = "@@ ".join(word)
        word = word[:-4]
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        split_tokens = []

        words = re.findall(r"\S+\n?", text)

        for token in words:
            split_tokens.extend([t for t in self.bpe(token).split(" ")])
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string

    def add_from_file(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            idx = line.rfind(' ')
            word = line[:idx]
            self.encoder[word] = len(self.encoder)

    @ classmethod
    def fromFile(cls, vocab_file, bpe_file):
        tokenize = cls()
        tokenize.vocab_file = vocab_file
        tokenize.bpe_file = bpe_file
        tokenize.__init__()
        return tokenize


def get_embedding_matrix():
    this_dir, _ = os.path.split(__file__)
    with open(os.path.join(this_dir, 'data', 'emb_1.pkl'), 'rb') as f:
        emb_1 = pickle.load(f)
    with open(os.path.join(this_dir, 'data', 'emb_2.pkl'), 'rb') as f:
        emb_2 = pickle.load(f)
    with open(os.path.join(this_dir, 'data', 'emb_3.pkl'), 'rb') as f:
        emb_3 = pickle.load(f)
    embedding_matrix = np.vstack([emb_1, emb_2, emb_3])
    return embedding_matrix
