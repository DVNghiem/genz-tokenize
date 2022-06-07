import re
from typing import Dict, List
import os


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

    def __tokenize(self, text: str, return_offset: bool):
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
        token[0] = 0
        token[-1] = 1
        index = token.index(None)
        token[index] = 0
        index = token.index(None)
        token[index] = 1
        return token

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
                 padding: bool = True,
                 truncation: bool = True,
                 return_offset: bool = False) -> Dict:
        """
        Convert text to number

        Args:
            text |`String`:
                text need convert

            pair_text |`String`:
                Default None. If it not None, will combine text and pair_text. <s> tetxt </s></> pair_text </s>

            max_len |`int`:
                maximum length of input

            padding |`bool`:
                padding with value 0 so that equal max_len. Default True

            truncation |`bool`:
                truncate with text longer than max_len. Default True

            return_offset |`bool`:
                True if return offset else False. Default False
        Return:
            Dict {
                input_ids: [...],
                attention_mask: [...],
                token_type_ids: [...],
                sequence_id: [...],
                offset: [...]
            }
        """
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
            result['token_type_ids'] = self.get_token_type(
                result['sequence_id'])
            if max_len is not None and padding:
                result['token_type_ids'] = self.__padding(
                    result['token_type_ids'], max_len, truncation)
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
