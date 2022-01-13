import re
from typing import Any, Dict, List

class Tokenize(object):
    def __init__(self, 
                vocab_file,
                bpe_file,
                pad_token = '<pad>',
                bos_token = '<s>',
                eos_token = '</s>',
                mask_token = '<mask>',
                unk_token = '<unk>') -> None:
        super().__init__()

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

        self.add_vocab_file(vocab_file)
        self.decoder = {v:k for k, v in self.encoder.items()}

        self.add_bpe_file(bpe_file)

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
        pairs = self.get_pairs(word)
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
                pairs = self.get_pairs(word)
        word = "@@ ".join(word)
        word = word[:-4]
        return word

    def __tokenize(self, text):
        split_tokens = []

        words = re.findall(r"\S+\n?", text)  # tách từ ra

        for token in words:
            split_tokens.extend([t for t in self.bpe(token).split(" ")]) # kết nối các ký tự với nhau
        return split_tokens  # tách từ ra
    
    def __convert_token_to_ids(self, word):
        return self.encoder.get(word, self.encoder.get(self.unk_token))

    def __convert_token_to_string(self, token):
        return self.decoder.get(token, self.unk_token)

    def encode(self, sentence) -> List[int]:
        tokens = self.__tokenize(sentence)
        tokens = [self.__convert_token_to_ids(i) for i in tokens]
        return [self.encoder[self.bos_token]]+tokens+[self.encoder[self.eos_token]]

    def decode(self, token):
        sentence = [self.__convert_token_to_string(i) for i in token]
        return ' '.join(sentence).replace('@@ ', '')

    def __padding(self, token, maxlen, truncate):
        if len(token) < maxlen:
            return token+[self.encoder[self.pad_token]]*(maxlen-len(token))
        if truncate:
            return token[:maxlen-1]+[self.encoder[self.eos_token]]
        return token

    def __call__(self, *args: Any, **kwds: Any) -> Dict:
        '''
            sentences: string,
            maxlen: int,
            truncate: boolean, default True
            padding: boolean, default True
        '''
        sentences = args[0]
        maxlen = kwds['maxlen']
        truncate = kwds.get('truncate', True)
        padding = kwds.get('padding', True)
        input_ids = []
        mask = []
        for i in sentences:
            token = self.encode(i)
            if padding:
                token = self.__padding(token, maxlen, truncate)
            mask.append([1 if i!=0 else 0 for i in token])
            input_ids.append(token)
        return {'input_ids': input_ids, 'mask': mask}

    @staticmethod
    def get_pairs(word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char

        pairs = set(pairs)
        return pairs