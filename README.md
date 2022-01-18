# Genz Tokenize

[Github](https://github.com/nghiemIUH/genz-tokenize)

## Cài đặt:

    pip install genz-tokenize

## Sử dụng cho tokenize thông thường

```python
    >>> from genz_tokenize import Tokenize
    # sử dụng vocab sẵn có của thư viện
    >>> tokenize = Tokenize()
    >>>  print(tokenize(['sinh_viên công_nghệ', 'hello'], maxlen = 5))
    # [[1, 288, 433, 2, 0], [1, 20226, 2, 0, 0]]
    >>> print(tokenize.decode([1, 288, 2]))
    # <s> sinh_viên </s>
    # Sử dụng vocab tự tạo
    >>> tokenize = Tokenize.fromFile('vocab.txt','bpe.codes')
```

## Sử dụng tokenize cho model bert của thư viện transformers

```python
    >>> from genz_tokenize import TokenizeForBert
    # sử dụng vocab sẵn có của thư viện
    >>> tokenize = TokenizeForBert()
    >>> print(tokenize(['sinh_viên công_nghệ', 'hello'], max_length=5, padding='max_length',truncation=True))
    # {'input_ids': [[1, 287, 432, 2, 0], [1, 20225, 2, 0, 0]], 'token_type_ids': [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]}
    # Sử dụng vocab tự tạo
    >>> tokenize = TokenizeForBert.fromFile('vocab.txt','bpe.codes')
```

### Có thể tạo vocab cho riêng mình bằng thư viện [subword-nmt (learn-joint-bpe-and-vocab)](https://github.com/rsennrich/subword-nmt)
