from genz_tokenize import Tokenize, TokenizeForBert

# tokenize = Tokenize()

tokenize = Tokenize.fromFile('./genz_tokenize/data/vocab.txt',
                             './genz_tokenize/data/bpe.codes')

# print(tokenize.encode('sinh_viên'))
print(tokenize.decode([1, 288, 2]))
# print(tokenize(['sinh_viên công_nghệ', 'hello'],
#       max_length=5, padding='max_length', truncation=True))
# print(tokenize(['sinh_viên công_nghệ', 'hello'],
#       maxlen=5, padding=True, truncation=True))
