from tokenize import Tokenize

tokenize = Tokenize('./vocab.txt', './bpe.codes')

# print(tokenize.encode('sinh_viên'))
# print(tokenize.decode([1, 288, 2]))
print(tokenize(['sinh_viên công_nghệ', 'hello'], maxlen = 10))