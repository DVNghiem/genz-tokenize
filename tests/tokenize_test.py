from genz_tokenize import Tokenize, TokenizeForBert, get_embedding_matrix
TokenizeForBert()
tokenize = Tokenize()

print(tokenize(['sinh_viên #'], maxlen=10))
print(tokenize.decode([1, 288, 2]))
print(get_embedding_matrix().shape)
