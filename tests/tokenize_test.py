from genz_tokenize import Tokenize, TokenizeForBert, get_embedding_matrix
TokenizeForBert()
tokenize = Tokenize()

print(tokenize.encode('ông'))
print(tokenize.decode([1, 17778, 2]))
print(len(tokenize.encoder))
print(len(tokenize.decoder))
get_embedding_matrix()
