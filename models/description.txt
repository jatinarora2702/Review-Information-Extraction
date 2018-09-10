MODELS
------

1. #LSTM Layer = 1, unidirectional, 300-emb-text8, UNK (0), PAD(0)
2. #LSTM Layer = 2, unidirectional, 300-emb-text8, UNK (0), PAD(0)
3. #LSTM Layer = 1, bidirectional, 300-emb-text8, UNK (0), PAD(0)		[found to work the best]
4. #LSTM Layer = 1, unidirectional, 300-emb-text8, UNK (0), PAD Bit
5. #LSTM Layer = 1, unidirectional, 100-emb-text8, UNK (0), PAD(0)
6. #LSTM Layer = 1, unidirectional, 100-emb-elecrev, UNK (0), PAD(0)
7. #LSTM Layer = 1, bidirectional, 300-emb-text8, UNK (0), PAD Bit0

UNK (0) - means that for unknown tokens (not found in the vocabulary), the embeddings are taken to be the zero vector.
PAD (0) - means, similarly for the padding tokens.
PAD Bit - means, specially, one dimension in the token embedding specially signifies whether the token is a padding or a part of the sentence.

text8 - refers to the word embeddings prepared from the standard Text-8 English corpus
elecrev - refers to word embeddings prepared from electronic-product reviews corpus, taken from Amazon.com & Amazon.in

For more details, you may refer the research paper.W