import torch
import numpy as np

batch_size = 3
seq_length = 4

embedding_dim = 6  # 小于vocab_size
vocab_size = 20

input_data = np.random.uniform(0, 19, size=(batch_size, seq_length))  # shape(3,4)
input_data = torch.from_numpy(input_data).long()
embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
lstm_input = embedding_layer(input_data)

print(lstm_input.shape)  # torch.Size([3, 4, 6]) （batch_size，max_length，embedding_dim）
