# %load model_protos.py

import torch
import torch.nn as nn
from model_base import ModelBase

ORDINAL_FEATURE_INDEX = 17


class LSTMOrdinal(ModelBase):
    def __init__(self, n_features, n_seqeuences, hidden_size=32, dropout=0.2, lr=0.01, batch_size=256, zero_heuristic=False) -> None:
        self.n_features = n_features
        self.n_sequences = n_seqeuences
        self.zero_heuristic = zero_heuristic
        self.learning_rate = lr
        self.batch_size = batch_size
        self.model_name = 'ordinal'

        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.output = nn.Linear(
            hidden_size,
            1
        )


        self.save_hyperparameters()

    def forward(self, x):


        x, _ = self.lstm(x)
        x = x[:, -1]
        return self.output(x)


class LSTMEmbedUser(ModelBase):
    def __init__(self, n_features, n_sequences, embedding_matrix, hidden_size=32, dropout=0.2, lr=0.001, batch_size=256, zero_heuristic=False) -> None:
        super().__init__()

        self.lstm_ordinal = nn.LSTM(
            input_size=n_features - 3,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.user_embedding = nn.Embedding(
            num_embeddings=embedding_matrix['user_embed'],
            embedding_dim=embedding_matrix['embed_out'],
            padding_idx=0
        )

        self.learning_rate = lr
        self.batch_size = batch_size
        self.model_name = 'embedded_user'

        trunk_linear = hidden_size + embedding_matrix['embed_out']

        self.out_trunk = nn.Sequential(
            nn.Linear(trunk_linear, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

        self.n_sequences = n_sequences 
        self.zero_heuristic = zero_heuristic

        self.save_hyperparameters()
    
    def forward(self, x):
        user_id, features = x[:, :, 1], x[:, :, 1:]
        user_embed = self.user_embedding(user_id)
        ordinal_out = self.lstm_ordinal(features)
        concatenated_out = torch.cat((ordinal_out[:, -1], user_embed[:, -1]), dim=1)
        return self.out_trunk(concatenated_out)

