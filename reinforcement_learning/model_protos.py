# %load model_protos.py
import pdb

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

class LSTMEmbedUserProject(ModelBase):
    def __init__(self, n_features, n_sequences, embedding_matrix, hidden_size=32, dropout=0.2, lr=0.01, batch_size=256, zero_heuristic=False) -> None:
        super().__init__()

        self.lstm_ordinal = nn.LSTM(
            input_size=n_features - 3,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.user_embedding = nn.Embedding(
            num_embeddings=embedding_matrix['user_id'][0] + 1,
            embedding_dim=embedding_matrix['user_id'][1],
            padding_idx=0
        )

        self.project_embedding = nn.Embedding(
            num_embeddings=embedding_matrix['project_id'][0] + 1,
            embedding_dim=embedding_matrix['project_id'][1],
            padding_idx=0
        )

        self.lstm_embedding = nn.LSTM(
            input_size=embedding_matrix['user_id'][1] + embedding_matrix['project_id'][1],
            hidden_size=(embedding_matrix['user_id'][1] + embedding_matrix['project_id'][1]) // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.learning_rate = lr
        self.batch_size = batch_size
        self.model_name = 'embedded'

        trunk_linear = hidden_size + ((embedding_matrix['user_id'][1] + embedding_matrix['project_id'][1]) // 3)

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
        ordinal_features, categorical_features = x[:, :, :ORDINAL_FEATURE_INDEX], x[:, :, ORDINAL_FEATURE_INDEX:].int()


        user, project = categorical_features[:, :, 0], categorical_features[:, :, 1]

        """
        assert no more than 2 unique users: padded and value
        """



        user, project = self.user_embedding(user), self.project_embedding(project)
        categorical_features = torch.cat((user, project), dim=2)
        ordinal_out = self.lstm_ordinal(ordinal_features)
        categorical_out = self.lstm_embedding(categorical_features)
        return self.out_trunk(torch.cat((ordinal_out[0][:, -1], categorical_out[0][:, -1]), dim=1))
    

class LSTMEmbedUser(ModelBase):
    def __init__(self, n_features, n_sequences, embedding_matrix, hidden_size=32, dropout=0.2, lr=0.01, batch_size=256, zero_heuristic=False) -> None:
        super().__init__()

        self.lstm_ordinal = nn.LSTM(
            input_size=n_features - 3,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.user_embedding = nn.Embedding(
            num_embeddings=embedding_matrix['user_id'][0] + 1,
            embedding_dim=embedding_matrix['user_id'][1],
            padding_idx=0
        )

        self.lstm_embedding = nn.LSTM(
            input_size=embedding_matrix['user_id'][1],
            hidden_size = max(3, (embedding_matrix['user_id'][1]) // 3),
            # hidden_size=(embedding_matrix['user_id'][1]) // 3,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )


        self.learning_rate = lr
        self.batch_size = batch_size
        self.model_name = 'embedded_user'

        trunk_linear = hidden_size + max(((embedding_matrix['user_id'][1]) // 3), 3)

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
        ordinal_features, categorical_features = x[:, :, :ORDINAL_FEATURE_INDEX], x[:, :, ORDINAL_FEATURE_INDEX:].int()
        user = categorical_features[:, :, 0]
        user = self.user_embedding(user)
        ordinal_out = self.lstm_ordinal(ordinal_features)
        user_out = self.lstm_embedding(user)
        return self.out_trunk(torch.cat((ordinal_out[0][:, -1], user_out[0][:, -1]), dim=1))


class LSTMEmbedOneLSTM(ModelBase):
    def __init__(self, n_features, n_sequences, embedding_matrix, hidden_size=32, dropout=0.2, lr=0.01, batch_size=256, zero_heuristic=False) -> None:
        super().__init__()

        self.lstm_combined = nn.LSTM(
            input_size=(n_features - 3) + embedding_matrix['user_id'][1],
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.user_embedding = nn.Embedding(
            num_embeddings=embedding_matrix['user_id'][0] + 1,
            embedding_dim=embedding_matrix['user_id'][1],
            padding_idx=0
        )

        self.learning_rate = lr
        self.batch_size = batch_size
        self.model_name = 'embedded_one_lstm'

        self.out = nn.Linear(hidden_size, 1)

        self.n_sequences = n_sequences
        self.zero_heuristic = zero_heuristic

        self.save_hyperparameters()
    
    def forward(self, x):
        ordinal_features, categorical_features = x[:, :, :ORDINAL_FEATURE_INDEX], x[:, :, ORDINAL_FEATURE_INDEX:].int()
        user = categorical_features[:, :, 0]
        user = self.user_embedding(user)
        features = torch.cat((ordinal_features, user), dim=2)
        out = self.lstm_combined(features)
        return self.out(out[0][:, -1])
