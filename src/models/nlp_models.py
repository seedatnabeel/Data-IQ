# third party
import torch
import torch.nn as nn


class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        Argument:
        reviews: a numpy array
        targets: a vector array
        """
        self.reviews = reviews
        self.target = targets

    def __len__(self):
        # return length of dataset
        return len(self.reviews)

    def __getitem__(self, index):
        # return review and target of that index in torch tensor
        review = torch.tensor(self.reviews[index, :], dtype=torch.long)
        target = torch.tensor(self.target[index], dtype=torch.float)

        return {"review": review, "target": target}


class LSTM(nn.Module):
    def __init__(self, embedding_matrix):

        super(LSTM, self).__init__()

        # Number of words = number of rows in embedding matrix
        num_words = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        # input embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embedding_dim,
        )

        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32),
        )

        # requires grad is false since we use pre-trained word embeddings
        self.embedding.weight.requires_grad = False

        # instantiate LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            128,
            bidirectional=True,
            batch_first=True,
        )
        self.out = nn.Linear(512, 2)

    def forward(self, x):
        # embed
        x = self.embedding(x)

        # pass embedding to lstm
        hidden, _ = self.lstm(x)

        # apply mean and max pooling on lstm output
        avg_pool = torch.mean(hidden, 1)
        max_pool, index_max_pool = torch.max(hidden, 1)

        out = torch.cat((avg_pool, max_pool), 1)

        # final pred
        out = self.out(out)
        return out
