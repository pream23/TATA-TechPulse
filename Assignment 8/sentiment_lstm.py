import re
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# -----------------------------
# 1. Dataset
# -----------------------------

positive = [
    "excellent service",
    "very good experience",
    "amazing product",
    "great quality",
    "I love this car",
    "comfortable and reliable",
    "excellent performance",
    "very happy with the service",
    "good customer support",
    "fantastic experience",
    "the product is wonderful",
    "high quality product",
    "fast and efficient service",
    "I really enjoyed it",
    "best experience ever",
    "very satisfied",
    "the car is excellent",
    "great performance",
    "friendly service",
    "I recommend this product",
]

negative = [
    "terrible service",
    "very bad experience",
    "awful product",
    "poor quality",
    "I hate this car",
    "uncomfortable and unreliable",
    "terrible performance",
    "very unhappy with the service",
    "bad customer support",
    "horrible experience",
    "the product is disappointing",
    "low quality product",
    "slow and inefficient service",
    "I really disliked it",
    "worst experience ever",
    "very dissatisfied",
    "the car is terrible",
    "poor performance",
    "unfriendly service",
    "I do not recommend this product",
]

texts = positive + negative
labels = [1] * len(positive) + [0] * len(negative)


# -----------------------------
# 2. Tokenization
# -----------------------------

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def build_vocab(texts):
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for text in texts:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def encode(text, vocab, max_len=10):
    tokens = tokenize(text)

    ids = [
        vocab.get(word, vocab["<UNK>"])
        for word in tokens
    ]

    ids = ids[:max_len]

    while len(ids) < max_len:
        ids.append(vocab["<PAD>"])

    return ids


# -----------------------------
# 3. LSTM Model
# -----------------------------

class SentimentLSTM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            64,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(
            embedded
        )

        last_hidden = hidden[-1]

        last_hidden = self.dropout(
            last_hidden
        )

        return self.fc(
            last_hidden
        ).squeeze(1)


# -----------------------------
# 4. Prepare data
# -----------------------------

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

vocab = build_vocab(texts)

X = torch.tensor(
    [encode(text, vocab) for text in texts],
    dtype=torch.long
)

y = torch.tensor(
    labels,
    dtype=torch.float32
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

train_dataset = TensorDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)


# -----------------------------
# 5. Train model
# -----------------------------

model = SentimentLSTM(
    len(vocab)
)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005
)

epochs = 100

print("\nTraining LSTM...\n")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        output = model(batch_x)

        loss = criterion(
            output,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:

        average_loss = (
            total_loss /
            len(train_loader)
        )

        print(
            f"Epoch {epoch + 1:03d} "
            f"| Loss: {average_loss:.4f}"
        )


# -----------------------------
# 6. Evaluation
# -----------------------------

model.eval()

with torch.no_grad():

    probabilities = torch.sigmoid(
        model(X_test)
    )

    predictions = (
        probabilities >= 0.5
    ).int().numpy()

actual = y_test.int().numpy()


accuracy = accuracy_score(
    actual,
    predictions
)

precision = precision_score(
    actual,
    predictions,
    zero_division=0
)

recall = recall_score(
    actual,
    predictions,
    zero_division=0
)

f1 = f1_score(
    actual,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    actual,
    predictions
)


print("\n-----------------------------")
print("MODEL EVALUATION")
print("-----------------------------")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print("\nConfusion Matrix:")
print(cm)


# -----------------------------
# 7. Custom predictions
# -----------------------------

def predict_sentiment(sentence):

    model.eval()

    encoded = torch.tensor(
        [encode(sentence, vocab)],
        dtype=torch.long
    )

    with torch.no_grad():

        probability = torch.sigmoid(
            model(encoded)
        ).item()

    sentiment = (
        "Positive"
        if probability >= 0.5
        else "Negative"
    )

    return sentiment, probability


test_sentences = [
    "excellent quality and great service",
    "terrible product and poor service",
    "I am very happy with this car",
    "this was a horrible experience"
]

print("\n-----------------------------")
print("CUSTOM PREDICTIONS")
print("-----------------------------")

for sentence in test_sentences:

    sentiment, probability = predict_sentiment(
        sentence
    )

    print(
        f"{sentence} -> "
        f"{sentiment} "
        f"({probability:.3f})"
    )


# -----------------------------
# 8. Save model
# -----------------------------

torch.save(
    model.state_dict(),
    "sentiment_lstm.pth"
)

print("\nModel saved as sentiment_lstm.pth")