
# ==========================================================
# BƯỚC 9: HUẤN LUYỆN LSTM BẰNG GPU GTX 1650
# Dataset: Sentiment140
# Mô hình: Word2Vec + Embedding + Dropout + LSTM + Dense
# ==========================================================

import os
import csv
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# ----------------------------------------------------------
# 1. CẤU HÌNH HUẤN LUYỆN
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "sentiment140_clean.csv"
SPLIT_PATH = BASE_DIR / "data_split.npz"
VOCAB_PATH = BASE_DIR / "vocab_lstm.json"
W2V_PATH = BASE_DIR / "word2vec.model"

LAST_PATH = BASE_DIR / "lstm_last.pth"
BEST_PATH = BASE_DIR / "lstm_best.pth"
HISTORY_PATH = BASE_DIR / "history_lstm.csv"

MAX_LEN = 300
EMBEDDING_DIM = 300
HIDDEN_SIZE = 100

BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Chạy thử 1 epoch trước, sau đó đổi thành 15
TOTAL_EPOCHS = 15

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ----------------------------------------------------------
# 2. KIỂM TRA GPU
# ----------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("Không nhận được GPU CUDA!")

device = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)

# ----------------------------------------------------------
# 3. ĐỌC DATASET VÀ CHỈ SỐ TRAIN/TEST
# ----------------------------------------------------------

print("\nĐang đọc Dataset...", flush=True)

df = pd.read_csv(
    DATA_PATH,
    dtype={"target": "int8", "text": str},
    keep_default_na=False
)

texts = df["text"].tolist()
labels = df["target"].to_numpy()

with np.load(SPLIT_PATH) as split:
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

# Chỉ dùng tập Train để huấn luyện và validation.
# Giữ nguyên tập Test cho bước đánh giá cuối cùng.
n_val = int(len(train_idx) * 0.1)

actual_train_idx = train_idx[:-n_val]
val_idx = train_idx[-n_val:]

print("Training:", len(actual_train_idx))
print("Validation:", len(val_idx))
print("Testing (chưa sử dụng):", len(test_idx))

# ----------------------------------------------------------
# 4. TẢI WORD2VEC VÀ TỪ ĐIỂN
# ----------------------------------------------------------

print("\nĐang tải Word2Vec...", flush=True)

w2v = Word2Vec.load(str(W2V_PATH))

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Tạo ma trận Embedding theo đúng từ điển Bước 8.
embedding_matrix = np.zeros(
    (len(vocab), EMBEDDING_DIM),
    dtype=np.float32
)

for i, word in enumerate(w2v.wv.index_to_key, start=2):
    embedding_matrix[i] = w2v.wv[word]

print("Kích thước từ vựng:", len(vocab))

# ----------------------------------------------------------
# 5. TẠO DATASET PYTORCH
# ----------------------------------------------------------

class SentimentDataset(Dataset):

    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        # Lấy câu và nhãn theo chỉ số đã chia
        row = int(self.indices[index])
        text = texts[row]
        label = float(labels[row])

        # Chuyển văn bản thành danh sách token ID
        tokens = text.split()[-MAX_LEN:]
        ids = [vocab.get(word, 1) for word in tokens]

        # Tránh chuỗi rỗng khi đưa vào LSTM
        if not ids:
            ids = [1]

        return ids, label


# ----------------------------------------------------------
# 6. TẠO BATCH VỚI DYNAMIC PADDING
# ----------------------------------------------------------

def collate_fn(batch):

    sequences, targets = zip(*batch)

    # Độ dài thực tế của từng câu
    lengths = torch.tensor(
        [len(seq) for seq in sequences],
        dtype=torch.long
    )

    # Chỉ padding đến câu dài nhất trong batch
    max_length = int(lengths.max())

    inputs = torch.zeros(
        (len(sequences), max_length),
        dtype=torch.long
    )

    for i, seq in enumerate(sequences):
        inputs[i, :len(seq)] = torch.tensor(
            seq, dtype=torch.long
        )

    targets = torch.tensor(targets, dtype=torch.float32)

    return inputs, lengths, targets


train_loader = DataLoader(
    SentimentDataset(actual_train_idx),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    SentimentDataset(val_idx),
    batch_size=256,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn
)

# ----------------------------------------------------------
# 7. XÂY DỰNG MÔ HÌNH LSTM
# ----------------------------------------------------------

class SentimentLSTM(nn.Module):

    def __init__(self, weights):
        super().__init__()

        # Dùng vector Word2Vec đã huấn luyện
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(weights),
            freeze=True,
            padding_idx=0
        )

        self.dropout = nn.Dropout(0.5)

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, lengths):

        # Chuyển token thành vector từ
        x = self.embedding(x)

        # Hạn chế overfitting
        x = self.dropout(x)

        # Bỏ qua vị trí padding khi tính LSTM
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        # Dùng trạng thái ẩn cuối cùng
        output = self.fc(hidden[-1])

        # Trả về logit để tính BCEWithLogitsLoss
        return output.squeeze(-1)


model = SentimentLSTM(embedding_matrix).to(device)

# ----------------------------------------------------------
# 8. LOSS VÀ OPTIMIZER
# ----------------------------------------------------------

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ----------------------------------------------------------
# 9. KHÔI PHỤC CHECKPOINT NẾU CÓ
# ----------------------------------------------------------

start_epoch = 1
best_accuracy = 0.0

if LAST_PATH.exists():

    print("\nPhát hiện checkpoint cũ!")

    checkpoint = torch.load(
        LAST_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint["epoch"] + 1
    best_accuracy = checkpoint["best_accuracy"]

    print("Tiếp tục từ Epoch:", start_epoch)
    print("Best Validation Accuracy:", best_accuracy)

# ----------------------------------------------------------
# 10. HÀM CHẠY MỘT EPOCH
# ----------------------------------------------------------

def run_epoch(loader, training=True):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, (inputs, lengths, targets) in enumerate(
        loader, start=1
    ):

        # Chuyển dữ liệu lên GPU
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        # Chỉ tính gradient ở chế độ Train
        with torch.set_grad_enabled(training):

            logits = model(inputs, lengths)
            loss = criterion(logits, targets)

            if training:
                loss.backward()

                # Giới hạn gradient để LSTM ổn định
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()

        # Tính Accuracy
        predictions = (logits >= 0).float()

        batch_size = len(targets)

        total_loss += loss.item() * batch_size
        total_correct += (
            predictions == targets
        ).sum().item()

        total_samples += batch_size

        # Hiển thị tiến độ sau mỗi 200 batch
        if step % 200 == 0 or step == len(loader):

            phase = "Train" if training else "Val"

            print(
                f"{phase} Batch {step}/{len(loader)}"
                f" - Loss: {total_loss / total_samples:.4f}"
                f" - Acc: {total_correct / total_samples:.4f}",
                flush=True
            )

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ----------------------------------------------------------
# 11. VÒNG LẶP HUẤN LUYỆN
# ----------------------------------------------------------

print("\nBẮT ĐẦU HUẤN LUYỆN LSTM")
print("Epoch bắt đầu:", start_epoch)
print("Epoch kết thúc:", TOTAL_EPOCHS)
print("Batch size:", BATCH_SIZE)

for epoch in range(start_epoch, TOTAL_EPOCHS + 1):

    print(f"\n========== EPOCH {epoch}/{TOTAL_EPOCHS} ==========")

    # Huấn luyện trên tập Train
    train_loss, train_acc = run_epoch(
        train_loader,
        training=True
    )

    # Đánh giá trên tập Validation
    val_loss, val_acc = run_epoch(
        val_loader,
        training=False
    )

    print(f"\nEpoch {epoch} hoàn thành!")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")

    # Cập nhật kết quả tốt nhất
    is_best = val_acc > best_accuracy

    if is_best:
        best_accuracy = val_acc

    # Lưu cả model và optimizer để có thể train tiếp
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_accuracy": best_accuracy
    }

    # Lưu checkpoint mới nhất
    temp_path = BASE_DIR / "lstm_last.tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, LAST_PATH)

    # Lưu mô hình tốt nhất theo Validation Accuracy
    if is_best:
        torch.save(checkpoint, BEST_PATH)
        print("Đã cập nhật mô hình tốt nhất!")

    # Lưu lịch sử Accuracy và Loss
    new_file = not HISTORY_PATH.exists()

    with open(
        HISTORY_PATH, "a", newline="", encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if new_file:
            writer.writerow([
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy"
            ])

        writer.writerow([
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc
        ])

    print("Đã lưu checkpoint Epoch:", epoch)

print("\nHOÀN THÀNH HUẤN LUYỆN!")
print("Best Validation Accuracy:", best_accuracy)
print("Checkpoint:", LAST_PATH)
