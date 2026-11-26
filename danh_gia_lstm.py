
# ==========================================================
# BƯỚC 10: ĐÁNH GIÁ LSTM TRÊN TẬP TEST
# Mô hình: Word2Vec + LSTM (PyTorch)
# GPU: NVIDIA GeForce GTX 1650
# ==========================================================

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. KHAI BÁO ĐƯỜNG DẪN
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "sentiment140_clean.csv"
SPLIT_PATH = BASE_DIR / "data_split.npz"
VOCAB_PATH = BASE_DIR / "vocab_lstm.json"
MODEL_PATH = BASE_DIR / "lstm_best.pth"

MAX_LEN = 300
BATCH_SIZE = 256

# 2. KIỂM TRA GPU
if not torch.cuda.is_available():
    raise RuntimeError("Không tìm thấy GPU CUDA!")

device = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0))

# 3. ĐỌC TỪ ĐIỂN VÀ TẬP TEST
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

df = pd.read_csv(
    DATA_PATH,
    usecols=["target", "text"],
    dtype={"target": "int8", "text": str},
    keep_default_na=False
)

with np.load(SPLIT_PATH) as split:
    test_idx = split["test_idx"]

# Chỉ lấy những mẫu Test chưa dùng để huấn luyện LSTM
test_df = df.iloc[test_idx].copy()
del df

test_texts = test_df["text"].tolist()
test_labels = test_df["target"].to_numpy()

print("Tổng số mẫu Test:", len(test_texts))

# 4. TẠO DATASET
class TestDataset(Dataset):

    def __len__(self):
        return len(test_texts)

    def __getitem__(self, idx):
        text = test_texts[idx]
        label = int(test_labels[idx])

        # Mã hóa giống quá trình Training
        words = text.split()[-MAX_LEN:]
        ids = [vocab.get(w, 1) for w in words]

        if not ids:
            ids = [1]

        return ids, label


# 5. PADDING BÊN PHẢI GIỐNG KHI TRAINING
def collate_fn(batch):
    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(s) for s in sequences],
        dtype=torch.long
    )

    inputs = torch.zeros(
        len(sequences),
        int(lengths.max()),
        dtype=torch.long
    )

    for i, seq in enumerate(sequences):
        inputs[i, :len(seq)] = torch.tensor(
            seq, dtype=torch.long
        )

    labels = torch.tensor(labels, dtype=torch.long)

    return inputs, lengths, labels


test_loader = DataLoader(
    TestDataset(),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn
)

# 6. ĐỊNH NGHĨA KIẾN TRÚC GIỐNG KHI TRAIN
class SentimentLSTM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            300,
            padding_idx=0
        )

        self.embedding.weight.requires_grad_(False)

        self.dropout = nn.Dropout(0.5)

        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(100, 1)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = self.dropout(x)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        return self.fc(hidden[-1]).squeeze(-1)


# 7. TẢI CHECKPOINT TỐT NHẤT
print("\nĐang tải mô hình tốt nhất...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model = SentimentLSTM(len(vocab)).to(device)
model.load_state_dict(checkpoint["model"])

# Tắt Dropout và chuyển sang chế độ đánh giá
model.eval()

print("Checkpoint Epoch:", checkpoint["epoch"])
print(
    "Best Validation Accuracy:",
    checkpoint["best_accuracy"]
)

# 8. DỰ ĐOÁN TRÊN TẬP TEST
all_true = []
all_pred = []

print("\nBẮT ĐẦU ĐÁNH GIÁ TEST", flush=True)

with torch.inference_mode():

    for step, (inputs, lengths, labels) in enumerate(
        test_loader, start=1
    ):
        inputs = inputs.to(device, non_blocking=True)

        logits = model(inputs, lengths)

        # Logit >= 0 tương đương xác suất >= 0.5
        predictions = (logits >= 0).long()

        all_true.extend(labels.tolist())
        all_pred.extend(predictions.cpu().tolist())

        if step % 250 == 0 or step == len(test_loader):
            print(
                f"Test Batch {step}/{len(test_loader)}",
                flush=True
            )

# 9. TÍNH CÁC CHỈ SỐ ĐÁNH GIÁ
accuracy = accuracy_score(all_true, all_pred)

report = classification_report(
    all_true,
    all_pred,
    labels=[0, 1],
    target_names=["Negative", "Positive"],
    digits=4,
    zero_division=0
)

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1]
)

print("\n========== KẾT QUẢ TEST ==========")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Accuracy (%): {accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT:")
print(report)

print("\nCONFUSION MATRIX:")
print(cm)

# 10. LƯU BÁO CÁO ĐÁNH GIÁ
REPORT_PATH = BASE_DIR / "ket_qua_test.txt"

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Best checkpoint epoch: {checkpoint['epoch']}\n")
    f.write(f"Test samples: {len(all_true)}\n")
    f.write(f"Test Accuracy: {accuracy:.6f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

# 11. VẼ CONFUSION MATRIX
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")

ax.set_title("LSTM - Confusion Matrix (Test Set)")
plt.tight_layout()

IMAGE_PATH = BASE_DIR / "confusion_matrix_lstm.png"

plt.savefig(
    IMAGE_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\nĐÃ LƯU:")
print(REPORT_PATH)
print(IMAGE_PATH)
print("\nHOÀN THÀNH BƯỚC 10!")
