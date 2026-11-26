
# ==========================================================
# BƯỚC 8: XÂY DỰNG MÔ HÌNH LSTM BẰNG PYTORCH
# Dự án: Sentiment Analysis - Word2Vec + LSTM
# Thiết bị: NVIDIA GTX 1650 - CUDA
# ==========================================================

import json
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from gensim.models import Word2Vec

# 1. KHAI BÁO ĐƯỜNG DẪN
BASE_DIR = Path(__file__).resolve().parent
W2V_PATH = BASE_DIR / "word2vec.model"
VOCAB_PATH = BASE_DIR / "vocab_lstm.json"

# 2. CẤU HÌNH MÔ HÌNH
MAX_LEN = 300       # Chiều dài chuỗi như Notebook gốc
EMBEDDING_DIM = 300 # Số chiều vector Word2Vec
HIDDEN_SIZE = 100   # Số đơn vị ẩn của LSTM
DROPOUT = 0.5       # Dropout trước LSTM

# 3. KIỂM TRA GPU
# Yêu cầu CUDA để tránh vô tình chạy bằng CPU.
if not torch.cuda.is_available():
    raise RuntimeError("Không tìm thấy GPU CUDA!")

device = torch.device("cuda:0")

print("THIẾT BỊ HUẤN LUYỆN:", device)
print("GPU:", torch.cuda.get_device_name(0))

# 4. TẢI WORD2VEC ĐÃ HUẤN LUYỆN
print("\nĐang tải Word2Vec...")

w2v = Word2Vec.load(str(W2V_PATH))

# 5. XÂY DỰNG TỪ ĐIỂN
# PAD=0: Vị trí bổ sung cho câu ngắn.
# UNK=1: Từ không có trong bộ từ vựng.
vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for i, word in enumerate(w2v.wv.index_to_key, start=2):
    vocab[word] = i

VOCAB_SIZE = len(vocab)

print("Kích thước từ vựng:", VOCAB_SIZE)

# Lưu từ điển để sử dụng khi huấn luyện và dự đoán
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

print("Đã lưu từ điển:", VOCAB_PATH)

# 6. TẠO MA TRẬN EMBEDDING
# Mỗi từ có một vector 300 chiều.
embedding_matrix = np.zeros(
    (VOCAB_SIZE, EMBEDDING_DIM),
    dtype=np.float32
)

# Chép vector Word2Vec vào ma trận
# Hai hàng đầu dành cho PAD và UNK.
embedding_matrix[2:] = w2v.wv.vectors

print("Kích thước Embedding:", embedding_matrix.shape)


# 7. ĐỊNH NGHĨA MẠNG LSTM
class SentimentLSTM(nn.Module):

    def __init__(self, weights):
        super().__init__()

        # Embedding: Biến chỉ số từ thành vector Word2Vec
        # freeze=True: Giữ cố định trọng số Word2Vec
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(weights),
            freeze=True,
            padding_idx=0
        )

        # Dropout giúp hạn chế overfitting
        self.dropout = nn.Dropout(DROPOUT)

        # LSTM xử lý chuỗi vector từ
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # Dense: Chuyển trạng thái ẩn thành một logit
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):

        # Chuyển token ID thành vector
        x = self.embedding(x)

        # Áp dụng Dropout
        x = self.dropout(x)

        # LSTM học thông tin theo thứ tự của từ
        _, (hidden, _) = self.lstm(x)

        # Lấy trạng thái ẩn cuối cùng
        x = hidden[-1]

        # Giá trị đầu ra chưa qua Sigmoid
        return self.fc(x).squeeze(-1)


# 8. KHỞI TẠO MÔ HÌNH VÀ CHUYỂN SANG GPU
model = SentimentLSTM(embedding_matrix)
model = model.to(device)

print("\nKIẾN TRÚC MÔ HÌNH:")
print(model)

# 9. ĐẾM SỐ THAM SỐ
total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("\nTổng số tham số:", total_params)
print("Số tham số được huấn luyện:", trainable_params)

# 10. THỬ CHẠY MÔ HÌNH TRÊN GPU
# Đây chỉ là kiểm tra forward pass, chưa training.
example_texts = [
    "i love this movie",
    "this movie is not good",
    "i hate this movie"
]

def encode_text(text):
    # Chuyển từ thành ID trong từ điển
    tokens = text.lower().split()
    ids = [vocab.get(word, 1) for word in tokens]

    # Cắt chuỗi về tối đa 300 token
    ids = ids[-MAX_LEN:]

    # Thêm PAD vào đầu chuỗi
    ids = [0] * (MAX_LEN - len(ids)) + ids
    return ids

# Tạo tensor đầu vào và đưa lên GPU
x = torch.tensor(
    [encode_text(t) for t in example_texts],
    dtype=torch.long,
    device=device
)

# Chuyển sang chế độ đánh giá
model.eval()

with torch.inference_mode():
    logits = model(x)
    probabilities = torch.sigmoid(logits)

print("\nKIỂM TRA FORWARD PASS:")
print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(probabilities.shape))
print("Device:", probabilities.device)

print("\nXác suất thử nghiệm (CHƯA HUẤN LUYỆN):")
for text, prob in zip(example_texts, probabilities.tolist()):
    print(f"{text}: {prob:.4f}")

print("\nHOÀN THÀNH BƯỚC 8!")
