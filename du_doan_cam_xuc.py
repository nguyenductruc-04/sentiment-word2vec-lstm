
# ==========================================================
# BƯỚC 12: DEMO DỰ ĐOÁN CẢM XÚC BẰNG LSTM
# Dataset đã học: Sentiment140
# Mô hình: Word2Vec + LSTM (PyTorch)
# GPU: NVIDIA GeForce GTX 1650
# ==========================================================

import re
import json
import torch
import torch.nn as nn
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from torch.nn.utils.rnn import pack_padded_sequence

# ----------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "lstm_best.pth"
VOCAB_PATH = BASE_DIR / "vocab_lstm.json"

MAX_LEN = 300
EMBEDDING_DIM = 300
HIDDEN_SIZE = 100

# ----------------------------------------------------------
# 2. CHỌN GPU ĐỂ DỰ ĐOÁN
# ----------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("Không tìm thấy GPU CUDA!")

device = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0))

# ----------------------------------------------------------
# 3. TẢI TỪ ĐIỂN ĐÃ TẠO KHI HUẤN LUYỆN
# ----------------------------------------------------------

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

print("Kích thước từ vựng:", len(vocab))

# ----------------------------------------------------------
# 4. CHUẨN BỊ STOPWORDS
# Sử dụng đúng dữ liệu NLTK trên ổ D
# ----------------------------------------------------------

nltk.data.path.insert(0, str(BASE_DIR / "nltk_data"))

stop_words = set(stopwords.words("english"))

# Giữ lại từ "not" vì mang ý nghĩa phủ định
stop_words.discard("not")

pattern = re.compile(
    r'@\S+|https?:\S+|http?:\S+|[^A-Za-z0-9]+'
)

# ----------------------------------------------------------
# 5. HÀM TIỀN XỬ LÝ VĂN BẢN
# Phải giống Bước 5 lúc chuẩn bị Dataset
# ----------------------------------------------------------

def preprocess(text):

    # Xóa username, URL và ký tự đặc biệt
    text = pattern.sub(" ", str(text))

    # Chuyển sang chữ thường
    text = text.lower()

    # Tách thành từng từ
    words = text.split()

    # Loại bỏ stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ----------------------------------------------------------
# 6. CHUYỂN VĂN BẢN THÀNH TOKEN ID
# ----------------------------------------------------------

def encode_text(text):

    # Tách văn bản đã xử lý thành các từ
    words = text.split()[-MAX_LEN:]

    # Chuyển từ sang ID
    # Từ không có trong vocab được gán ID = 1 (UNK)
    ids = [vocab.get(word, 1) for word in words]

    # Xử lý trường hợp câu rỗng
    if not ids:
        ids = [1]

    # Chuyển thành Tensor
    inputs = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device
    )

    # Độ dài câu, phục vụ LSTM
    lengths = torch.tensor([len(ids)], dtype=torch.long)

    return inputs, lengths

# ----------------------------------------------------------
# 7. ĐỊNH NGHĨA KIẾN TRÚC LSTM
# Phải trùng với kiến trúc trong train_lstm.py
# ----------------------------------------------------------

class SentimentLSTM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # Biểu diễn mỗi từ bằng vector 300 chiều
        self.embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            padding_idx=0
        )

        # Không cập nhật trọng số Word2Vec
        self.embedding.weight.requires_grad_(False)

        # Dropout giúp hạn chế Overfitting
        self.dropout = nn.Dropout(0.5)

        # LSTM học mối quan hệ giữa các từ
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # Lớp phân loại nhị phân
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, lengths):

        # Chuyển ID thành vector
        x = self.embedding(x)

        # Dropout (tự tắt khi ở chế độ eval)
        x = self.dropout(x)

        # Đóng gói chuỗi để LSTM xử lý
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Lấy trạng thái ẩn cuối cùng
        _, (hidden, _) = self.lstm(packed)

        # Đưa qua Dense
        return self.fc(hidden[-1]).squeeze(-1)

# ----------------------------------------------------------
# 8. TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN
# ----------------------------------------------------------

print("\nĐang tải mô hình LSTM tốt nhất...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model = SentimentLSTM(len(vocab)).to(device)

# Nạp trọng số đã học từ file checkpoint
model.load_state_dict(checkpoint["model"])

# Tắt Dropout, chuyển sang chế độ dự đoán
model.eval()

print("Checkpoint Epoch:", checkpoint["epoch"])
print(
    "Best Validation Accuracy:",
    f"{checkpoint['best_accuracy'] * 100:.2f}%"
)

print("ĐÃ TẢI MÔ HÌNH THÀNH CÔNG!")

# ----------------------------------------------------------
# 9. HÀM DỰ ĐOÁN CẢM XÚC
# ----------------------------------------------------------

def predict_sentiment(sentence):

    # Tiền xử lý giống lúc Training
    cleaned = preprocess(sentence)

    if not cleaned:
        print("Câu không còn từ hợp lệ sau tiền xử lý.")
        return

    # Kiểm tra từ nằm ngoài từ điển
    tokens = cleaned.split()
    unknown_count = sum(
        word not in vocab for word in tokens
    )

    # Chuyển thành Tensor
    inputs, lengths = encode_text(cleaned)

    # Không cần tính Gradient khi dự đoán
    with torch.inference_mode():

        logits = model(inputs, lengths)

        # Sigmoid chuyển logit thành giá trị 0-1
        probability = torch.sigmoid(logits).item()

    # Ngưỡng phân loại 0.5
    if probability >= 0.5:
        sentiment = "POSITIVE (Tích cực)"
    else:
        sentiment = "NEGATIVE (Tiêu cực)"

    print("\n========== KẾT QUẢ DỰ ĐOÁN ==========")
    print("Câu gốc:", sentence)
    print("Sau tiền xử lý:", cleaned)

    print("\nCẢM XÚC:", sentiment)

    print(
        f"Điểm Positive: {probability * 100:.2f}%"
    )
    print(
        f"Điểm Negative: {(1 - probability) * 100:.2f}%"
    )

    if unknown_count:
        print(
            f"Lưu ý: {unknown_count}/{len(tokens)} từ "
            "không nằm trong từ điển Word2Vec."
        )

    print("======================================")

# ----------------------------------------------------------
# 10. CHƯƠNG TRÌNH NHẬP CÂU TỪ BÀN PHÍM
# ----------------------------------------------------------

print("\n======================================")
print("DEMO SENTIMENT ANALYSIS - LSTM")
print("Nhập câu tiếng Anh để dự đoán cảm xúc")
print("Nhập q để thoát chương trình")
print("======================================")

while True:

    sentence = input("\nNhập câu tiếng Anh: ").strip()

    if sentence.lower() == "q":
        print("Đã thoát chương trình!")
        break

    if not sentence:
        print("Vui lòng nhập câu hợp lệ!")
        continue

    predict_sentiment(sentence)
