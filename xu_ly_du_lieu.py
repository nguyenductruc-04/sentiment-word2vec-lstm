
# ==========================================================
# BƯỚC 5: TIỀN XỬ LÝ VĂN BẢN CHO MÔ HÌNH LSTM
# Dataset: Sentiment140 - 1.600.000 tweet
# ==========================================================

import re
import nltk
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords

# 1. Khai báo thư mục dự án và đường dẫn dữ liệu
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "training.1600000.processed.noemoticon.csv"
OUTPUT_FILE = BASE_DIR / "sentiment140_clean.csv"

# Sử dụng dữ liệu NLTK trên ổ D
nltk.data.path.insert(0, str(BASE_DIR / "nltk_data"))

# 2. Tải danh sách Stopwords tiếng Anh đã chuẩn bị
stop_words = set(stopwords.words("english"))

# Giữ lại từ "not" vì từ này ảnh hưởng đến cảm xúc.
# Ví dụ: "good" và "not good" có ý nghĩa khác nhau.
stop_words.discard("not")

# 3. Biểu thức chính quy dùng để làm sạch văn bản
pattern = re.compile(
    r'@\S+|https?:\S+|http?:\S+|[^A-Za-z0-9]+'
)

# 4. Hàm tiền xử lý một câu tweet
def preprocess(text):

    # Loại bỏ tên người dùng, URL và ký tự đặc biệt
    text = pattern.sub(" ", str(text))

    # Chuyển toàn bộ văn bản sang chữ thường
    text = text.lower()

    # Tách câu thành các từ
    words = text.split()

    # Loại bỏ Stopwords nhưng vẫn giữ "not"
    words = [word for word in words if word not in stop_words]

    # Ghép các từ lại thành câu
    return " ".join(words)


# 5. Đọc dữ liệu theo từng phần để tiết kiệm RAM
columns = ["target", "ids", "date", "flag", "user", "text"]

chunk_size = 50000
total_processed = 0

print("BẮT ĐẦU TIỀN XỬ LÝ DATASET SENTIMENT140")

reader = pd.read_csv(
    INPUT_FILE,
    encoding="latin-1",
    header=None,
    names=columns,
    usecols=["target", "text"],
    chunksize=chunk_size
)

# 6. Xử lý và lưu từng phần vào ổ D
for i, chunk in enumerate(reader, start=1):

    # Chuẩn hóa nhãn: 0 = Negative, 1 = Positive
    chunk["target"] = chunk["target"].replace({4: 1})

    # Làm sạch văn bản
    chunk["text"] = chunk["text"].map(preprocess)

    # Hiển thị ví dụ sau khi xử lý ở phần đầu tiên
    if i == 1:
        print("\n5 câu sau khi tiền xử lý:")
        print(chunk.head().to_string(index=False))

    # Ghi dữ liệu ra CSV, không giữ toàn bộ trong RAM
    chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if i == 1 else "a",
        header=(i == 1),
        index=False,
        encoding="utf-8"
    )

    total_processed += len(chunk)

    print(f"Đã xử lý: {total_processed:,}/1,600,000")

print("\nHOÀN THÀNH TIỀN XỬ LÝ!")
print("Tổng số mẫu:", total_processed)
print("File kết quả:", OUTPUT_FILE)
