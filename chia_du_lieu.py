
# ==========================================================
# BƯỚC 6: CHIA DATASET THÀNH TRAIN VÀ TEST
# Dự án: Sentiment Analysis sử dụng Word2Vec + LSTM
# ==========================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Khai báo đường dẫn dữ liệu
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sentiment140_clean.csv"
SPLIT_PATH = BASE_DIR / "data_split.npz"

# 2. Đọc dữ liệu đã tiền xử lý
# keep_default_na=False: Giữ câu rỗng thành chuỗi rỗng.
df = pd.read_csv(
    DATA_PATH,
    dtype={"target": "int8", "text": str},
    keep_default_na=False
)

print("TỔNG SỐ MẪU:", len(df))

# 3. Tạo chỉ số của từng mẫu dữ liệu
indices = np.arange(len(df), dtype=np.int32)

# 4. Chia dữ liệu theo tỷ lệ 80% Train - 20% Test
# random_state=123: Giữ kết quả chia cố định.
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=123
)

# 5. Lấy dữ liệu Train và Test để kiểm tra
train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

print("\nKẾT QUẢ CHIA DỮ LIỆU")
print("Số mẫu Train:", len(train_df))
print("Số mẫu Test:", len(test_df))

# 6. Kiểm tra phân bố nhãn cảm xúc
print("\nPhân bố nhãn Train:")
print(train_df["target"].value_counts().sort_index())

print("\nPhân bố nhãn Test:")
print(test_df["target"].value_counts().sort_index())

# 7. Lưu chỉ số Train/Test thay vì nhân đôi toàn bộ CSV
# Giúp tiết kiệm dung lượng ổ đĩa.
np.savez_compressed(
    SPLIT_PATH,
    train_idx=train_idx,
    test_idx=test_idx
)

print("\nĐÃ LƯU CHỈ SỐ CHIA DỮ LIỆU")
print("File:", SPLIT_PATH)

# 8. Kiểm tra kích thước file vừa lưu
size_mb = SPLIT_PATH.stat().st_size / (1024 ** 2)
print(f"Dung lượng: {size_mb:.2f} MB")

print("\nHOÀN THÀNH BƯỚC 6!")
