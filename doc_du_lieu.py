
# BƯỚC 4: ĐỌC VÀ KIỂM TRA DATASET SENTIMENT140
# Dataset này được sử dụng để huấn luyện mô hình phân tích cảm xúc LSTM.

from pathlib import Path
import pandas as pd

# 1. Xác định đường dẫn Dataset
# File CSV nằm cùng thư mục với file Python này.
DATA_PATH = Path(__file__).resolve().parent / "training.1600000.processed.noemoticon.csv"

# 2. Tên 6 cột trong bộ dữ liệu gốc
columns = ["target", "ids", "date", "flag", "user", "text"]

# 3. Đọc dữ liệu bằng Pandas
# encoding='latin-1': Đọc đúng định dạng ký tự của Dataset.
# header=None: File CSV gốc không có dòng tiêu đề.
# usecols: Chỉ lấy nhãn và văn bản để tiết kiệm RAM.
df = pd.read_csv(
    DATA_PATH,
    encoding="latin-1",
    header=None,
    names=columns,
    usecols=["target", "text"]
)

# 4. Hiển thị kích thước Dataset
print("Kích thước Dataset:", df.shape)
print("Tổng số mẫu:", len(df))

# 5. Hiển thị 5 dòng dữ liệu đầu tiên
print("\n5 dòng dữ liệu đầu tiên:")
print(df.head().to_string(index=False))

# 6. Kiểm tra số lượng mẫu thuộc từng nhãn
print("\nSố lượng mẫu theo nhãn:")
print(df["target"].value_counts().sort_index())

# 7. Kiểm tra dữ liệu bị thiếu
print("\nSố giá trị bị thiếu:")
print(df.isnull().sum())
