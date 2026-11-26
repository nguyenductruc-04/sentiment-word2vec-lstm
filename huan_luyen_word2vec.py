
# ==========================================================
# BƯỚC 7: HUẤN LUYỆN WORD2VEC
# Dự án: Sentiment Analysis - Word2Vec + LSTM
# Dataset: Sentiment140
# ==========================================================

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# 1. Khai báo đường dẫn
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "sentiment140_clean.csv"
SPLIT_PATH = BASE_DIR / "data_split.npz"
MODEL_PATH = BASE_DIR / "word2vec.model"

# 2. Cấu hình Word2Vec theo Notebook giảng viên
VECTOR_SIZE = 300  # Mỗi từ được biểu diễn bằng 300 số
WINDOW = 7         # Xem xét tối đa 7 từ ngữ cảnh mỗi phía
MIN_COUNT = 10     # Bỏ qua từ xuất hiện dưới 10 lần
EPOCHS = 30        # Số lượt huấn luyện Word2Vec
WORKERS = min(4, os.cpu_count() or 1)
CHUNK_SIZE = 50000

# 3. Đọc chỉ số Train đã tạo ở Bước 6
with np.load(SPLIT_PATH) as split:
    train_idx = split["train_idx"]

# Đánh dấu các dòng thuộc tập Train
train_mask = np.zeros(1600000, dtype=bool)
train_mask[train_idx] = True

print("Số mẫu Train:", train_mask.sum())


# 4. Tạo bộ đọc dữ liệu Train theo từng phần
# Không nạp toàn bộ danh sách token vào RAM.
# Có thể đọc lại nhiều lần để huấn luyện 30 epochs.
class TrainCorpus:
    def __iter__(self):
        offset = 0

        reader = pd.read_csv(
            DATA_PATH,
            usecols=["text"],
            keep_default_na=False,
            chunksize=CHUNK_SIZE
        )

        for chunk in reader:
            end = offset + len(chunk)

            # Chỉ chọn các dòng thuộc tập Train
            selected = chunk.loc[
                train_mask[offset:end], "text"
            ]

            # Chuyển từng câu thành danh sách từ
            for text in selected:
                words = str(text).split()

                if words:
                    yield words

            offset = end

        if offset != len(train_mask):
            raise ValueError("Số dòng Dataset không khớp!")


# 5. Theo dõi tiến độ huấn luyện
class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.start = time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        elapsed = time.time() - self.start

        print(
            f"Hoàn thành Epoch {self.epoch}/{EPOCHS}"
            f" - Thời gian tích lũy: {elapsed:.1f} giây",
            flush=True
        )


# 6. Khởi tạo mô hình Word2Vec
# sg=0: Sử dụng CBOW, giống mặc định của Notebook gốc.
corpus = TrainCorpus()

model = Word2Vec(
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    workers=WORKERS,
    sg=0,
    seed=42
)

# 7. Xây dựng từ vựng từ tập Train
print("\nĐANG XÂY DỰNG TỪ VỰNG...", flush=True)

model.build_vocab(corpus_iterable=corpus)

print("Số câu dùng học từ vựng:", model.corpus_count)
print("Kích thước từ vựng:", len(model.wv))

# 8. Huấn luyện mô hình
print("\nBẮT ĐẦU HUẤN LUYỆN WORD2VEC", flush=True)

model.train(
    corpus_iterable=corpus,
    total_examples=model.corpus_count,
    epochs=EPOCHS,
    callbacks=[EpochLogger()]
)

# 9. Lưu mô hình vào ổ D
model.save(str(MODEL_PATH))

print("\nĐÃ LƯU MÔ HÌNH:", MODEL_PATH)

# 10. Kiểm tra các từ gần nghĩa/ngữ cảnh
if "good" in model.wv:
    print("\nCác từ gần với 'good':")
    for word, score in model.wv.most_similar(
        "good", topn=5
    ):
        print(f"{word}: {score:.4f}")

print("\nHOÀN THÀNH BƯỚC 7!")
