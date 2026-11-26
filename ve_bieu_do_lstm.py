
# =====================================================
# BƯỚC 11: VẼ BIỂU ĐỒ HUẤN LUYỆN LSTM
# Accuracy và Loss qua 15 Epochs
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Xác định đường dẫn lưu dữ liệu
BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BASE_DIR / "history_lstm.csv"

# 2. Đọc lịch sử huấn luyện
df = pd.read_csv(HISTORY_PATH)

# Loại bỏ epoch trùng nếu từng chạy lại
df = df.drop_duplicates(
    subset=["epoch"], keep="last"
).sort_values("epoch")

print("LỊCH SỬ HUẤN LUYỆN:")
print(df.to_string(index=False))

# 3. VẼ BIỂU ĐỒ ACCURACY
plt.figure(figsize=(9, 5))

plt.plot(
    df["epoch"],
    df["train_accuracy"] * 100,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    df["epoch"],
    df["val_accuracy"] * 100,
    marker="s",
    label="Validation Accuracy"
)

plt.title("LSTM Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(df["epoch"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    BASE_DIR / "bieu_do_accuracy_lstm.png",
    dpi=200
)

plt.close()

# 4. VẼ BIỂU ĐỒ LOSS
plt.figure(figsize=(9, 5))

plt.plot(
    df["epoch"],
    df["train_loss"],
    marker="o",
    label="Train Loss"
)

plt.plot(
    df["epoch"],
    df["val_loss"],
    marker="s",
    label="Validation Loss"
)

plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(df["epoch"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    BASE_DIR / "bieu_do_loss_lstm.png",
    dpi=200
)

plt.close()

# 5. XÁC ĐỊNH EPOCH TỐT NHẤT
best = df.loc[df["val_accuracy"].idxmax()]

print("\nKẾT QUẢ TỐT NHẤT:")
print("Best Epoch:", int(best["epoch"]))
print(
    "Best Validation Accuracy:",
    f"{best['val_accuracy'] * 100:.2f}%"
)

print("\nĐÃ LƯU 2 BIỂU ĐỒ:")
print("bieu_do_accuracy_lstm.png")
print("bieu_do_loss_lstm.png")

print("\nHOÀN THÀNH BƯỚC 11!")
