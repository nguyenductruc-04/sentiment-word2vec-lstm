# KIỂM TRA GPU CHO DỰ ÁN LSTM

import sys
import torch

# Kiểm tra môi trường Python
print("Python:", sys.executable)

# Kiểm tra PyTorch
print("PyTorch:", torch.__version__)

# Kiểm tra CUDA
print("CUDA:", torch.cuda.is_available())

# Chọn GPU để huấn luyện
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Thiết bị:", device)

# Hiển thị thông tin GPU
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    ), "GB")