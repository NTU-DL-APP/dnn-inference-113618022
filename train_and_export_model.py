import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist

# === 建立 model/ 資料夾 ===
os.makedirs("model", exist_ok=True)

# === 載入資料集並正規化 ===
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# === 建立模型 ===
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === 訓練模型 ===
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# === 評估準確率 ===
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# === 儲存 H5 模型 ===
h5_path = "model/fashion_mnist.h5"
model.save(h5_path)

# === 將 H5 模型轉換為 JSON + NPZ ===
model_arch = []
weight_dict = {}

for layer in model.layers:
    layer_type = type(layer).__name__
    layer_name = layer.name
    config = {}

    if layer_type == "Dense":
        config = {
            "units": layer.units,
            "activation": layer.activation.__name__
        }
        weights = layer.get_weights()
        weight_dict[layer_name + "_kernel"] = weights[0]
        weight_dict[layer_name + "_bias"] = weights[1]
        model_arch.append({
            "name": layer_name,
            "type": "Dense",
            "config": config,
            "weights": [layer_name + "_kernel", layer_name + "_bias"]
        })

    elif layer_type == "Flatten":
        model_arch.append({
            "name": layer_name,
            "type": "Flatten",
            "config": {},
            "weights": []
        })

# === 儲存 JSON 與 NPZ ===
json_path = "model/fashion_mnist.json"
npz_path = "model/fashion_mnist.npz"
with open(json_path, "w") as f:
    json.dump(model_arch, f, indent=4)
np.savez(npz_path, **weight_dict)

print("✅ 模型轉換完成，已儲存：")
print(f"- {json_path}")
print(f"- {npz_path}")

# === （選擇性）執行測試 ===
# print("🚀 執行測試中...")
# os.system("python model_test.py")
