import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist

os.makedirs("model", exist_ok=True)

# === 資料載入與正規化 ===
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# === 改良模型架構（加大層數 + 擴展參數）===
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === 增加訓練輪數以穩定泛化效果 ===
model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# === 儲存為 .h5 模型 ===
model.save("model/fashion_mnist.h5")

# === 轉換為 json + npz ===
model_arch = []
weight_dict = {}

for layer in model.layers:
    layer_type = type(layer).__name__
    layer_name = layer.name
    if layer_type == "Dense":
        weight_dict[layer_name + "_kernel"], weight_dict[layer_name + "_bias"] = layer.get_weights()
        model_arch.append({
            "name": layer_name,
            "type": "Dense",
            "config": {
                "units": layer.units,
                "activation": layer.activation.__name__
            },
            "weights": [layer_name + "_kernel", layer_name + "_bias"]
        })
    elif layer_type == "Flatten":
        model_arch.append({
            "name": layer_name,
            "type": "Flatten",
            "config": {},
            "weights": []
        })

with open("model/fashion_mnist.json", "w") as f:
    json.dump(model_arch, f, indent=4)
np.savez("model/fashion_mnist.npz", **weight_dict)

print("✅ 模型轉換完成：JSON + NPZ 儲存成功")
