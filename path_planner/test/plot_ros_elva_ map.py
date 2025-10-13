import math

import matplotlib.pyplot as plt
import numpy as np
from conftest import *


def read_elevation_map_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 去除换行符
    lines = [line.strip() for line in lines if line.strip()]

    # 解析头部信息
    resolution = None
    map_size = None
    data_dims = None
    data_start_idx = None

    for i, line in enumerate(lines):
        if line.startswith("# Resolution:"):
            resolution = float(line.split(":")[1].strip())
        elif line.startswith("# Map size:"):
            parts = line.split(":")[1].split("x")
            map_size = tuple(float(x.strip()) for x in parts)
        elif line.startswith("# Data dimensions:"):
            parts = line.split(":")[1].split("x")
            data_dims = tuple(int(x.strip()) for x in parts)
        elif line.startswith("# Data:"):
            data_start_idx = i + 1
            break

    if data_start_idx is None:
        raise ValueError("No data section found in the file.")

    # 读取数据（空格或换行分隔）
    data_str = " ".join(lines[data_start_idx:])
    data_list = data_str.split()

    # 转换为 float，处理 NaN
    data = np.array([float(x) if x != "NaN" else np.nan for x in data_list])

    # 按照数据维度 reshape
    h, w = data_dims[1], data_dims[0]  # 注意：94x81 -> width x height
    data = data.reshape((h, w))

    return data, resolution, map_size


def plot_elevation_map(data, resolution, map_size):
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap="terrain", origin="lower", interpolation="nearest")
    plt.colorbar(label="Elevation (m)")
    plt.title("Elevation Map")
    plt.xlabel(f"X (cells, {resolution} m each)")
    plt.ylabel(f"Y (cells, {resolution} m each)")
    plt.show()


# ==== 使用 ====
if __name__ == "__main__":
    file_path = Path(ROOT_DIR / "resource/elevation_map.txt")
    data, res, size = read_elevation_map_txt(file_path)
    print(f"Resolution: {res}, Size: {size}, Data shape: {data.shape}")
    plot_elevation_map(data, res, size)
