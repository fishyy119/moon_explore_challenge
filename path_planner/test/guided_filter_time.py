import cv2
import numpy as np
from plot.plot_utils import MAP_DEM
from scipy.ndimage import zoom


def main():
    dem = MAP_DEM.astype(np.float32)
    dem_large = zoom(dem, zoom=2, order=3)

    elevation_filtered = cv2.ximgproc.guidedFilter(dem_large, dem_large, radius=5, eps=1e-2)
    elevation_blur = cv2.bilateralFilter(dem_large, d=5, sigmaColor=0.05, sigmaSpace=3)


if __name__ == "__main__":
    import datetime
    from pathlib import Path as fPath

    from line_profiler import LineProfiler

    lp = LineProfiler()

    lp_wrapper = lp(main)
    lp_wrapper()
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    name = fPath(__file__).stem
    short_name = "_".join(name.split("_")[:2])  # 取前两个单词组合
    profile_filename = f"profile_{short_name}_{timestamp}.txt"
    with open(profile_filename, "w", encoding="utf-8") as f:
        lp.print_stats(sort=True, stream=f)
