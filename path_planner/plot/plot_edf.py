import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from plot_utils import *


def generate_png_arrays_from_arrays(Z_list: List[NDArray], cmaps: List[str], dpi=50, figsize=(5, 5)):
    """
    将二维数组渲染为 PNG 图片的 RGBA 数组（不写文件）
    """
    if len(Z_list) != len(cmaps):
        raise ValueError("两个输入列表长度不匹配")

    m, n = Z_list[0].shape
    for Z in Z_list:
        if Z.shape != (m, n):
            raise ValueError("所有二维数组尺寸必须相同")

    png_arrays = []

    for Z, cmap_name in zip(Z_list, cmaps):
        # 判断是否为 bool 类型
        if Z.dtype == bool:
            # 转 int 并映射为 0/1
            Z_int = Z.astype(int)
            # 使用黑白 colormap
            cmap = matplotlib.colormaps.get_cmap(cmap_name)
            norm = Normalize(vmin=0, vmax=1)
            rgba_img = cmap(norm(Z_int))
        else:
            # 连续值处理
            norm = Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
            rgba_img = cm.ScalarMappable(norm=norm, cmap=cmap_name).to_rgba(Z)

        # 创建 figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_axis_off()
        ax.imshow(rgba_img, origin="lower", aspect="equal")

        # 将 figure 渲染到 RGBA numpy 数组
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()  # 返回 memoryview
        image = np.asarray(buf)  # 转为 numpy 数组
        png_arrays.append(image)

        plt.close(fig)
    print(png_arrays[0].shape)

    return png_arrays


def stack_png_images_3d(pngs: List[NDArray], offsets: List[float], save_path="stacked_images.png", dpi=300):
    """
    将三个 PNG 图片在三维空间堆叠
    png_paths: list of three PNG file路径
    offsets: 各层垂直偏移
    """
    if len(pngs) != len(offsets):
        raise ValueError("两个输入列表长度不匹配")

    imgs = pngs
    # 假设三张图片大小一致
    m, n = imgs[0].shape[:2]

    fig = plt.figure(figsize=(6, 5))
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # 坐标网格
    X, Y = np.meshgrid(np.arange(n), np.arange(m))

    for img, offset in zip(imgs, offsets):
        # 如果 img 是 uint8，转成 float 0-1
        facecolors = img.astype(np.float32) / 255.0
        facecolors[..., 3] = 1.0  # alpha = 1
        # 将 PNG 映射到高度平面
        ax.plot_surface(
            X,
            Y,
            np.full((m, n), offset),
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

    # 调整视角
    ax.view_init(elev=25, azim=-60)
    ax.set_box_aspect((n, m, abs(offsets[-1]) + 20))

    # 去掉轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax._axis3don = False
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
        except Exception:
            pass

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    Z_dem = MAP_DEM
    Z_slope = np.load(NPY_ROOT / "map_slope_new.npy")
    Z_rough = np.load(NPY_ROOT / "map_rought_new.npy")
    Z_passable = MAP_PASSABLE_NEW

    # 生成三张 PNG 的 RGBA 数组
    png_arrays = generate_png_arrays_from_arrays(
        [Z_dem, Z_slope, Z_rough, Z_passable],
        ["terrain", "viridis", "magma", "binary"],
    )
    stack_png_images_3d(pngs=png_arrays, offsets=[0, -100, -200, -300])
