import pickle
from pathlib import Path
from typing import List

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from animated_models import AnalyticFrame, ExpandFrame, FinalFrame, Frame
from manim import *  # pyright: ignore[reportWildcardImportFromLibrary]
from manim.utils.color import rgb_to_color
from plot_utils import MAP_PASSABLE_NEW

# 设置方形视频
config.pixel_width = 1080
config.pixel_height = 1080
config.frame_width = 50
config.frame_height = 50

# 验证是否有白边
config.background_color = GREEN


class PathPlanningScene(MovingCameraScene):
    def construct(self):
        # *加载障碍物地图
        map_data = np.where(np.flipud(MAP_PASSABLE_NEW), 0, 255).astype(np.uint8)
        map_img = ImageMobject(np.stack([map_data] * 3, axis=-1))
        map_img.width = config.frame_width  # 填满空间(wh只设置一次即可，不然会翻转)
        map_img.set_resample(False)  # 禁止缩放插值
        self.add(map_img)

        # *放大到局部
        focus_region = (27.5, 38.5, 15)  # x, y, size (示例)
        cx, cy, size = focus_region
        cx = min(50 - size / 2, max(cx, size / 2))
        cy = min(50 - size / 2, max(cy, size / 2))
        self.play(
            self.camera.frame.animate.set(width=size).move_to(np.array([cx - 25, cy - 25, 0])),  # type: ignore
            run_time=1,
            rate_func=smooth,
        )

        # *加载帧记录
        with open(Path(__file__).parent / "frames.pkl", "rb") as f:
            frames: List[Frame] = pickle.load(f)
            print(f"load successfully: {len(frames)} frames")
        costs = [f.total_cost for f in frames if isinstance(f, ExpandFrame)]
        norm = mcolors.Normalize(vmin=min(costs), vmax=max(costs))
        cmap = cm.get_cmap("plasma")  # 可换成 viridis, inferno, turbo 等

        # *绘制帧
        skip = 5  # 路径点跳略显示
        batch_size = 7
        batch_paths: list[VMobject] = []
        for i, frame in enumerate(frames):
            match frame:
                case ExpandFrame(node=node, total_cost=cost):
                    points = [np.array([x - 25, y - 25, 0]) for x, y in zip(node.x_list, node.y_list)]
                    rgba = cmap(norm(cost))

                    traj = VMobject(stroke_color=rgb_to_color(rgba[:3]), stroke_width=4)
                    traj.set_points_as_corners(points[::skip])
                    batch_paths.append(traj)
                    if len(batch_paths) >= batch_size:
                        self.play(
                            AnimationGroup(
                                *[Create(p) for p in batch_paths],
                                lag_ratio=0.0,  # 同时播放
                                run_time=0.2,  # 控制总时长
                                rate_func=linear,
                            )
                        )
                        batch_paths.clear()  # 清空缓冲区

                case AnalyticFrame(path=path, success=success):
                    points = [np.array([x - 25, y - 25, 0]) for x, y in zip(path.x, path.y)]
                    line_color, stroke_opacity = (YELLOW, 1.0) if success else (RED, 0.5)

                    traj = VMobject(stroke_color=line_color, stroke_width=4, stroke_opacity=stroke_opacity)
                    traj.set_points_as_corners(points[::skip])
                    self.add(traj)  # 直接加入场景

                    # 不成功尝试渐隐
                    if not success:
                        traj.add_updater(self.get_fade_updater(0.5))

                case FinalFrame(path=path):
                    if batch_paths:
                        # 如果此时还有未播放的 batch，也要播放掉
                        self.play(
                            AnimationGroup(
                                *[Create(p) for p in batch_paths], lag_ratio=0.0, run_time=0.2, rate_func=linear
                            )
                        )
                        batch_paths.clear()
                    points = [np.array([x - 25, y - 25, 0]) for x, y in zip(path.x_list, path.y_list)]

                    final_path = VMobject(stroke_color=GREEN, stroke_width=8)
                    final_path.set_points_as_corners(points[::skip])
                    final_path.set_stroke(opacity=1.0)
                    self.play(Create(final_path), run_time=2.5, rate_func=smooth)

                    self.wait(1)

    def get_fade_updater(self, fade_speed: float = 0.5):
        """返回一个 updater 函数，用于非阻塞渐隐"""
        speed = fade_speed

        def updater(mob: Mobject, dt: float):
            new_opacity = mob.get_stroke_opacity() - speed * dt
            if new_opacity <= 0:
                mob.set_stroke(opacity=0)
                mob.remove_updater(updater)  # 自动解绑，停止更新
            else:
                mob.set_stroke(opacity=new_opacity)

        return updater


if __name__ == "__main__":
    from plot_utils import *

    fig, ax = plt.subplots()
    plot_binary_map(MAP_PASSABLE_NEW, ax)
    plt_tight_show()
