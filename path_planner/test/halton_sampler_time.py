from scipy.stats import qmc


def main():
    sampler = qmc.Halton(d=2, scramble=False)
    # 多取一些以防止不落入区域内，此处采样开销不大，应该无需分批次采样
    halton_points = sampler.random(n=10)
    halton_points = sampler.random(n=100)
    halton_points = sampler.random(n=1000)
    halton_points = sampler.random(n=10000)
    halton_points = sampler.random(n=100000)


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
