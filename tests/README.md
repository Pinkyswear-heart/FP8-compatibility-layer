# tests/ — 测试、基准与验证（与核心代码分离）

本目录集中存放原来散落在仓库根目录、`tfp8/tests/`、`train/` 里的测试、基准、构建与独立校验代码。
核心实现仍在原处：C++ 核心 `fp8.h` / `fp8_gemm_kernel.cu` 在仓库根目录，Python 运行库在 `tfp8/`
（含 `examples/`），训练模拟在 `train/`，演示在 `demos/`。核心代码不依赖本目录。

所有命令都在**仓库根目录**运行（基准把 JSON 写入 `.verify/bench/...`）。

## 内容与分类

| 文件 | 类别 | 说明 |
|---|---|---|
| `test_tfp8.py` | 数值门禁 | tfp8 的 19 项门禁（pytest 兼容）；覆盖 GEMV / WMMA GEMM / chunked 三条路径 |
| `test_fp8_sim.py` | 数值自检 | `train/fp8_sim.py` 的量化原语检查（编码器逐位一致、block 缩放、反向公式） |
| `verify_lut.py` | 独立校验 | 把 C++ demo 导出的 `lut_*.bin` 与 `torch.float8_e4m3fn/e5m2` 逐位对比 |
| `bench_transcode.py` | GPU 基准 | FP8 ↔ FP16 转译吞吐（带宽 / 解码 / 编码 / GEMV / GEMM / 地板），约 2 min |
| `bench_cpu_transcode.py` | CPU 基准 | 纯 CPU 转译吞吐（不初始化 CUDA），约 23 s |
| `benchmark.py` | 层基准 | 单层 4096×4096 各路径（BF16 / FP16 / 完整反量化 / fused / chunked）对比 |
| `bench_rows.py` | 微基准 | 按纯 kernel 时间比较 `GEMV_RADIUS` 变体 |
| `perf_decode.py` | 微基准 | M=1..8 decode 的 interleaved A/B：tfp8 GEMV vs 缓存 FP16 cuBLAS |
| `build_fp8_host.bat` | 构建 | 编译宿主测试程序（`cl`）；kernel 由 NVRTC 在运行时编译 |
| `fp8_gemm_host.cpp` | 宿主测试驱动 | 6 个配置对照 double 参考；同时导出 LUT 文件 |
| `fp8_gemm.exe` | 构建产物 | 宿主测试程序可执行文件（由上面两者生成） |
| `__init__.py` | 包标记 | 允许 `python -m tests.test_tfp8` 形式运行 |

## 常用命令

```bash
# 数值门禁 / 自检
python -I tests/test_tfp8.py
python -I tests/test_fp8_sim.py

# C++ 核心：重编译宿主测试驱动 + LUT 校验
tests\build_fp8_host.bat
tests\fp8_gemm.exe
python -I tests/verify_lut.py .

# 性能基准（GPU / CPU）
python -I tests/bench_transcode.py
CUDA_VISIBLE_DEVICES="" python -I tests/bench_cpu_transcode.py
python -I tests/benchmark.py
python -I tests/bench_rows.py
python -I tests/perf_decode.py
```

## 约定

- 从仓库根目录运行；`bench_transcode.py` / `bench_cpu_transcode.py` 用脚本位置推导仓库根目录，
  输出仍落在 `.verify/bench/`。
- 测试脚本在导入核心包之前把仓库根目录插入 `sys.path`（`test_fp8_sim.py` 还会加入 `train/`），
  因此 `python -I tests/xxx.py` 在任意 cwd 都能找到 `tfp8` / `fp8_sim`，同时 `-I` 仍可隔离外部环境变量。
- 历史验证报告（`.verify/`）保留写作当时的旧路径，不再随本次分类改动。
