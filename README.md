# FP8 Compatibility Layer

在**没有原生 FP8 硬件**的 NVIDIA Turing 显卡（RTX 2080 Ti，sm_75）上运行 FP8 模型的兼容层。

FP8 权重以 `uint8` 编码 + FP32 scale 的形式常驻显存，在 kernel 的 tile 内用查表/位运算解码成 FP16，再交给 FP16 Tensor Core 做 FP32 累加。**全程不生成完整的 FP16 权重副本。**

> 这个项目省的是**显存和带宽**，不是算力。Turing 没有 FP8 张量核，它不会让你得到“原生 FP8 速度”。具体数字见下方[实测结果](#实测结果rtx-2080-ti)。

## 特性

- **两种 FP8 格式**：E4M3FN、E5M2。256 个编码全部与 `torch.float8_*` 逐位一致，包括 ±0、次正规数、NaN、inf。
- **编码器**：最近偶数舍入（RNE），超出范围时饱和到 ±max_finite，不会产生 inf。
- **缩放粒度**：per-tensor、per-channel、per-K-group（32 / 128），scale 可以是 FP32 或 UE8M0（2 的幂）。
- **融合反量化 kernel**：
  - GEMV（M ≤ 8，用于 decode）
  - 64×64 WMMA GEMM，带寄存器预取双缓冲
  - chunked 路径：先反量化到固定 32 MB workspace，再调用 cuBLAS，最后用一个融合 epilogue 收尾
- **`tfp8` 运行库**：可以直接替换 `nn.Linear` 的 `FP8Linear`。它按 M 自动选择计算路径，输出出现 NaN/Inf 时自动回退，还会检测 tied weight，并逐模块规划哪些层量化、哪些保持原精度。
- **不依赖 nvcc**：kernel 由 NVRTC 在运行时编译，再用 ctypes 调 `cuLaunchKernel` 启动。这样可以绕开 Windows 上 MSVC 版本比 CUDA 新导致 nvcc 崩溃的问题。
- **FP8 训练数值模拟**：在小 Transformer 上对比 5 种 FP8 训练方案和 FP32 基线。

## 环境要求

- NVIDIA GPU，在 RTX 2080 Ti / sm_75 上开发和测试（理论上其他 sm_70+ 显卡也能运行，但未验证）
- Python 3.10+、PyTorch（CUDA 版，需要带 `torch.float8_*` dtype 和 NVRTC）
- 只编译 C++ 宿主测试程序时才需要 MSVC（`cl`）

```bash
pip install -r requirements.txt
```

## 快速开始

所有命令都在仓库根目录运行。

```bash
python -m tfp8                    # 自检 + 环境报告
python -I tests/test_tfp8.py      # 19 项数值门禁
```

### 在自己的模型上使用

```python
import torch
from tfp8 import FP8Config, prepare_model, report, weight_bytes

model = MyModel().half().cuda()

cfg = FP8Config(fmt="e4m3fn", profile="llm_safe", weight_group=None)  # None = per-channel；128 = 按 K 分组
model = prepare_model(model, domain="llm", config=cfg)                # 图像模型用 domain="image"

report(model)                 # 打印每个模块的处理决定（量化 / 保持原精度）及原因
print(weight_bytes(model))
```

也可以单独替换某一层：

```python
from tfp8 import FP8Linear

fp8_layer = FP8Linear.from_linear(linear, fmt="e4m3fn", group_size=128)
```

### 示例

```bash
python -m tfp8.examples.demo_llm --size tiny                       # Llama 结构 LLM：显存、logits 一致性、prefill/decode
python -m tfp8.examples.demo_image --size 64 --dim 512 --depth 8   # DiT 图像生成：显存、PSNR/SSIM、每步耗时
python -I demos/tiny_dlss.py --steps 800                           # 自建时域超分小网络，用 W8A8 融合 GEMM 推理
```

## 项目结构

| 路径 | 内容 |
|---|---|
| `tfp8/` | W8A16 运行库：`formats`（编解码 / LUT）、`quantize`、`linear`（`FP8Linear`）、`planner`、`_ext`（NVRTC + 启动）、`csrc/tfp8_kernels.cu`、`examples/` |
| `fp8.h`、`fp8_gemm_kernel.cu` | 独立的 C++/CUDA 核心：FP8 格式、编码器、LUT 和融合反量化 GEMM（W8A8，支持 block32 与 row/col 两种缩放） |
| `fp8_cuda.py` | 把 `fp8_gemm_kernel.cu` 接入 PyTorch 的最小 ctypes 绑定 |
| `lut_e4m3fn.bin`、`lut_e5m2.bin` | 宿主测试程序导出的 256 项解码表 |
| `train/` | FP8 训练数值模拟（`fp8_sim.py`）、对比实验（`train_compare.py`）、报告（`report.html`） |
| `demos/` | `tiny_dlss.py`：时域超分 demo |
| `tests/` | 数值门禁、GPU/CPU 基准、C++ 宿主测试驱动，详见 [tests/README.md](tests/README.md) |
| `docs/` | C++ 核心、超分 demo 和训练实验的详细说明 |

## 实测结果（RTX 2080 Ti）

**单层 4096×4096 的延迟**（单位 ms，取中位数，`python -I tests/benchmark.py`）：

| M | BF16 | FP16 | 完整反量化 | tfp8 fused | tfp8 chunked |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.111 | 0.140 | 1.481 | 0.184 | 0.193 |
| 8 | 0.355 | 0.111 | 1.487 | 0.186 | 0.187 |
| 128 | 0.591 | 0.106 | 1.521 | 0.329 | 0.412 |
| 2048 | 8.791 | 0.766 | 2.092 | 4.304 | 1.528 |

**端到端**（模型均为随机权重，验证的是数值和显存，不代表生成质量）：

| 模型 | 权重显存 | 精度 | 速度 |
|---|---|---|---|
| LLM 1B（16 层） | 1501 → 846 MB（**−43.6%**） | — | decode ≈1.0×，prefill ≈1.0× |
| LLM tiny（2 层，dim 1024） | 59.0 → 38.6 MB（−34.6%） | logits cosine 0.9973 | decode 0.92× |
| DiT 64px（8 层，dim 512） | 76.4 → 52.6 MB（−31.1%） | PSNR 51.0 dB / SSIM 0.9951 | 慢 1.8× |

**FP8 训练模拟**（4 层 Transformer，3000 步）：只要做了缩放，FP8 方案和 FP32 的差距都在噪声范围内。不做缩放时，99.66% 的梯度下溢为 0，训练完全停滞。

| 方案 | val loss | 序列准确率 |
|---|---|---|
| FP32 基线 | 0.0004 | 99.80% |
| FP8 per-tensor E4M3 / E5M2 | 0.0005 | 99.85% |
| FP8 block32 UE8M0 E4M3 / E5M2 | 0.0009 | 99.76% |
| FP8 无缩放 E4M3 / E5M2 | 1.4922 | 0.00% |

更多数据和分析：

- [tfp8/README.md](tfp8/README.md)：各计算路径的延迟、decode 专项 A/B 测试、已修复的 bug
- [tfp8/BENCH_TRANSCODE.md](tfp8/BENCH_TRANSCODE.md)：FP8 转译吞吐实测
- [docs/fp8_gemm_and_demos.md](docs/fp8_gemm_and_demos.md)：C++ 核心、超分 demo、训练实验

## 已知限制

- `tfp8` 目前只支持 **W8A16**。W8A8 激活、FP8 KV Cache 会显式报 `NotImplementedError`，不会静默降级。
- 融合 GEMM 是教学级 tile 实现（约 19–21 TFLOPS），远低于 cuBLAS FP16（约 76 TFLOPS）。M 大时默认走 chunked 路径。
- 在 Windows/WDDM 上，经 ctypes 启动一次 kernel 就要约 28 µs，小层的 decode 只能达到 FP16 的约 0.7×。
- `FP8Linear.from_fp8_checkpoint` 只校验 dtype 和 scale 形状，还没有对接任何社区 checkpoint 的真实 metadata 布局。
- 目前只在单张共享使用的 RTX 2080 Ti 上测过，绝对耗时会受外部负载和时钟状态影响，比值更有参考价值。

## 许可证

[MIT](LICENSE)
