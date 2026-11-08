# TFP8 — Turing FP8 兼容运行时（W8A16 demo）

> **注意：只针对 NVIDIA GeForce RTX 2080 Ti，其他型号尚不明确。**

按 `tfp8-llm-image-solution.md` 实现的 W8A16 演示：FP8 权重常驻显存（uint8 + f32 scale），
在 tile 内解码成 FP16 后走 Tensor Core / FP32 FMA，**不生成完整 FP16 权重**。

```
python -m tfp8                              # 自检 + 环境报告
python -I tests/test_tfp8.py                # 14 项数值门禁
python -I tests/benchmark.py                # 单层 4096x4096 各路径对比
python -m tfp8.examples.demo_llm --size tiny
python -m tfp8.examples.demo_image --size 64 --dim 512 --depth 8
python -I tests/perf_decode.py              # M=1..8 decode 专项：tfp8 vs 缓存 FP16
python -I tests/bench_transcode.py          # FP8 转译算力基准（GPU）
python -I tests/bench_cpu_transcode.py      # FP8 转译算力基准（CPU，纯 CPU）
```

> 测试/基准/构建脚本已从包内移到仓库顶层 `tests/`（分类说明见 `tests/README.md`）；`tfp8/` 只保留运行库与示例。以上命令在仓库根目录运行。

## 目录

| 文件 | 对应方案章节 | 内容 |
|---|---|---|
| `formats.py` | §4.1 / §5.1 | E4M3FN / E5M2 位段解码、256 项 LUT、RNE+饱和编码 |
| `quantize.py` | §4.2–4.3 | 对称量化，per-channel / per-K-group(128) scale，可 2 的幂 |
| `csrc/tfp8_kernels.cu` | §5.2 / §5.4 | 融合 GEMV（M≤8）、融合 64×64 WMMA GEMM（寄存器预取双缓冲）、dequant、chunked 融合 epilogue、激活行缩放 |
| `linear.py` | §5.2 / §9 | `FP8Linear`：按 M 分流，运行时 NaN 回退，FP8 checkpoint 入口 |
| `planner.py` | §6 / §7.3 / §8.2 | 逐模块选路、`precision_owner`、tied weight 检测 |
| `_ext.py` | §5.2 | NVRTC 编译 + ctypes `cuLaunchKernel`（cached-arg 快路径） |
| `examples/demo_llm.py` | §7 | Llama 结构 LLM：显存、logits 一致性、prefill/decode |
| `examples/demo_image.py` | §8 | DiT：显存、PSNR/SSIM、每步耗时 |

分路径延迟表与转译基准已移到 `tests/`（`benchmark.py`、`bench_transcode.py` 等）。

内核用 `torch.cuda._utils` 的 NVRTC 在运行时编译（不走 nvcc，规避 Windows 上 MSVC 版本问题），
并由 `_ext.py` 用 `cuLaunchKernel` 直接启动（torch 的 `_CudaKernel.__call__` 每次调用额外 ~0.3 ms，
对 DiT 这种几百个小层是主要开销）。

## 已验证

19 项门禁全部通过（`python -I tests/test_tfp8.py`）：

- E4M3FN / E5M2 **全部 256 个编码**与 `torch.float8_*` 逐位一致，含 ±0、次正规数、NaN、inf
- 编码器 RNE 与 torch 一致，超范围饱和到 ±max_finite（不产生 inf）
- GEMV / GEMM / chunked 三条路径，M ∈ {1,2,3,7,8,9,15,16,31,32,127,128,255,256,300}
- N 尾 tile（N=200）、per-channel 与 per-K-group(128) scale、E4M3FN/E5M2、FP16/BF16 输出
- 分组 scale 逐块应用（两组相差 1000 倍时仍与 FP64 参考一致）
- 空张量、非连续输入、tied lm_head 不被量化、重复转换幂等、NaN 自动回退
- chunked 路径按层的 fmt 解码（e5m2 不再被当作 e4m3fn），融合 epilogue 覆盖多 slice 列窗口 / 奇数 N / f16·bf16·f32 输出
- fused GEMM 把组 scale 折进解码后的权重：2 的幂 scale 精确，非 2 的幂 scale 误差在 fp16 权重精度内（组 scale 跨度到 2^-20）；fused 支持 float32 输出
- GEMV 预构建参数块快路径与通用路径逐位一致，buffer 替换 / 设备往返 / out_dtype 变化后不留失效指针；f16/bf16/f32 输入的 GEMV 都对照 FP64（fp32 输入曾只暂存一半 x）
- 双缓冲 WMMA GEMM 的流水边界（K=32 单步、每步都是组边界、M/N 尾）；prep kernel 的行 amax 是块级最大值（离群值不溢出）
- `_launch` 参数槽按完整类型签名隔离：同名同参数个数、不同 float/int/tensor 组合不会复用错误 holder（回归门禁 `test_launch_slot_kinds_are_isolated`）

## 实测（RTX 2080 Ti 11 GB，torch 2.14.1 + CUDA 13.2）

单层 4096×4096，中位数（`python -I tests/benchmark.py`）：

| M | BF16 | FP16 | 完整反量化 | tfp8 fused | tfp8 chunked |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.111 | 0.140 | 1.481 | 0.184 | 0.193 |
| 8 | 0.355 | 0.111 | 1.487 | **0.186** | 0.187 |
| 32 | 0.361 | 0.082 | 1.500 | 0.272 | 0.273 |
| 128 | 0.591 | 0.106 | 1.521 | 0.329 | 0.412 |
| 2048 | 8.791 | 0.766 | 2.092 | 4.304 | **1.528** |

LLM（`--size tiny`，2 层 dim 1024）：权重 59.0 → 38.6 MB（**−34.6%**），logits cosine 0.9973、
top-1 一致率 82.8%，decode 0.92×、prefill 0.94×（相对 BF16）。
LLM（`--size 1b`，16 层）：权重 1501 → 846 MB（**−43.6%**），decode ≈1.0×、prefill ≈1.0×。

图像 DiT（64px，dim 512，8 层）：权重 76.4 → 52.6 MB（**−31.1%**），
PSNR 51.0 dB / SSIM 0.9951，每步 22.9 → 41.2 ms（**慢 1.8×**）。

### Decode 专项：tfp8 vs 预先缓存的 FP16 权重（A/B 交替测量）

`python -I tests/perf_decode.py`。这张卡与其他进程共用、单次计时抖动 2–4×，所以每个候选
在同一轮 A/B 交替中测量后取中位数；非交替的计时已被证明会给出相反结论。

小层启动开销优化前后的对比（M=1）：

| 层 (N×K) | 权重 | fp16 | 优化前 | 优化后 | 优化前 | 优化后 |
|---|---:|---:|---:|---:|---:|---:|
| qkv  12288×4096 | 48 MB | 0.188 | 0.270 | 0.212 | 0.78× | **0.88×** |
| gate 11008×4096 | 43 MB | 0.174 | 0.196 | 0.168 | 0.94× | **1.04×** |
| down  4096×11008 | 43 MB | 0.171 | 0.227 | 0.190 | 0.83× | **0.90×** |
| o      4096×4096 | 16 MB | 0.065 | 0.235 | 0.095 | 0.44× | **0.68×** |
| 1b     5632×2048 | 11 MB | 0.074 | 0.200 | 0.108 | 0.41× | **0.69×** |
| dit    4608×1152 | 5 MB  | 0.079 | 0.210 | 0.116 | 0.41× | **0.68×** |

优化内容（`_ext.py` / `linear.py`）：

1. 预分配并复用 ctypes 参数块，只改写 `.value`——原先每次调用重建约 12 个
   `ctypes` 对象再组装数组，实测 33 µs，现在 2.6 µs。
2. 用 `torch._C._cuda_getCurrentRawStream` 取流（0.3 µs），替代 `torch.cuda.current_stream()`（9.1 µs）。
3. `cuLaunchKernel` 只绑定一次 argtypes；kernel handle 按 (device, name) 缓存。
4. `FP8Linear` 不再每次调用重新探测 `_ext.available()` 和查 LUT 表（合计 ~9 µs），
   LUT 按设备缓存并按需重解析（层先建在 CPU 再 `.cuda()` 也正确）。
5. 单 block 内核的启动下限：82.5 → 37.1 µs。

结果：小层从 0.41× 提到 0.68–0.69×，大层从 0.78–0.94× 提到 0.88×/1.04×。

仍然存在的限制（**没有解决**）：

- 单次 `cuLaunchKernel` 经 ctypes 本身就要 ~28 µs（Windows/WDDM 路径），这是纯 Python
  侧无法绕开的地板；而 M=1 的 5 MB 层理论计算时间只有 ~17 µs。
- 小层（≤16 MB）仍为 0.68–0.69×：启动占了一半时间，且 GEMV 每线程的串行 FMA 链在
  小 K 上跑不满。
- **大层能到 1.04×，但没有稳定的 1.0× 以上收益。** 结论不变：W8A16 在 Turing 上省的
  是显存和带宽占用，不是时间。
- 端到端 1b 模型（16 层，A/B 交替）decode **1.02×**、prefill 1.06×；单次测量的
  0.81× 是噪声（bf16 单步在 83–126 ms 间抖动）。

## 与方案预期不符的地方

方案 §2 预期「LLM Decode（M=1..8）更可能从 FP8 权重收益」。实测**没有收益**：

- 相对 **BF16** 的 1.1–5.6× 提速是真实的，但那是因为 Turing 的 BF16 没有 Tensor Core——
  对标的应该是 `tbf16`（预先缓存 FP16 权重），而不是 BF16。对 FP16 是 0.41–0.94×。
- 端到端 LLM（1B，16 层）：decode 相对 BF16 1.11×、prefill 0.97×；权重 −43.6%。
  加速主要来自 BF16 基线慢，不是 tfp8 快。

结论：**W8A16 的价值是显存和带宽，不是算力**。在 11 GB 卡上，43.6%（1B）/ 更大模型上约 50%
的权重下降才是这个 demo 真正交付的东西。

图像侧同样符合方案 §8.1 的预期：M 很大，GEMM 进入计算受限区间，融合 GEMM（4.3 ms）
和 chunked（1.5 ms）都**慢于** cuBLAS FP16（0.77 ms），因为它们是教学级 tile 实现。
DiT 的收益只在显存（−31%），代价是 2× 延迟。

## 这次性能测试中发现并修复的 bug

`GEMV` 内部按 `GEMV_BK=512` 切 K，但仅在 K 是 512 的倍数时才有边界保护。**K=1152（DiT 的
真实层宽）时 GEMV 结果完全错误（相对误差 94×）**：尾部 chunk 的 lane 越界读到下一行。
之前所有测试的 K 都是 512 的倍数，所以没有暴露——而 DiT demo 的 K=1152 恰好全程走
`chunked` 路径，绕开了 GEMV，我也没发现。

已修复（尾部 `rem = min(GEMV_BK, K - k0)`，不在范围内的 lane 直接跳过），并补了回归门禁
`test_kernel_ragged_k_tail`（K = 528/1040/1152/1168/1184 × M = 1/8/16/64）。当时 14 项门禁全通过（现为 19 项）。

## 已知限制

- 只做 W8A16。W8A8、QK FP8、FP8 KV Cache 按方案 §1 放在后续阶段（`FP8Config` 会显式报
  `NotImplementedError`，不静默降级）。
- 融合 GEMM 是 64×64 tile + 寄存器预取双缓冲（~21 TF per-channel / ~19 TF g128，组 scale 折进 B tile、无 spill），无跨 block K 切分，仍远低于 cuBLAS（~76 TF）；M≥256 默认走 chunked。
- large-M 默认走 `chunked`（反量化到固定 32 MB workspace + cuBLAS）；`fused` 可切换。
- 兼容模式（`FP8Linear.from_fp8_checkpoint`）只校验 dtype/格式和 scale 形状，未对接社区 checkpoint
  的真实 metadata 布局（方案阶段 3）。
- DiT/LLM 均为随机权重的小模型，验证的是数值与显存，不是生成质量。
