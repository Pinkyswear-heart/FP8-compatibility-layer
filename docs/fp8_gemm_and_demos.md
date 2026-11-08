# FP8 LUT 解码 + 融合反量化 GEMM demo

在 RTX 2080 Ti (sm_75，无原生 FP8) 上用 FP16 tensor core 模拟 FP8 GEMM。

```
tests\build_fp8_host.bat        # 只编译宿主测试程序（cl）；kernel 运行时由 NVRTC 编译
tests\fp8_gemm.exe              # 6 个配置对照 double 参考，全部 PASS（在仓库根目录运行）
python -I tests/verify_lut.py . # LUT 与 torch.float8_e4m3fn / e5m2 逐位对比
```

| 方案步骤 | 位置 |
|---|---|
| 1. 格式/缩放配置 | `fp8.h` 的 `Fp8Format`、`Case`（E4M3FN/E5M2、block32 或 row/col、FP32 或 UE8M0 scale）——全部是可切换参数，**没有假定** DLSS5 泄露版的实际取值，需你从权重文件/头信息确认后填入 |
| 2. LUT 解码 | `fp8_build_lut`：按位段解码，保留 ±0、NaN；E4M3FN 与 E5M2 各一张表；kernel 里 LUT 拷到 shared memory（constant memory 在 warp 发散访问时会串行化） |
| 3. 融合反量化 | `fp8_gemm_kernel.cu`：FP8 tile → LUT → FP16 smem → WMMA → FP32 累加，不生成完整 FP16 权重；block-K 缩放在 K 循环内逐块乘 `sa*sb`，row/col 缩放在 epilogue 做；UE8M0 用 `ldexpf` 精确还原 2 的幂 |
| 4. 精度 | 只量化 GEMM 输入，输出 FP32（+bias），后续 Softmax/LayerNorm 不在 demo 内 |
| 5. Linear 优先 | 已实现 Linear/GEMM；Conv、Attention 未做 |

编码器 `fp8_encode`：最近偶数舍入 + 饱和（不产生 inf），NaN 输入 → NaN 码；未实现随机舍入。

限制：M/N/K 须为 32 的倍数；kernel 是正确性优先的朴素实现（32×32 tile），未做性能优化；A 与 B 可使用不同 FP8 格式。
构建说明：已装 MSVC 比 CUDA 12.9 支持的版本新，nvcc 的 cudafe++ 会崩溃，所以改用 NVRTC 运行时编译 kernel。

---

# 自建 DLSS 风格时域超分 demo（FP8 推理）

```
python -I demos/tiny_dlss.py --steps 800   # 约 2 分钟训练 + 评测，输出 dlss_out.png
```

- `demos/tiny_dlss.py`：程序化场景（背景平移 + 独立运动的圆，抖动采样的 64×64 低分辨率帧；运动向量直接取程序化场景真值（oracle），不是估计出来的，属演示级）→ 2× 超分到 128×128。输入为抖动补偿后的上采样当前帧 + 按运动向量重投影的历史帧，3×3 取邻域后经 3 层逐像素 Linear 预测残差；历史帧使用网络自己的上一帧输出（递归，同 DLSS 思路）。
- `fp8_cuda.py`：用 ctypes + NVRTC 把 `fp8_gemm_kernel.cu` 接入 PyTorch。三层 Linear 走融合 FP8 GEMM：权重与激活都是 FP8（W8A8），激活按行、每 32 个 K 动态量化，FP32 累加，层间 ReLU 与激活保持 FP32。
- 运行结果（第 8 帧，8 个序列平均，对比 3×3 超采样真值）：

| 方案 | PSNR |
|---|---|
| 双三次（无网络） | 21.94 dB |
| 网络 FP32 | 24.76 dB |
| 网络 FP8 E4M3 + FP32 scale | 24.21 dB |
| 网络 FP8 E4M3 + UE8M0 scale | 24.59 dB |
| 网络 FP8 E5M2 + FP32 scale | 23.80 dB |

- 融合 kernel 与 float64 反量化参考的自检误差在 1e-6 量级（6 种配置最差 1.08e-6，rowcol UE8M0）。
- 说明：这是在合成场景上训练的玩具网络，不是 NVIDIA DLSS 或任何社区模型的复刻；UE8M0 略优于 FP32 scale 属于该小样本下的结果，不具普遍性。

## 训练场景：`train/`

在同一个小 Transformer（4 层，d=128，16 位数字排序任务）上用 5 种精度方案训练 3000 步，对比 FP32 基线（模拟前）。种子、初始化、数据顺序和超参数都一样，唯一不同的是 Linear 层 GEMM 的精度。

```
python -I tests/test_fp8_sim.py      # 量化原语自检（从仓库根目录运行）
cd train
python train_compare.py --steps 3000 # 在 2080 Ti 上约 25 分钟 -> results.json
python make_report.py                # -> report.html（交互图表 + 表格）
```

- `fp8_sim.py`：`FP8Linear` 在训练时量化每个 Linear 的三次 GEMM 的输入：前向 X·Wᵀ，反向 dY·W 和 dYᵀ·X。量化用 torch 转换，做最近偶数舍入并饱和；`test_fp8_sim.py` 验证了它与 CUDA demo 的编码器在有限值范围内逐位一致（inf 的处理不同：torch cast 得 ±max_finite，`fp8.h` 编码器修复前得 ±0，现已统一为饱和到 ±max_finite；NaN 同为 NaN 码）。解码使用 CUDA demo 导出的 `lut_*.bin`。乘积以 FP32 累加。LayerNorm、Softmax、残差、注意力里的矩阵乘、主权重和 AdamW 状态都保持 FP32。
- block 模式沿 GEMM 的收缩维每 32 个元素取一个 scale，所以 W、X、dY 在前向和反向会按各自的布局分别量化。

| 方案 | val loss | 序列准确率 | 梯度下溢 |
|---|---|---|---|
| FP32 基线 | 0.0004 | 99.80% | — |
| FP8 per-tensor E4M3 / E5M2 | 0.0005 | 99.85% | 0.19% |
| FP8 block32 UE8M0 E4M3 / E5M2 | 0.0009 | 99.76% | 0.00% |
| FP8 per-tensor E4M3 / E4M3 | 0.0007 | 99.85% | 14.21% |
| FP8 无缩放 E4M3 / E5M2 | 1.4922 | 0.00% | 99.66% |

结论：只要做了缩放，FP8 方案与 FP32 的差距都在噪声范围内。不缩放时，大部分梯度低于 E5M2 的最小次正规数，被量化成 0，训练停滞。反向也用 E4M3 时，梯度下溢达到 14%，但这个小模型上仍能收敛。这个模型很小、任务也简单，不能推断大模型同样没有差距。这只是数值模拟，不节省显存，也不提速：带缩放的 FP8 方案每步约 110–150 ms（无缩放 recipe 约 54–67 ms，因为不做缩放 pass），FP32 为 22 ms。

---

## 项目结构（核心 / 测试分离）

| 目录 / 文件 | 类别 | 内容 |
|---|---|---|
| `fp8.h`、`fp8_gemm_kernel.cu` | 核心（C++） | FP8 格式/编码器/LUT 与融合反量化 GEMM kernel（纯 CUDA，可被 demo 与 tfp8 复用） |
| `fp8_cuda.py` | 核心（Python 桥） | ctypes + NVRTC 把 `fp8_gemm_kernel.cu` 接入 PyTorch（DLSS demo 的最小绑定） |
| `tfp8/` | 核心（库） | W8A16 运行时：`formats`/`quantize`/`linear`/`planner`/`_ext`/`csrc`/`examples` |
| `train/` | 核心（训练） | `fp8_sim.py` 训练量化模拟、`train_compare.py`、`make_report.py` |
| `tests/` | 测试与基准 | 数值门禁、GPU/CPU 基准、构建脚本、宿主测试驱动、独立 LUT 校验（详见 `tests/README.md`） |
| `demos/` | 演示 | `tiny_dlss.py`：FP8 推理的 DLSS 风格时域超分 demo |
| `.verify/` | 验证产物 | 独立复核报告、基准 JSON/日志（历史文档保留其写作时的旧路径） |

约定：核心库不依赖 `tests/`；测试脚本通过 `sys.path` 引导在仓库根目录导入 `tfp8`，基准输出仍写入 `.verify/bench/`。
