# FP8 转译算力实测报告 — RTX 2080 Ti (Turing, sm_75)

> **注意：只针对 NVIDIA GeForce RTX 2080 Ti，其他型号尚不明确。**

> 对象：`tfp8`（Turing 上以 W8A16 模拟 FP8 的推理库）
> 机器：RTX 2080 Ti / Xeon E5-2666 v3 / Windows 10 / torch 2.14.1+cu132
> 数据：`.verify/bench/gpu/transcode_gpu_run1.json`、`run2.json`（11-07 两次全量）、`run3.json`~`run8.json`（11-08 复测 6 轮；run7/run8 为解锁状态复核轮）、`run9.json`/`run10.json`（11-08 14:2x 代码修复后复测）、`diag_dequant.json`（根因探针）
> 脚本：`tests/bench_transcode.py`（GPU）、`tests/bench_cpu_transcode.py`（CPU）、`.verify/bench/gpu/diag_dequant.py`（根因）
> 日期：2026-11-07（初测）/ 2026-11-08（复测 + 解锁状态复核 + 代码修复后复测，见 §12）

**一句话结论**：这张卡没有 FP8 硬件，所谓「转译 FP8 的算力」全部是软件位运算 + 显存带宽。
实测**解码 117–169 Gelem/s（352–507 GB/s）、编码 128 Gelem/s（387 GB/s）**，满 boost 下解码最快样本
与同 run copy 上限（506.8 GB/s）几乎相等 → 转译本身是**纯带宽受限、算术免费**。直接把转译塞进融合 GEMM 时，
每 64 行 M 重复解码所有权重，吞吐被锁死在 **12–15 TFLOPS**；生产 dequant kernel 因一个 grid 参数浪费了
**1.4–4.6×**（已实测修复方案，输出 bit-exact）。

> **2026-11-08 复测**（§12，共 6 轮 run3–run8）：整体比 11-07 再快 4–12% —— copy 380–389 GB/s、解码 121.8–126.5 Gelem/s、编码 134.8–138.7 Gelem/s、chunked 41.2–41.6 TF；结论与排名不变。**状态更正**：§12 原来写的「未解锁」有误 —— 这张卡的 FP16 tensor 半速锁自 11-02 起就已解除（证据与复核见 §12.2）；300 W 功耗墙仍是默认值，而转译负载只吃 ~123–133 W，所以解锁不改变本报告的转译数字。**2026-11-08 14:2x**：生产 dequant 的 grid 修复已由项目自身落地（`tfp8/_ext.py:183`），run9/run10 实测 qkv **351.6 µs / 384.8 GB/s（1.42×）**、DiT **46.0 µs（4.37×）**、1B **92.0 µs（2.67×）**，chunked 的 dequant 占比 11%→8%，详见 §12.3。**2026-11-08 晚（§12.4）**：chunked 的 epilogue 融合成单 kernel 后 M=2048 **1605.9 → 1281.9 µs（42.8 → 53.6 TF，cuBLAS 的 70%）**，A/B 三形状 1.29–2.02× 且 bit-exact；另修 3 个正确性 bug（chunked 忽略 e5m2、prep 行 amax 未做块归约、`_launch` 把 float 参数按 double 传导致 ACT_TARGET_EXP 恒为 0）；BM=128 试验**否证**了「融合 GEMM 受限于重复解码」的判断（§12.4.3）。**随后（§12.5）给融合 GEMM 加了寄存器预取双缓冲：交替 A/B 1.17–1.84×（24 点全部变快、全部 bit-exact），平台期 13 → 17–23 TF。** §12.6：去掉 g128 的组累加 spill（scale 折进解码后的 B），g128 再快 1.04–1.12×（≈19–19.6 TF），并修了 fused GEMM 的 float32 输出写成 bf16 的 bug。

---

## 0. 结论摘要（TL;DR）

| # | 问题 | 实测答案 |
|---|---|---|
| 1 | FP8→FP16 解码（读出 1 B/元素，写回 2 B/元素） | **117.2 Gelem/s = 351.6 GB/s**（两轮共享状态中位）；满 boost 独立复测 **124.2 中位 / 169.0 最优 Gelem/s**（373 / **507 GB/s**，与同 run copy 上限 506.8 相等）；e4m3fn / e5m2 / LUT / 加 scale 四种变体差 <2%（带宽瓶颈，解码算术免费） |
| 2 | FP16→FP8 编码（读 2 B，写 1 B） | **128.3 Gelem/s = 387 GB/s**（与 `torch` 原生 cast 持平，说明硬件/软件路径已同速） |
| 3 | 带 scale 的量化 `quantize_weight`（GPU） | 9.4–9.8 Gelem/s（≈19 GB/s），比裸 cast 慢 **13.6×**：amax 归约 + pow2 scale + 除法的多 pass 是主因 |
| 4 | 融合 GEMM（decode 融进 WMMA 主循环） | **12–15 TFLOPS，与 M 无关（M=128…2048 全平台期）**；折算有效解码率 95.7–117.9 Gelem/s，天花板 = 解码率 × 128 FLOP/权重 |
| 5 | 大 M 的 chunked 路径（先反量化再 cuBLAS） | **11-08 晚 epilogue 融合后：53.6 TF（1281.9 µs，cuBLAS 76.3 TF 的 70%），见 §12.4**。以下为原始数据 —— M=2048 生产实现：**38.96 TFLOPS**（fp16 cuBLAS 69.57 TF 的 56%），转译只占 **11.5%**，mm 达 cuBLAS 的 93%；同数学的精简实现独立复测 **53.8 TF** → 生产路径仍有 ~30% 非转译开销可省 |
| 6 | M=1 decode（GEMV）vs FP16 cuBLAS | **0.93–1.06×**（6 个真实形状全部打平）：FP8 省一半字节，但解码 ALU 吃掉一半带宽收益 |
| 7 | 生产 `tfp8_dequant_f16` 的转译吞吐 | 只有 254.5 GB/s（比微基准低 27%）；**根因是 grid 里 64 个 x-block 只有 1–2 个有活**，空 block 使耗时随行数增长（同元素不同行 1.46–1.70×，grid sweep 单调跨度 4.2×）。精确 grid 修复已于 11-08 12:47 落地（`tfp8/_ext.py:183`）：run9/run10 实测 **351.6 µs / 384.8 GB/s（1.42×）**、DiT **46.0 µs（4.37×）**、1B **92.0 µs（2.67×）**，输出 bit-exact，已与手写微基准持平 |
| 8 | CPU 对照（同机 10 线程） | 编码 0.587 Gelem/s、解码 1.216 Gelem/s → **比 GPU 慢 219× / 96×**；1B 权重离线量化 ≈1.71 s，7B ≈11.95 s |

**口径**：Gelem/s = 每秒 10⁹ 个元素；GB/s = 每秒 10⁹ 字节（十进制，非 GiB）。解码按 3 B/元素（1 B 读 + 2 B 写）计有效带宽，编码同样 3 B/元素。

---

## 1. 平台与环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti，sm_75，68 SM，11104 MB，L2 6 MB（torch 报告值；NVIDIA 规格为 5.5 MB） |
| 硬件 FP8 支持 | **无**。Turing 没有 FP8 张量核，"FP8" 在这里只是 `uint8` 位模式 + 软件解码 |
| Driver / CUDA | 621.01 / UMD 13.2（torch 2.14.1+cu132） |
| CPU | Intel Xeon E5-2666 v3 @2.90 GHz，10 物理核 / 20 逻辑线程，64 GB |
| OS / Python | Windows 10 19045 / CPython 3.13.16 |
| 可流带宽基准 | copy 342–362 GB/s，fill/read 405–422 GB/s（规格 616 GB/s 的 56–69%，共享卡降频） |
| 卡的状态 | **共享卡**：wallpaper_engine / ChatGPT / Edge 等常驻。全量 run 期间 nvidia-smi 快照：util 36–38%→76–82%，SM clock 1350–2010 MHz / 2205，功耗 76→126 W |
| 时钟状态（重要） | 空闲 570–630 MHz / ~30 W，满 boost 1770–2010 MHz / 190–281 W。**转译吞吐强依赖该状态**：本报告两轮 run 处于偏中窗口；独立复测先做 2 s clock_burn 再测，满 boost 下 copy 428.6 GB/s（最快 506.8）、解码最快 507 GB/s。主表用两轮中位（同状态可比），§9 给满 boost 复测值 |

**这张卡上「FP8 转译」到底在做什么**：解码 = 每 4 个字节用 `__byte_perm`/`__vadd2` 拆成 4 个 FP16（e4m3fn 还要处理 subnormal/NaN，走 256 项 shared-memory LUT 兜底），再乘 scale、转 fp16 写回。所以它是一次**读 1 B、写 2 B 的访存密集型转换**，算术开销可以忽略（见 §4.1 的四变体对照）。

---

## 2. 测量方法（为什么这些数字可信）

1. **CUDA event 包一整个循环，取多轮中位数**。Windows/WDDM 下单次调用的 event 计时含 28–35 µs 的提交/同步地板（`floor` 段实测：空调用 35.6 µs，16 B copy 单次 68–80 µs），直接测单次会把 100 µs 以下的 kernel 全部污染。所有数字都用 `gpu_time(fn, iters, rounds=7)`：先 warmup，再由一次 round 的耗时自动选 `iters`（目标 ~80 ms/round），7 轮取中位数，JSON 里保留全部 samples。
   > **口径提示（2026-11-08 补，§12.2）**：loop-event 与「逐迭代 event + sync」两种计时对 4096³ 级大 GEMM 会差 ~30%（同机同日：loop-event 1.889 ms = 72.8 TF vs 逐迭代 1.455 ms = 94.5 TF）。小 kernel 用 loop-event 是为了避开 28–35 µs 提交地板；大 kernel 的差值来自批量排队/WDDM 上下文切换被整段计入。**本报告所有绝对 TF 都是 loop-event 口径（偏保守），不能与逐迭代口径的 ~95 TF 直接比较；报告内部比值两边同口径，结论不受影响。**
2. **interleaved A/B**：tfp8 与 fp16 cuBLAS 在**同一进程、同一时刻**交替测量（`gemv` 段每次先跑 tfp8 再跑 cuBLAS），抵消争用漂移；跨段的绝对值不能直接比。
3. **两次全量 run** 检验稳定性：run2/run1 中位数的比值，46 个可比项里 40 项在 0.93–1.03，最差 1.24（chunked M=32，小 kernel 受争用影响），见 §9。
4. **正确性门禁**：微基准解码与 256 项 LUT 参考 **bit-exact**（e4m3 / e5m2 / 带 scale）；生产 kernel 每形状抽查前 8 行与 `dequantize_weight` 参考 bit-exact；微基准编码与 `torch` 原生 cast bit-exact（1 M 元素 0 mismatch，含 ±0、±inf、NaN 特例）。
5. 已知偏差来源：**共享卡的外部负载**（同一 kernel 跨 run 中位漂移可达 30–40%，run 内 max/min 1.5–2.6×；小 kernel 还叠加 event 地板，<50 µs 的数字应读作上限）；**时钟/功耗状态**（空闲 570–630 MHz 会低估 2–3×）。因此本报告重点看**比值与带宽效率**，绝对 µs 是「本机当前状态」下的值；§9 的独立复测额外做了 clock_burn 并记录时钟，用来给出满 boost 上限。需要绝对值时请同时看 median 与 min。

---

## 3. 可流带宽基准（后面的所有转译数字的物理上限）

| 测量（同进程，loop-event 中位数） | 时间 | 有效带宽 |
|---|---:|---:|
| micro：copy u8 256 MB（读+写） | 1571 µs | 342 GB/s |
| micro：copy f16 128 MB（读+写） | 811 µs | 331 GB/s |
| micro：fill f16 128 MB（写） | 407 µs | 330 GB/s |
| torch：`copy_` u8 256 MB | 1484 µs | **362 GB/s** |
| torch：`copy_` f16 128 MB | 760 µs | 353 GB/s |
| torch：`fill_` f16 | 318 µs | **422 GB/s** |
| torch：`sum` f16（纯读） | 332 µs | **405 GB/s** |
| cuBLAS FP16 4096³（tensor cores） | 1971 µs | **69.8 TFLOPS** |
| cuBLAS FP32 4096³ | 13418 µs | 10.3 TFLOPS |

→ 本次两轮测量时段的可流带宽 ≈ 342–362 GB/s（双向）、405–422 GB/s（单向）；**独立复测在 clock_burn 满 boost 状态下 copy 达 428.6 GB/s（最快 506.8）**，且 micro 解码最快样本 507 GB/s 与 copy 最快值几乎相等。所以「带宽墙」不是固定值：本机在 350–510 GB/s 之间随争用/时钟漂移。解码/编码各 3 B/元素，理论吞吐 = 带宽/3 → 117–169 Gelem/s（解码）/ 128–170 Gelem/s（编码）。**无论稳态中位还是满 boost，e4m3/e5m2/LUT/加 scale 四变体差异 <2%，证明转译本身是纯带宽受限、算术免费；省时间只能省字节。**

---

## 4. 解码（FP8 → FP16）：352–507 GB/s / 117–169 Gelem/s

### 4.1 微基准：45.1 M 个权重（11008×4096，Llama gate_proj 尺寸）

| 变体 | 时间 | Gelem/s | GB/s |
|---|---:|---:|---:|
| e4m3fn → f16 | **384.9 µs** | **117.2** | **351.6** |
| e5m2 → f16 | 382.4 µs | 117.9 | 353.7 |
| LUT（fmt 未知，走 256 项查表） | 379.3 µs | 118.9 | 356.6 |
| e4m3fn → f16 + group=128 scale | 382.8 µs | 117.8 | 353.4 |
| e4m3fn → f16 + per-channel scale | 386.3 µs | 116.7 | 350.2 |
| e4m3fn → f32（5 B/元素） | 679.1 µs | 66.4 | 332.0 |

- **四种解码变体差 <2%**：e5m2 的纯移位、e4m3fn 的 `__byte_perm`、LUT 查表、再加一次 scale 乘法，在 350–500 GB/s 量级的带宽墙面前完全不可见。**解码算术是免费的，省时间只能省字节。**
- 尺寸扫描（e4m3→f16）：1 M 33.4 µs（94 GB/s，launch 地板）、4 M 40.1 µs（313 GB/s）、16 M 145.8 µs（346 GB/s）、45.1 M 384.9 µs（352 GB/s）→ **≥16 M 元素才饱和；≤4 M 的读数会被 30 µs 级地板抬高**。
- 换算：**1 B 个权重的解码 = 8.5 ms**（含把 fp16 写回显存；若只算读出则 2.9 ms）。
- **独立复测（满 boost）**：自写 16-code/线程微内核（bit-exact vs LUT）45.1 M 元素中位 **363.0 µs = 124.2 Gelem/s（373 GB/s）**、最快 **266.8 µs = 169.0 Gelem/s（507 GB/s）**；最快样本与同 run copy 最快 506.8 GB/s 重合 → **满 boost 下解码确实贴在带宽墙上**。本次两轮的 117.2 Gelem/s 是偏中时钟窗口的值，不是 ceiling。

### 4.2 生产 kernel `tfp8_dequant_f16`：254.5 GB/s → 根因已定位、修复已落地（384.8 GB/s）

| 形状（e4m3fn, g128） | run1 / run2 | 有效带宽 | 备注 |
|---|---:|---:|---|
| qkv 11008×4096（45.1 M） | 544.7 / 518.7 µs | 254.5 GB/s | 比微基准低 27% |
| o 4096×4096（16.8 M） | 198.7 / 199.7 µs | 252.7 GB/s | |
| 1B 5632×2048（11.5 M） | 262.4 / 252.3 µs | 134.5 GB/s | |
| DiT 4608×1152（5.3 M） | 221.9 / 206.6 µs | 74.4 GB/s | |
| tiny 256×512 | 46.1 / 37.0 µs | — | 地板 |

**根因**：`tfp8/_ext.py` 里 `dequant_into` 的 grid 曾是 `(min(nchunk, 64), N, 1)`（原 175 行；修复后 183 行），`blockDim=128`，而 `nchunk = K/16` 是每行的 16-code chunk 数。一个 128 线程 block 恰好消费 128 个 chunk，所以**每行真正有用的 x-block 只有 `ceil(K/512)` 个**：

| K | nchunk | grid.x | 有用 x-block | 空 block/行 |
|---:|---:|---:|---:|---:|
| 4096 | 256 | 64 | 2 | **62** |
| 2048 | 128 | 64 | 1 | **63** |
| 1152 | 72 | 64 | 1 | **63** |
| 512 | 32 | 32 | 1 | 31 |

这些空 block 只做一次 `c < nchunk` 判断就退出，但它们的**调度本身就是开销**：耗时随行数 N（≈ 空 block 总数）单调增长，明显偏离「只由元素数决定」。注意不是严格正比——每行 K 不同，有用 block 数 `ceil(K/512)` 也不同。

**决定性证据**（`.verify/bench/gpu/diag_dequant.py`，同进程、同一 kernel）：

| 形状 | 元素数 | 行数 | 空 block 数 | shipped | 精确 grid | 提速 |
|---|---:|---:|---:|---:|---:|---:|
| qkv 11008×4096 | 45.1 M | 11008 | 682 496 | 526.5 µs / 257 GB/s | **361.3 µs / 374 GB/s** | **1.46×** |
| qkv 4096×11008（同元素，转置） | 45.1 M | 4096 | 237 568 | 360.4 µs / 375 GB/s | 359.0 µs / 377 GB/s | 1.00× |
| o 4096×4096 | 16.8 M | 4096 | 253 952 | 193.9 µs / 260 GB/s | **135.0 µs / 373 GB/s** | **1.44×** |
| 1B 5632×2048 | 11.5 M | 5632 | 354 816 | 245.4 µs / 141 GB/s | **93.9 µs / 369 GB/s** | **2.61×** |
| DiT 4608×1152 | 5.3 M | 4608 | 290 304 | 210.5 µs / 76 GB/s | **48.7 µs / 327 GB/s** | **4.32×** |
| qkv 11008×4608 | 50.7 M | 11008 | 671 488 | 576.6 µs / 264 GB/s | **410.6 µs / 371 GB/s** | **1.40×** |
| tiny 256×512 | 0.13 M | 256 | 7 936 | 58.2 µs | 57.7 µs | 1.01×（地板） |

- **同为 4510 万个权重，11008 行要 526.5 µs，4096 行只要 360.4 µs**（1.46×）。注意转置后每行 K=11008（有用 block 6 个/行），不是 K=4096（2 个/行），所以这是「同元素、不同行数且每行工作量不同」的对照，不能要求严格比例；但它与 grid sweep 一起排除了「只由元素数决定」的解释。独立复测同一对形状得 465.5 vs 274.4 µs = **1.70×**。
- **独立复测的 grid sweep 进一步确认空 block 代价**（同一 kernel、同 run）：qkv `gx=64`（生产值）**572.3 µs** vs `gx=1` **281.9 µs**；DiT `gx=64` **160.1 µs** vs `gx=3` **38.0 µs**（4.2× 跨度、单调）。按 `gx=ceil(K/512)` 修复后：qkv 528.5→332.0 µs（**1.59×**）、DiT 174.2→61.5 µs（**2.83×**，同配置 sweep 最优 38.0 µs = **4.6×**），两者全部 bit-exact。
- 修复只需把 launcher 的 grid.x 从 `min(nchunk, 64)` 改成 `max(1, ceil(nchunk/128))`（kernel 内部 `c += gridDim.x*blockDim.x` 的循环已经支持任意 grid.x），**所有形状输出与原 grid bit-exact**。修复已落地（`tfp8/_ext.py:183`）：run9/run10 实测生产 kernel 达到 **345–395 GB/s**，与手写微基准持平（§12.3）。
- 对推理的直接影响：DiT 层 210.5→48.7 µs（独立复测 174.2→61.5 µs，最快 38.0）、1B 模型 245→94 µs。**这是本次测试中最值得立刻落地的一条**（一行改动，1.4–4.6×）。

### 4.3 参考实现与 torch 的对比

| 路径 | 时间（45.1 M） | Gelem/s | 说明 |
|---|---:|---:|---|
| tfp8 微基准 kernel | 384.9 µs | 117.2 | 贴带宽墙 |
| **torch 原生 `float8→fp16` cast** | 606.8–624.1 µs | 72.2–75.7 | 本卡可用的原生 cast，比自研慢 **1.6×** |
| torch `lut[q.long()]` | 2288.6–2420.8 µs | 18.6–19.6 | 索引张量化 + gather，慢 6× |
| `tfp8.quantize.dequantize_weight`（参考路径） | 4027.3–4390.0 µs | 10.3–11.5 | fp32 中间表示 + 逐元素乘，慢 10× |

→ **自研 bit-trick 解码比 torch 原生 FP8 cast 还快 1.6×**（2080 Ti 的 fp8 转换没有硬件路径，torch 走的是逐元素通用 kernel），参考实现慢 10× 只适合做正确性金标准。

---

## 5. 编码（FP16 → FP8）：387 GB/s / 128.3 Gelem/s

| 路径 | 时间（45.1 M） | Gelem/s | GB/s | 备注 |
|---|---:|---:|---:|---|
| 微基准 `enc_e4m3`（自写精确 RNE+饱和） | 349.5 / 353.4 µs | **128.3** | **387** | 与 torch cast bit-exact |
| torch 原生 `fp16→float8` cast | 350.3 / 338.7 µs | 130.9 | 392 | 两条路径同速 |
| 微基准 f32→e4m3（5 B/元素） | 752.1 / 747.7 µs | 60.1 | 300 | f32 读 4 B 是瓶颈 |
| torch f32→float8 | 565.6 / 560.1 µs | 80.1 | 400 | torch 赢在少一次转换 |
| torch f32→bf16→f8（两步） | 1066.5 / 1012.0 µs | 43.4 | — | 不要两步走 |
| **GPU `quantize_weight`**（带 amax/pow2 scale） | 1830.1 / 1736.8 µs（4096²，16.8 M） | **9.4** | 18.8 | 比裸 cast 慢 **13.6×** |
| GPU `quantize_weight` per-channel | 1736.1 / 1676.4 µs（16.8 M） | 9.8 | — | scale 布局几乎无影响 |
| GPU `quantize_weight` 11008×4096 | 4619.2 / 4581.1 µs | 9.8 | — | 大形状同速 |

- **编码的物理上限也是带宽**：3 B/元素 × 387 GB/s ↔ 128.3 Gelem/s，实测与 torch 原生并列到小数点后一位，说明这条路径**没有进一步优化空间**（除非换字节数更少的输入格式）。
- **加上 scale 就掉 13.6×**：`quantize_weight` 是纯 Python 多层 pass（amax 归约 → pow2 scale → 除法 → clamp → RNE cast），GPU 上只有 9.4–9.8 Gelem/s。1 B 权重离线量化 ≈ **106 ms**（GPU），比裸 cast 的 7.8 ms 贵一个数量级——**这是目前整条链路里最慢的一步**，价值远不如 §4.2 的 grid 修复。
- 换算：**1 B 个权重的裸编码 = 7.8 ms**（读 2 GB + 写 1 GB）。

---

## 6. 融合 GEMM：转译算力直接决定的天花板 = 12–15 TFLOPS

### 6.1 4096×4096 单层，e4m3fn g128（M=32…2048）

| M | fused（WMMA 内解码） | chunked（先反量化+cuBLAS） | fp16 cuBLAS | fused TF | chunked TF | cuBLAS TF | fused 重复解码的权重 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.270 ms | 0.574 ms | 0.103 ms | 3.98 | 1.87 | 10.45 | 0.017 G |
| 64 | 0.315 ms | 0.520 ms | 0.099 ms | 6.81 | 4.13 | 21.59 | 0.017 G |
| 128 | 0.318 ms | 0.614 ms | 0.108 ms | **13.51** | 7.00 | 39.79 | 0.034 G |
| 256 | 0.675 ms | 0.559 ms | 0.173 ms | 12.72 | 15.37 | 49.72 | 0.067 G |
| 512 | 1.379 ms | 0.619 ms | 0.257 ms | 12.46 | 27.77 | 66.95 | 0.134 G |
| 1024 | 2.841 ms | 1.020 ms | 0.501 ms | 12.10 | 33.67 | 68.56 | 0.268 G |
| 2048 | 5.580 ms | 1.764 ms | 0.988 ms | 12.32 | **38.96** | **69.57** | 0.537 G |

**融合路径的 12–15 TFLOPS 平台期不是巧合，而是转译算力的直接换算**（本报告两轮为 12.1–13.5，满 boost 独立复测为 10.8–14.9）：

- 融合 kernel 的 block tile 是 64×64，**每 64 行 M 就要把整层权重重新解码一遍**（`tfp8_kernels.cu:284-305` 的 `k0` 循环里每个 block 都从 `uint8` 现场解码 sB）。
- 实测融合耗时对 tile 数（M/64）完全线性：**M=128 的 318.0 µs + (tiles−2)×175.4 µs** 能预测到 M=2048 的 5579.5 µs（误差 <0.1%）。即 **每 tile ≈175.4 µs**，而每 tile 解码 16.78 M 个权重 → **有效解码率 95.7 Gelem/s**。
- 每解码一个权重只换来 64 个 MAC（=128 FLOP），所以理论天花板 = 95.7 × 128 = **12.25 TFLOPS**，与实测 12.1–13.5 TF 吻合。**M 再大、算力再多也没用，瓶颈在转译。**
  > **更正（11-08 晚，§12.4.3）**：把 BM 从 64 加到 128（每层权重的重复解码次数减半）后实测反而 **0.57–0.99×**，bit-exact。若真是解码受限，时间应接近减半。所以「对 tile 数线性」只说明每个 M-tile 的成本固定，**主因是单缓冲 smem→WMMA 主循环（每 32 宽 K 步两次 `__syncthreads`、加载与计算不重叠）**，不是解码。上面的「天花板 = 解码率 × 128」只是巧合吻合。双缓冲落地后（§12.5）同一个 BM=64 kernel 达到 21–23 TF（per-channel），进一步证实了这一点。
- **独立复测（满 boost，自写计时）**：M=128/256/512/1024/2048 → 0.396/0.673/1.239/2.591/4.621 ms = 10.8/12.8/13.9/13.3/14.9 TF，线性拟合 **斜率 142.2 µs/M-tile、截距 140 µs、R²=0.9968**，折算 **117.9 Gelem/s**（比本报告 175.4 µs/tile、95.7 Gelem/s 高 21%，方向与量级一致，差异来自时钟状态）。两条独立测量都指向「每 M-tile 重解码整层权重」，换算天花板 12.2–15.1 TF（随状态），与实测 plateau 一致。

### 6.2 chunked 路径分解（M=2048，总 1763.8 µs）

| 阶段 | 时间 | 占比 | 说明 |
|---|---:|---:|---|
| prepare_activations（pow2 行 scale，1 次 kernel） | 114.9 µs | 6.5% | |
| **反量化转译**（`tfp8_dequant_f16`，只解码 1 次） | **206.5 µs** | **11.7%** | 修 grid 后还能降到约 145 µs（374 GB/s） |
| torch.mm FP16（workspace @ fp16 act，FP32 累加） | 1058.0 µs | 60.0% | 64.9 TF = cuBLAS 的 **93%** |
| epilogue（`out.mul_(xs[:,None])` 广播乘 + bias） | 359.6 µs | 20.4% | 明显偏慢，可融合 |
| cast fp32→fp16 | 133.7 µs | 7.6% | 50 MB @ 374 GB/s，已贴墙 |
| **合计** | **1763.8 µs** | | vs fp16 cuBLAS 987.8 µs = **1.79× 慢，权重显存 0.5×** |

**结论**：大 M 下**转译只占 11.7%**，时间主要花在 GEMM（60%）和 epilogue（20%）。所以「w8a16 省显存不省时间」这句 README 结论在 chunked 路径上成立的原因是**非转译开销**，而不是转译本身慢——这与 fused 路径（转译占 82%）形成鲜明对照。

**生产实现 vs 同数学精简实现**：独立复测手写了等价 chunked（反量化一次 → `torch.mm(out_dtype=float32)` → scale → cast，满 boost 时钟），M=2048 只用 **1.277 ms = 53.8 TF**，比生产路径 1.764 ms / 38.96 TF 快 28%；三段拆分 0.180（dequant）+0.131（scale）+0.100（cast）+0.890（mm）≈1.30 ms，与实测自洽。差额主要来自生产 `_chunked` 的 `prepare_activations`、bias、输出 cast 与本报告两轮的偏低时钟。**把 epilogue 融进 GEMM 或 cast 是下一个明确的优化点（生产路径约有 30% 非转译开销可省）。**

---

## 7. M=1 decode（GEMV）：与 FP16 cuBLAS 打平，原因不是转译强而是对方已到带宽墙

| 形状（M=1） | tfp8 GEMV | fp16 cuBLAS（同进程交替） | 比值 | tfp8 等效带宽 |
|---|---:|---:|---:|---:|
| 12288×4096 | 238.5 µs | 251.3 µs | **0.95×** | 211 GB/s |
| 11008×4096 | 220.6 µs | 218.8 µs | **1.01×** | 204 GB/s |
| 4096×4096 | 94.5 µs | 89.2 µs | **1.06×** | 178 GB/s |
| 4096×11008 | 243.5 µs | 235.7 µs | **1.03×** | 185 GB/s |
| 5632×2048 | 66.4 µs | 67.1 µs | **0.99×** | 174 GB/s |
| 4608×1152 | 72.6 µs | 78.1 µs | **0.93×** | 73 GB/s |

- **6 个真实形状全部在 0.93–1.06×**：FP8 权重只读 1 B/参数，比 FP16 少一半字节，**但没有变成 2× 加速**。
- 计算一下：fp16 cuBLAS M=1 读 86 MB 权重用 218.8 µs = **393 GB/s**（已在带宽墙）；tfp8 只读 43 MB 却用了 220.6 µs = 204 GB/s。也就是说 **tfp8 GEMV 被解码 ALU 限制在约 205 Gelem/s**（45.1 M / 220.6 µs），正好把省下的一半带宽吃掉，于是打平。
- GEMV 随 M 的增长很慢：11008×4096 的 M=2/4/8 分别 233/276/389 µs（1.85 TF @ M=8，只有 FP32 峰值的 14%），是 FP32 累加链 + 串行 FMA 的延迟受限。**小 batch 场景想赢 FP16，必须提高 GEMV 的 ILP（多累加器/更宽的 decode），把 204 GB/s 拉到 350 GB/s 上限（潜在 1.7×）。**

---

## 8. CPU 侧对照（同机，10 线程，纯 CPU，独立 teammate 实测）

| 路径 | 吞吐 | 相对 GPU |
|---|---:|---:|
| 编码 `quantize_weight`（e4m3fn per-channel，4096²） | **0.587 Gelem/s**（28.57 ms） | GPU 版 9.4 Gelem/s → **16×** |
| 裸 bf16→float8 cast | 2.351 Gelem/s | 比 GPU 裸编码 128.3 慢 55× |
| 解码 `lut[q.long()]`（16.8 M） | 1.216 Gelem/s（纯 gather 2.397） | 比 GPU 解码 117.2 慢 **96×** |
| `formats.encode` 参考路径 | 0.054 Gelem/s | 仅正确性金标准 |
| 线程缩放 | 10 vs 1 = 4.78×；20 逻辑线程无收益 | |
| 模型级（编码，10 线程） | **1B ≈ 1.71 s、7B ≈ 11.95 s** | |

→ **离线量化/转译应该放 GPU**（快 16–96×）；CPU 上 7B 全量编码 12 s 也不构成瓶颈，但它不是「推理算力」。
数据：`.verify/bench/cpu/transcode_cpu.json`、`SUMMARY.md`。

---

## 9. 稳定性和独立复测

**两次全量 run（run1 / run2）**：46 个可比项中 40 项落在 0.93–1.03；偏离较大的集中在两类：(a) 30–50 µs 的小 kernel（event 地板 + 争用，如 tiny 0.80×）；(b) chunked 的 M=32 小形状（1.24×）。**1B/DiT/o_proj 等真实形状的 decode 与 M=1 GEMV 偏差都在 ±5% 内**，结论不依赖单次测量。

**独立复测（teammate `gpu-verify`：自写 566 行脚本，未读/未 import 本报告的计时与微内核；共 4 次完整 run，取时钟最稳的第 4 次）**：

| # | 项目 | 本报告（两轮中位） | 独立复测 median（最快） | 偏差 | 判定 |
|---|---|---|---|---|---|
| 1 | copy 带宽（128 MiB `copy_`） | 322–356 GB/s | **428.6（506.8）GB/s** | +26% | 可复现，本报告为偏低下界 |
| 2 | micro 解码 45.1 M | 384.9 µs / 117.2 Gelem/s | **363.0（266.8）µs / 124.2（169.0）Gelem/s** | −5.7% | 可复现，ceiling 未被夸大 |
| 3 | 生产 dequant qkv | 519–545 µs / 248–262 GB/s | **465.5（395.3）µs / 290.6 GB/s** | −12.5% | 可复现 |
| 4 | 生产 dequant DiT | 207–222 µs / 72–79 GB/s | **190.5（157.1）µs / 83.6 GB/s** | −11.2% | 可复现 |
| 5 | grid 修复 | qkv 1.46×、DiT 4.32×（bit-exact） | qkv **1.59×**、DiT **2.83×**（同配置 sweep 最优 **4.6×**），bit-exact | — | 可复现，方向一致 |
| 6 | fused M=2048 | 5.40–5.75 ms / 11.9–13.0 TF | **5.865（5.745）ms / 11.7 TF** | +5.2% | 可复现 |
| 7 | chunked M=2048 | 1.74–1.79 ms / 38–41 TF | **1.277（1.205）ms / 53.8 TF**（等价精简实现） | −27.7% | 不可直接比：对方省略 prep/bias/输出 cast 且时钟更高；方向成立（仍比同口径 cuBLAS 慢 1.44×） |
| 8 | cuBLAS M=2048 | 0.98–1.00 ms / 69–75 TF（f32 out） | **0.890（0.667）ms / 77.2 TF**（f32 out；fp16 out 97.7 TF） | −10.1% | 可复现 |

**总判定：6 个 headline 的方向与量级全部可复现，没有一条被推翻。** 8 个子项里 5 个的 |偏差| ≤15%；最大的两处是 copy（+26%，本报告取到的是偏低下界）与 chunked（−27.7%，口径差异 + 时钟差已解释）。

**根因的两条独立证据**：
- RC1（fused 每 M-tile 重解码整层权重）：复测 M 扫描拟合 **142.2 µs/tile**（本报告 175.4），R²=0.9968，折算 **117.9 Gelem/s** ≈ 其独立微内核 124.2 Gelem/s → **支持**。
- RC2（空 block 使耗时随行数增长）：复测同元素两形状 465.5 vs 274.4 µs = **1.70×**（本报告 1.46×），而行数比 2.69× → 「随行数单调增长」成立，但**不是严格正比**；grid sweep（qkv `gx=64→1`：572→282 µs；DiT `gx=64→gx=3`：160→38 µs，单调、4.2× 跨度）是更干净的证据。§4.2 已按此改写。

**方法学审计（复测者对本报告的反向检查）**：
- cuBLAS 口径公平：同卡 f32-out 77.2 TF 与本报告 69–75 TF 同量级（差 ~10% 可由时钟解释），chunked 与 cuBLAS 的差距来自 dequant+scale+cast 与 mm 效率，不是累加精度。
- 绝对值受时钟状态支配：空闲 570–630 MHz / ~30 W，满 boost 1770–2010 MHz / 190–281 W；同一 kernel 跨 run 漂移 30–40%（dequant qkv 四次 run 421.8/466.3/485.1/465.5 µs，regrid DiT 77.9/44.7/52.6/61.5 µs），run 内 max/min 1.5–2.6×。**所以本报告主表只用于同一状态内部对比，跨状态绝对值以本节满 boost 值为上限。**
- 复测者未读本报告脚本（保证独立性），对账基于数值本身；本报告用「两轮 run 中位 + 全部 samples」发布数据，满足可复核要求。

**顺带发现的库隐患（非算力结论，2026-11-08 已定位并加固）**：`tfp8._ext._launch` 的 cached-argument slot 曾偶发 `OSError: exception: access violation reading 0x...5F5F`（复测中两次复现）。后续调试证明可稳定复现的缺陷是「同名内核 + 相同参数个数、不同类型（tensor/int/float）」时，旧实现只按 `(dev, name)` 和 `len(args)` 判重，复用了错误的 ctypes holder 类型，导致静默错误或把非法指针传给内核；现已改为按完整类型签名 `(dev, name, kinds)` 缓存并加锁，并新增回归门禁 `test_launch_slot_kinds_are_isolated`（门禁共 14 项）。原 `access violation` 崩溃本身在数千次旧代码启动（含验证者的全部调用模式）中未能复现，torch 的 module 生命周期假设也被排除，故按「已加固、原始触发条件未复现」记录。本报告主表数据均在修复前的 11-07/11-08 采集；源码于 11-08 之后修复，证据见 `.verify/bench/gpu/a1_debug.md`。

**测量地板（必须记住）**：本机空 kernel 的 loop-event 地板 28–32 µs、单次调用 68–71 µs（`floor` 段）。任何 <50 µs 的测量读数只能当上限；tfp8 小层（128×128）的 event-loop 是 65.6–67.4 µs、wall/call 62.5–67.5 µs——旧 README 里 37 µs 的 128×128 GEMV 数字在当前 torch 2.14 + WDDM 下**已不可复现**。

---

## 10. 结论

1. **本卡的「转译 FP8 算力」= 解码 117–169 Gelem/s（352–507 GB/s）、编码 128 Gelem/s（387 GB/s，独立复测未覆盖编码）**。满 boost 下解码最优样本 507 GB/s 与同 run copy 上限 506.8 GB/s 相等；e4m3fn/e5m2/LUT/加 scale 差异 <2% → **转译是纯带宽问题，算术免费**。主表里的 117.2 Gelem/s 是两轮共享状态的中位，不是 ceiling。
2. **模型级换算**：1 B 权重的解码 ≈ **8.5 ms**、裸编码 ≈ **7.8 ms**（都含 3 B/元素的读写）；带 amax+scale 的完整量化是 106 ms（GPU）/1.7 s（CPU）。
3. **融合 GEMM 的吞吐天花板由转译决定：12–15 TFLOPS，与 M 无关**。机制：每 64 行 M 重复解码一遍权重（本报告 175.4 µs/tile → 95.7 Gelem/s；独立复测 142.2 µs/tile、R²=0.9968 → 117.9 Gelem/s），×128 FLOP/权重 = 12.2–15.1 TF，与实测 plateau 吻合。**W8A16 在 fused 路径上是「用 4–5× 的转译换 1× 的算力」，大 M 不要走这条路。**
4. **大 M 的正确姿势是 chunked**：转译只占 11.7%（修复后 8.2%），mm 达 cuBLAS 93–96%，生产实现端到端 38.96 TF（修复后 42.79 TF，cuBLAS 的 56–57%），代价是 1.79× 时间换 0.5× 权重显存；同数学的精简实现独立复测达 **53.8 TF**（同口径 cuBLAS 77 TF 的 70%），说明生产路径的 prep/epilogue/cast（≈30%）可省——epilogue（20%）是下一个优化点。
5. **M=1 decode 与 FP16 cuBLAS 打平（0.93–1.06×）**：该状态下 FP16 cuBLAS 已到 393 GB/s 带宽墙；tfp8 只读 43 MB 却用同样时间（204 GB/s），受解码 ALU ≈205 Gelem/s 限制，省下的字节被解码吃掉。要赢需把 GEMV 拉到带宽墙（潜在 1.7×）。
6. **生产 dequant kernel 有一个免费的 1.4–4.6×**：`grid.x = min(nchunk, 64)` 让 62–63/64 的 x-block 是空的，耗时随行数/空 block 单调增长（同元素不同行 1.46–1.70×，grid sweep 单调跨度 4.2×）。改成 `max(1, ceil(nchunk/128))` 后 327–493 GB/s、bit-exact（§4.2）；该修复已于 11-08 12:47 落地，run9/run10 实测 qkv 384.8 GB/s、DiT 4.37×、1B 2.67×（§12.3）。
7. **2080 Ti 没有 FP8 硬件**，以上是 Turing 软件模拟的全部上限；换到 Hopper/Ada 的原生 FP8，这部分工作由硬件完成，量级完全不同——本报告的数字只代表「用 CUDA core + 显存带宽模拟 FP8」的成本。
8. **绝对值受时钟状态支配，结论看比值**：同一 kernel 跨 run 中位漂移 30–40%、run 内 max/min 1.5–2.6×，空闲与满 boost 时钟差 2–3×。主表两轮同状态中位只适合内部对比；跨状态绝对值以 §9 的满 boost 复测为上限。需要绝对值时请同时报 median 与 min。
9. 数据/脚本可复现：`python -I tests/bench_transcode.py`（GPU 全量，约 2 min）、`CUDA_VISIBLE_DEVICES="" python -I tests/bench_cpu_transcode.py`（CPU，约 23 s）、`python .verify\bench\gpu\diag_dequant.py`（根因探针）、`python .verify\bench\gpu\verify\verify_independent.py`（独立复测，约 2.5 min）。另有一条非算力隐患（已定位并加固）：`tfp8._ext._launch` 的 cached-arg slot 在多 shape 交替调用后可能复用错误 holder 类型（§9）。

---

## 11. 复现

```bash
cd /h/demo
# GPU：可流带宽 / 解码 / 编码 / GEMV / GEMM / 地板（写 .verify/bench/gpu/transcode_gpu.json）
python -I tests/bench_transcode.py
# 分节（快速）
python -I tests/bench_transcode.py --only bw,decode --json .verify/bench/gpu/transcode_probe.json
# 生产 dequant 的 grid 根因探针（shipped vs exact grid vs micro）
python .verify/bench/gpu/diag_dequant.py
# 11-08 复测对比（依赖 run1/run2/run3/run4 的 JSON，输出 compare_new.log）
python .verify/bench/gpu/compare_new.py
# CPU 侧（不初始化 CUDA）
CUDA_VISIBLE_DEVICES="" python -I tests/bench_cpu_transcode.py
# 数值门禁
python -I tests/test_tfp8.py
```

| 产物 | 内容 |
|---|---|
| `tests/bench_transcode.py` | GPU 全量基准（6 段，NVRTC 微内核 + 生产 kernel + torch 参考 + bit-exact 门禁） |
| `tests/bench_cpu_transcode.py` | CPU 全量基准（编码/解码/基线/线程缩放/模型外推） |
| `.verify/bench/gpu/transcode_gpu_run1.json` / `run2.json` | 11-07 两次全量数据（每项 median/min/max/mean/全部 7 轮 samples + 机器信息 + GPU 快照） |
| `.verify/bench/gpu/transcode_gpu_run3.json` / `run4.json`、`run3.log` / `run4.log` | 11-08 复测两轮原始数据与 stdout（§12） |
| `.verify/bench/gpu/compare_new.py`、`compare_new.log`、`run3_run4_compare.json` | 复测对比脚本、日志与机读结果（§12） |
| `.verify/bench/gpu/transcode_gpu_run5.json` ~ `run8.json`、`run5.log` ~ `run8.log` | 11-08 后四轮原始数据与 stdout（§12.1 / §12.2；run7/run8 为解锁复核轮） |
| `.verify/bench/gpu/compare_all8.py`、`run1_8_compare.log`、`run1_8_compare.json` | 8 轮全量对比脚本/日志/机读结果；`compute_unlock_probe.py` / `.json` 为解锁状态的独立探针（§12.2） |
| `.verify/bench/gpu/transcode_gpu_run9.json` / `run10.json`、`run9.log` / `run10.log` | 11-08 14:2x 代码修复后两轮原始数据与 stdout（§12.3） |
| `.verify/bench/gpu/compare_run9.py`、`compare_run9.log`、`run9_run10_compare.json` | 修复后 vs 修复前（r5~r8）对比脚本/日志/机读结果（§12.3） |
| `.verify/bench/gpu/diag_dequant.json` | grid 根因探针数据 |
| `.verify/bench/cpu/transcode_cpu.json`、`SUMMARY.md` | CPU 数据与中文汇总 |
| `.verify/bench/gpu/verify/verify_independent.py` / `.json` / `VERIFY.md` | 独立复测：566 行脚本、每项 median/min/samples 与对账数据、中文结论（6/6 headline 可复现） |
| `.verify/bench/gpu/run1.log`、`run2.log` | 原始 stdout |

**注意**：`tests/bench_transcode.py` 的 `_MICRO_SRC` 是从 `tfp8/csrc/tfp8_kernels.cu` 复制的解码原语（dec1/dec4/dec4x4），并自写了一份精确 RNE+饱和的 `enc_e4m3()` 编码；微基准只用于测「这条位运算在带宽墙前的成本」，不是生产代码。生产 kernel 的修复建议见 §4.2；本报告初版未修改任何 tfp8 源码——11-08 12:47 项目自身已把该 grid 修复与 `_launch` 槽位修复落地，修复后数据见 §12.3。

---

## 12. 2026-11-08 复测（run3 ~ run10）

**背景与状态**：用户要求再次测试（11-08 09:06 起）。复查 nvidia-smi：功耗墙 Current = Default = **300 W**、Max = **366 W**，VBIOS `90.02.17.00.9f`、驱动 621.01、torch 2.14.1+cu132 均未变；本轮未做任何显卡设置改动，只是把同一脚本再跑两遍（run3 / run4）。**更正（见 §12.2）**：当时把「366 W 未启用」误写成「算力未解锁」——实际上 FP16 tensor 半速锁自 11-02 00:03 冷启动后已解除，run1–run6 全部是解锁态数据；300 W 只是功耗墙。

**与 11-07 两轮的环境差异**：run3 起点 util 22%、VRAM 1417 MB，run4 起点 util 16%、1426 MB（11-07 两轮为 36–38%、2029–2147 MB）→ 桌面争用（wallpaper_engine / ChatGPT / Edge）明显更少；两轮结束均为 SM 2010 MHz、功耗 122.9–133.4 W（仍远低于 300 W 墙）。

| 项 | 11-07 双轮中位 | 11-08 run3/run4 中位 | 变化 |
|---|---|---|---|
| micro copy u8 256 MB | 342 GB/s | **389.2 GB/s** | +13.8% |
| micro copy f16 128 MB | 331 GB/s | 366.2 GB/s | +10.6% |
| micro fill f16 | 330 GB/s | 363.9 GB/s | +10.3% |
| cuBLAS fp16 4096³ | 69.8 TF | 73.8 TF | +5.8% |
| 解码 micro e4m3→f16 | 117.2 Gelem/s（351.6 GB/s） | **121.8 Gelem/s（365.3 GB/s）** | +3.9% |
| 生产 dequant qkv g128 | 531.7 µs（254.5 GB/s） | **495.9 µs（272.8 GB/s）** | +7.2% |
| 生产 dequant dit | 214.2 µs | 196.0 µs | +9.3% |
| 编码 micro f16→e4m3 | 128.3 Gelem/s（387 GB/s） | **138.7 Gelem/s（416.1 GB/s）** | +8.1% |
| torch cast f16→f8 | 130.9 Gelem/s | 142.8 Gelem/s | +9.1% |
| `quantize_weight` 4096² g128 | 9.41 Gelem/s | 10.38 Gelem/s | +10.3% |
| GEMV M=1（6 形状） | 0.93–1.06× fp16 cuBLAS | **0.80–1.05×** | 同结论 |
| 融合 GEMM（M=128…2048） | 12.3–13.5 TF | 13.2–14.4 TF | +4–7% |
| chunked M=2048 | 1763.8 µs / 38.96 TF | **1654.4 µs / 41.60 TF** | +6.7% |
| cuBLAS M=2048 | 987.8 µs / 69.6 TF | 917.8 µs / 74.9 TF | +7.6% |

> 表内变化按「时间越短/吞吐越高」为正；个别小 kernel（1 M 解码、5632×2048 M=2）落在噪声内（±10–30%），未列入。

**run3 chunked 分解（M=2048）**：prep 104.4 µs、dequant 189.7 µs（**11.4%**）、mm 935.3 µs（58.7%，73.5 TF）、epilogue 322.9 µs（20.3%）、cast 122.0 µs（7.7%）、total 1592.2 µs；与 11-07 的比例结构一致（dequant ≈11–12%、epilogue ≈20%）。

**结论**：
1. 全部数值从旧到新加快 **4–12%**（时间比 0.88–0.96），但**所有结论与排名不变**：解码/编码贴带宽墙、生产 dequant 的 grid 缺陷仍是 1.4–4.6× 的免费提速点、融合 GEMM 仍 12–15 TF 平台期、chunked 仍是生产 38–42 TF（精简实现可到 53.8 TF）、GEMV 仍与 fp16 cuBLAS 打平。
2. 差异来自**桌面争用/时钟状态**：功耗墙仍 300 W（上限 366 W 未启用）、VBIOS/驱动未变；负载功耗仍只有 ~123–133 W。FP16 张量半速锁虽自 11-02 已解除（§12.2），但本工作负载是带宽/解码受限，解锁不改变这些数字——run7/run8 在确认解锁后再测，结果同量级。
3. 与独立复测的「满 boost」上限（copy 428.6 GB/s、解码 124.2/169.0 Gelem/s）相比，本轮仍低于其上限（copy 389 vs 428.6）；即「温和争用下的当前默认状态」。
4. 若后续启用 366 W 功耗墙或进一步减少争用，预计只会小幅上移（功耗墙不是瓶颈，当前负载离 300 W 墙还有 ~170 W），真正的上限参考 §9 满 boost 列。

**数据产物**：`.verify/bench/gpu/transcode_gpu_run3.json`、`transcode_gpu_run4.json`、`run3.log`、`run4.log`、`compare_new.py`、`compare_new.log`、`run3_run4_compare.json`；复测命令与 §11 相同（`python -I tests/bench_transcode.py --json .verify/bench/gpu/transcode_gpu_run3.json`）。

### 12.1 2026-11-08 10:2x 再复测（run5 / run6）

同一脚本又跑两轮（均 exit 0），与 09 点两轮 r3/r4 对比多数项在噪声内（0.96–1.04），比 11-07 快 4–11%：

| 项 | 11-07 r1/r2 | 09:0x r3/r4 | **10:2x r5/r6** |
|---|---|---|---|
| micro copy u8 256 MB | 342 GB/s | 389.2 GB/s | **382.6 GB/s** |
| 解码 micro e4m3→f16 | 117.2 Gelem/s | 121.8 Gelem/s | **126.2 Gelem/s（378.6 GB/s）** |
| 生产 dequant qkv g128 | 531.7 µs | 495.9 µs | **507.6 µs（266.5 GB/s）** |
| 生产 dequant dit | 214.2 µs | 196.0 µs | **196.5 µs（81.0 GB/s）** |
| 编码 micro f16→e4m3 | 128.3 Gelem/s | 138.7 Gelem/s | **136.3 Gelem/s（409.0 GB/s）** |
| `quantize_weight` 4096² g128 | 9.41 Gelem/s | 10.38 Gelem/s | **10.2 Gelem/s** |
| 融合 GEMM M=2048 | 13.0 TF | 13.2 TF | **13.2 TF（5.20 ms）** |
| chunked M=2048 | 38.96 TF | 41.60 TF | **41.42 TF（1659 µs）** |
| cuBLAS fp16 M=2048 | 69.6 TF | 74.9 TF | **75.5 TF（911 µs）** |
| GEMV M=1（6 形状） | 0.93–1.06× | 0.80–1.05× | **0.92–1.14×** |

- 起点 util 20%/16%、SM 1350/1440 MHz（11-07 为 36–38%）；两轮结束均 SM 2010 MHz、功耗 123.1/122.5 W（远低于 300 W 墙）；bit-exact 全 True。
- chunked 分解 M=2048（r5/r6）：dequant 187.9/192.6 µs（11–12%）、mm 955.9/968.1 µs（≈73.6 TF）、epilogue 332.4/330.2 µs（20%）、total 1656.9/1661.1 µs。
- 结论：与 09 点两轮同状态、同量级，差异主要是时钟/桌面争用噪声；所有结论与排名不变，仍低于满 boost 上限（copy 428.6 GB/s、解码 124.2/169.0 Gelem/s）。
- 产物：`.verify/bench/gpu/transcode_gpu_run5.json`、`transcode_gpu_run6.json`、`run5.log`、`run6.log`、`compare_run5.py`、`run5_run6_compare.json`。

### 12.2 2026-11-08 11:39–11:41 复测（run7 / run8）：解锁状态复核 + 计时口径对照

**背景**：用户确认「显卡算力已解锁」，要求复测转译。11:30–11:35 先做了独立复核，结论是**解锁为真、但与本报告的转译数字无关**，且 run1–run6 本来就是在解锁态下测的；run7 / run8 是确认后在同一状态下的两轮复测（均 exit 0、bit-exact 门禁全绿）。

#### 12.2.1 解锁状态：为真，自 11-02 起生效

| 判据 | 锁定态（11-01，原厂 BIOS） | 解锁态（11-02 冷启动后） | 本地复核（11-08 11:30） |
|---|---|---|---|
| FP16(FP32 累加) 4096³ tail | 58.70 TFLOPS @ SM 1950 | **94.97 TFLOPS @ SM 1743** | **94.78 TFLOPS** |
| 每时钟吞吐 | 30,251 FLOP/clk | **54,452 FLOP/clk = 1.56× 34,816（per-clock ×1.80）** | 约 54.3k FLOP/clk（按 SM≈1743 MHz 折算） |
| 相对锁定基准 | 0.87× 34,816 | **1.56×（判据 ≥1.20×）** | **1.56×** |
| 功耗墙 | 300 W（默认） | 300 W（默认，未改） | 300 W（默认，未改） |

- **机制**：Turing FP16 tensor 半速锁被「预 OS UEFI FEAT/PLM 写」解除（SS0=0x88888888、SS1=0x8、PLM=0xffffffef 的 sticky 位），只在 GSP-RM 模式下被接受；VBIOS `90.02.17.00.9f`、驱动 621.01、GSP `[1,1]` 均未变。不是功耗墙解锁，也不是超频——FP32/INT8/FP64/带宽全部保持原厂水平（FP32 10.52 TF、INT8 49.4 TOPS、copy 464–510 GB/s）。
- **证据**：`H:\gpu-unlock-test\report.md` §14（11-02 00:03 冷启动后首次生效）、§15（11-08 三轮持续 95.44 / 94.65 / 95.18 TF，60 s 浸泡 95.56 TF、0 停顿）；本机 11:30 用其 `_retest_1101_child.py fp16 4096 200000 15` 独立复现：iters=7467、best 1.321 ms（104.0 TF）、median 1.451 ms（94.69 TF）、tail 1.450 ms（**94.78 TF**）、0 停顿。上游 nv-uefi-unlocker README 也把这型号列为成功案例（58.70→94.97 TF，+62%）。
- 直接读回（`ssdiag`）与 366 W 功耗墙测试需要管理员，本次未做；`nvidia-smi -pl 366` 普通 shell 报 `Insufficient Permissions`。
- **注意**：我最初用「20 次一批的循环 event」自写探针只测到 76.3 TF（`.verify/bench/gpu/compute_unlock_probe.json`），一度误以为「部分解锁」——这是计时口径问题，不是显卡状态问题，见 12.2.2。

#### 12.2.2 计时口径对照（重要）

| 计时方法 | 4096³ fp16 cuBLAS | TFLOPS |
|---|---|---|
| 本报告 `gpu_time`（一次 event 包住整轮 iters≈40 次调用） | 1.889 ms | 72.8 |
| 20 次一批循环 event（独立微测） | 1.824 ms | 75.4 |
| 逐迭代 event + sync（`H:\gpu-unlock-test` 口径） | 1.455 ms | **94.5** |

- 同一张卡、同一小时内、同一个 `torch.mm` 调用，仅计时方式不同就差 ~30%。两种都是真实 event 时间：loop 计时把连续排队执行整段计入（含提交/WDDM 上下文切换与共享卡争用），逐迭代计时让每个 kernel 在空闲 GPU 上单独跑。差值来源未进一步拆分（需要独占卡/关桌面），但两种口径各自可复现。
- 因此：(a) 本报告的 cuBLAS 参考（69.8–75.5 TF）是 loop-event 口径，**不能**与解锁复核的 94.5–95 TF 直接比较；(b) 报告内部 tfp8/fp16、mm/cuBLAS 等比值两边同口径，结论有效（§6.2 的「mm 达 cuBLAS 93%」即同口径；run7/run8 同口径为 96%）；(c) 要与 `H:\gpu-unlock-test` 对齐时用逐迭代口径重测——本次**未改** `bench_transcode.py`，只记录结论。

#### 12.2.3 run7 / run8 结果（解锁确认后，同状态对比）

| 项 | 09:0x r3/r4 | 10:2x r5/r6 | **11:39–11:41 r7/r8** |
|---|---|---|---|
| micro copy u8 256 MB | 389.2 GB/s | 382.6 GB/s | **380.6 GB/s** |
| 解码 micro e4m3→f16 | 121.8 Gelem/s（365.3 GB/s） | 126.2（378.6） | **126.5（379.4）** |
| 生产 dequant qkv g128 | 272.8 GB/s | 266.5 GB/s | **271.6 GB/s** |
| 编码 micro f16→e4m3 | 138.7 Gelem/s（416.1 GB/s） | 136.3（409.0） | **134.8（404.4）** |
| 融合 GEMM M=2048 | 13.2 TF | 13.2 TF | **13.27 TF（5.180 ms）** |
| chunked M=2048 | 41.60 TF | 41.42 TF | **41.16 TF（1670 µs）** |
| cuBLAS M=2048（loop-event 口径） | 74.9 TF | 75.5 TF | **74.90 TF（917 µs）** |
| GEMV M=1（6 形状，vs fp16 cuBLAS） | 0.80–1.05× | 0.92–1.14× | **0.85–1.04×** |

- chunked 分解 M=2048（r7/r8）：prep 105.9/108.3 µs、dequant 186.0/191.0（≈11%）、mm 959.9/966.0（71.6 TF，同轮 cuBLAS 的 96%）、epilogue 329.1/337.9（≈20%）、cast 121.3/124.0、total 1653.0/1686.9 µs。
- 环境：起点 util 28%/23%、SM 1350 MHz、80.4/74.9 W、VRAM 1418/1320 MB；结束均 SM 2010 MHz、123.4/129.6 W（远低于 300 W 墙）。bit-exact 全 True；GEMV 最大相对误差 6.25%（12288×4096），与 11-07 相同。
- 与 r3/r4 相比多数项在 ±1–3% 内（噪声，r7/r8 略低）；与 r5/r6 基本重合。三批同状态复测说明：**解锁与否不改变转译吞吐**。

#### 12.2.4 结论

1. **解锁为真**（11-02 起，11-08 复核仍有效）：FP16 tensor 每时钟 1.56× 锁定值，仅此路径变化；功耗墙仍是 300 W 默认，与解锁无关。
2. **本报告的转译数字不需要重测**：run1–run6 已全部在解锁态测得（11-02 之后才开始跑），run7/run8 在明确确认解锁后再测得到同一组数（±3%）。解码/编码是带宽受限、融合 GEMM 是解码 ALU 受限、chunked 的 mm 与同口径 cuBLAS 只差 4%，都不吃「解锁多出来的 FP16 张量算力」。
3. run7/run8 产物：`transcode_gpu_run7.json`、`run8.json`、`run7.log`、`run8.log`、`compare_all8.py`、`run1_8_compare.log`、`run1_8_compare.json`；解锁复核数据在 `H:\gpu-unlock-test\report.md` §14/§15 与本机 `.verify/bench/gpu/compute_unlock_probe.json`。
4. 继续提升转译吞吐的方向不变：带宽/解码硬上限（117–169 Gelem/s）、生产 dequant 的 grid 修复（1.4–4.6×）、chunked epilogue（~20%）；解锁后的 95 TF FP16 张量核只对 chunked 的 mm 段有意义，而 mm 已接近同口径 cuBLAS。

### 12.3 2026-11-08 14:2x：代码修复后复测（run9 / run10）—— grid 修复落地效果

**背景**：用户再次要求测试。此时项目已被另一轮修改（12:16–14:07，非本报告所做）：`tfp8/tests/` 整理为顶层 `tests/`；`tfp8/_ext.py` 落地两处修复——`dequant_into` 的 grid 改为 `max(1, ceil(nchunk/128))`（即 §4.2 的修复）、`_launch` 的槽键改为 `(dev, name, kinds)` 并加锁（修掉多 shape 交替时的 cached-slot access violation 隐患）；`tfp8/linear.py` 缓存 probe/LUT、NaN 检查只做前 3 次。`tfp8/csrc/tfp8_kernels.cu` 未变。`python -I tests/test_tfp8.py` 14 项门禁全 PASS（`CUDA kernels tested`）。

**环境**：功耗墙仍 300 W 默认（上限 366 W 未启用）、空闲 SM 1350 MHz；run9 起点 util 22%/VRAM 1460 MB、run10 起点 27%/1460 MB；两轮结束 SM 2010 MHz、功耗 131.2/128.7 W。两轮均 exit 0、bit-exact 全 True。

| 项 | 修复前 r7/r8 中位 | **修复后 r9/r10 中位** | 变化 |
|---|---:|---:|---:|
| 生产 dequant qkv 11008×4096 e4m3 g128 | 497.96 µs / 272 GB/s | **351.56 µs / 384.8 GB/s** | **1.42×** |
| 生产 dequant qkv perchan | 496.82 µs | **345.52 µs / 391.5 GB/s** | 1.44× |
| 生产 dequant qkv e5m2 g128 | 478.32 µs | **342.76 µs / 394.6 GB/s** | 1.40× |
| 生产 dequant o 4096×4096 | 195.88 µs / 253 GB/s | **132.57 µs / 379.7 GB/s** | 1.48× |
| 生产 dequant 1B 5632×2048 | 245.21 µs / 134 GB/s | **92.02 µs / 376.0 GB/s** | **2.67×** |
| 生产 dequant DiT 4608×1152 | 201.09 µs / 74 GB/s | **46.04 µs / 345.9 GB/s** | **4.37×** |
| 解码 micro e4m3→f16 45.1 M | 356.50 µs / 126.5 Gelem/s | 365.38 µs / 123.4 Gelem/s | 0.98×（噪声） |
| 编码 micro f16→e4m3 45.1 M | 334.48 µs / 134.8 Gelem/s | 324.69 µs / 138.9 Gelem/s | 1.03× |
| 融合 GEMM M=2048 | 13.27 TF（5.180 ms） | 13.22 TF（5.198 ms） | 1.00× |
| chunked M=2048 | 1670 µs / 41.16 TF | **1605.9 µs / 42.79 TF** | 1.04× |
| cuBLAS fp16 M=2048（loop-event 口径） | 74.90 TF（917 µs） | 74.84 TF（918.5 µs） | 1.00× |
| GEMV M=1（6 形状 vs fp16 cuBLAS） | 0.85–1.04× | 0.93–1.21× | 同结论 |
| micro copy u8 256 MB | 380.6 GB/s | 377.6 GB/s | 1.00× |

- **修复后生产 dequant 已与微基准持平**：qkv 384.8 GB/s vs 同轮 micro 370.2 GB/s（略高），perchan/e5m2 达 391–395 GB/s；DiT 从 74 GB/s 提到 345.9 GB/s。grid 缺陷对「行数多、K 小」的形状伤害最大（1B 2.67×、DiT 4.37×），与 §4.2 的机理一致。
- **chunked 分解（M=2048，r9/r10 中位）**：prep 111.4 µs、**dequant 132.4 µs（8.2%，修复前 186–191 µs / 11–12%）**、mm 973.6 µs（70.6 TF，同轮 cuBLAS 的 94%）、epilogue 329.4 µs（20.5%）、cast 122.0 µs（7.6%）、total 1605.9 µs（修复前 1670 µs）。转译占比降到 8%，剩余大头仍是 mm（61%）与 epilogue（21%）。
- **不受影响的部分**：解码/编码微基准、融合 GEMM（其 WMMA 主循环自带解码，不走 `dequant_into`）、cuBLAS 参考——修复只作用于生产 dequant 路径。
- **GEMV 表**：本轮小形状（5632×2048、4608×1152）的 tfp8 与 fp16 cuBLAS 同时变慢（92.6 vs 92.8 µs；93.8 vs 77.3 µs），floor 段的 launch 地板也从 ~65 µs 升到 87–101 µs——是 14:2x 桌面争用导致的提交延迟升高，不是代码回归；比值（tfp8/fp16）仍在 0.93–1.21×，结论不变。
- **结论**：§4.2 的「一行 grid 修复」在真实代码上兑现：生产解码吞吐 272 → **384.8 GB/s（同状态 +41%）**，DiT/1B 这类小 K 形状提升 2.7–4.4×，输出 bit-exact；转译在 chunked 中的占比降到 8%。其余结论（带宽受限、融合 GEMM 12–15 TF 天花板、GEMV 打平、chunked 剩余开销在 epilogue）全部不变。
- **产物**：`.verify/bench/gpu/transcode_gpu_run9.json`、`run10.json`、`run9.log`、`run10.log`、`compare_run9.py`、`compare_run9.log`、`run9_run10_compare.json`。

### 12.4 2026-11-08 晚：按 §6.2 / §12.3 指出的优化点改代码（chunked epilogue 融合 + 3 个正确性修复 + BM=128 试验）

**改动**（`tfp8/csrc/tfp8_kernels.cu`、`tfp8/_ext.py`、`tfp8/linear.py`、`tests/test_tfp8.py`、`tests/bench_transcode.py`）：

1. **chunked epilogue 融合**：新 kernel `tfp8_epilogue_{f16,bf16,f32}`，一次读 FP32 cuBLAS 结果、乘行 scale、加 bias、直接写输出 dtype，并写进输出的列窗口（`_ext.epilogue_into`）。旧实现有四遍：`out[:, n0:n1] = mm(...)` 的 32 MB 拷贝、`mul_`、`add_`、`.to()`，而且 §6.2 的分解漏算了那次拷贝。不再分配 FP32 [M,N] 全尺寸缓冲。
2. **正确性修复**（都没被原来 14 项门禁覆盖）：
   - `_chunked` 调 `dequant_into` 时没传 `fmt`，**e5m2 层走 chunked 时按 e4m3fn 解码**（输出错误）。
   - `prep_rows_kernel` 的行 amax 只取 thread 0 那一段（无块归约）。离群值落在别的线程时 scale 偏小，最坏会让 fp16 溢出成 inf。
   - `_launch` 把 Python float 装进 `c_double`，而 kernel 形参是 4 字节 `float`：kernel 读到 8.0 的低 32 位 = 0.0，**`ACT_TARGET_EXP` 实际一直是 0**（行 amax 被映射到 ≤1 而非 [2^7, 2^8)），离 fp16 次正规数少约 8 个 binade 的余量。改成 `c_float`。
3. **BM=128 融合 GEMM 试验**：`gemm_w8a16` 模板化为 `WM`（BM=32·WM），新增 `tfp8_gemm128_*`（256 线程）。默认不自动选用，见下文。
4. 门禁 14 → **17 项全 PASS**：新增 `test_chunked_formats_dtypes_and_slices`（e4m3fn/e5m2 × f16/bf16/f32 × 分组；多 slice 列窗口、奇数 N 标量路径）、`test_wide_tile_gemm_matches_narrow`（BM128 与 BM64 逐位相等）、`test_prep_row_amax_is_block_wide`（离群值不溢出，并且与 `activation_to_fp16` 逐位相等）；`test_launch_slot_kinds_are_isolated` 的 holder 类型同步改为 `c_float`。

#### 12.4.1 交替 A/B（同进程同时钟，`.verify/bench/gpu/ab_epilogue_bm128.py`，15 轮中位）

| chunked（e4m3 g128，fp16 out，带 bias） | 旧 | **新** | 加速 | 新 TF / cuBLAS TF | bit-exact |
|---|---:|---:|---:|---:|:-:|
| o 4096×4096 M=256 | 538.9 µs | **348.5 µs** | 1.55× | 24.6 / 51.2 | ✓ |
| o 4096×4096 M=512 | 601.9 µs | **411.7 µs** | 1.46× | 41.7 / 78.1 | ✓ |
| o 4096×4096 M=2048 | 1729.0 µs | **1139.1 µs** | 1.52× | **60.3** / 80.6 | ✓ |
| gate 11008×4096 M=256 | 1422.2 µs | **1022.0 µs** | 1.39× | 22.6 / 54.9 | ✓ |
| gate 11008×4096 M=512 | 1500.5 µs | **1166.9 µs** | 1.29× | 39.6 / 68.9 | ✓ |
| gate 11008×4096 M=2048 | 4645.3 µs | **3153.6 µs** | 1.47× | 58.6 / 78.4 | ✓ |
| DiT 4608×1152 M=256 | 472.7 µs | **330.1 µs** | 1.43× | 8.2 / 34.4 | ✓ |
| DiT 4608×1152 M=512 | 404.8 µs | **285.6 µs** | 1.42× | 19.0 / 47.1 | ✓ |
| DiT 4608×1152 M=2048 | 1139.2 µs | **564.3 µs** | **2.02×** | 38.5 / 65.9 | ✓ |

只看 epilogue（M=2048、N=4096、fp16 out）：旧（拷贝 + mul + add + cast）**635.7 µs** → 新融合 **123.2 µs**（409 GB/s，贴近 copy 上限）。（A/B 表里的 chunked 不含 prep，是直接调 `_chunked(x16, xs)`。）

#### 12.4.2 生产基准（`bench_transcode.py --only gemm`，run11）

| M（4096²） | fused | chunked（前 r9/r10） | **chunked（r11）** | cuBLAS |
|---:|---:|---:|---:|---:|
| 256 | 13.5 TF | — | **18.1 TF** | 53.6 TF |
| 512 | 13.3 TF | — | **33.2 TF** | 74.1 TF |
| 1024 | 13.4 TF | — | **48.7 TF** | 76.0 TF |
| 2048 | 13.3 TF | 42.79 TF（1605.9 µs） | **53.6 TF（1281.9 µs）** | 76.3 TF |

分解（M=2048）：prep 107.5 µs、dequant 132.5 µs（10%）、mm 942.8 µs（74%）、**epilogue 121.1 µs（9%，前 329.4 + cast 122.0 = 451.4 µs / 28%）**、cast 0。chunked 在 M=256 已超过 fused（18.1 vs 13.5 TF），默认 `large_m=256` 仍是合适的切换点。剩余可省的只有 prep（8%），把它和 mm 的 A 侧融合以外，基本已到「dequant + cuBLAS」结构的上限：53.6 TF ≈ 70% cuBLAS，差额基本是 dequant + prep + epilogue 三遍各 ~110–130 µs 的带宽。

端到端 `demo_llm --size tiny`：logits cosine 0.99734（与改前 0.9973 一致）、top-1 82.81%。

#### 12.4.3 BM=128：否证「融合 GEMM 受限于重复解码」

| 形状 | M=64 | 96 | 128 | 192 | 255 | 512 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| o 4096² BM128/BM64 速度比 | 0.96× | 0.96× | 0.97× | 0.99× | 0.97× | 0.94× | 0.91× |
| gate 11008×4096 | 0.67× | 0.94× | 0.94× | 0.71× | 0.96× | 0.97× | 0.96× |
| DiT 4608×1152 | 0.57× | 0.89× | 0.94× | 0.71× | 0.93× | 0.99× | 0.94× |

BM=128 让每层权重的解码次数减半，但所有点都没变快（结果逐位相等）。所以 §6.1 的「12–15 TF = 解码率 × 128 FLOP」只是数值巧合。真正的限制是 WMMA 主循环本身：单缓冲 smem、每 32 宽 K 步两次 `__syncthreads`、全局加载和 mma 不重叠。BM 加倍以后每个 block 的 MMA 量翻倍，而延迟没被隐藏。BM=128 因此保留为可选试验（`_ext.GEMM_WIDE_MIN_M = None`，`gemm_prepared(wide=True)` 可强制），生产仍用 BM=64。**融合 GEMM 的下一步应当是双缓冲 / cp.async 风格的软件流水（Turing 没有 cp.async，可以用寄存器预取下一 K 块）**，而不是加大 BM —— 已在 §12.5 实现。BM=128 kernel 随后删除（它的双缓冲版 smem 需要 63.5 KB，超过 48 KB 静态上限，而且结论已经明确）。

**产物**：`.verify/bench/gpu/ab_epilogue_bm128.py` / `.json` / `.log`、`transcode_gpu_run11_gemm.json`、`run11_gemm.log`。

### 12.5 2026-11-08 晚：融合 GEMM 软件流水（寄存器预取双缓冲）

**改动**（`tfp8/csrc/tfp8_kernels.cu` 的 `gemm_w8a16`）：
- sA/sB 改为两级 stage。第 k 步先把 K 块 k+1 的全局加载发到寄存器（2×A uint4 + 1×B uint4），然后在 stage k&1 上做 WMMA，再把预取的块解码写进 stage (k+1)&1，**每步只用一次 `__syncthreads`**（原来是两次，而且加载和计算串行）。stage (k+1)&1 上一次被读是在第 k−1 步，那一步以 barrier 结束，所以写入是安全的。
- 组边界 spill 缓冲 `sT` 从 32×32 改为 16×32 / warp（分两次 spill），每 block smem 29 KB。**第一版没改 sT，smem 37 KB → 每 SM 只能驻留 1 个 block，饱和 grid 上反而 0.89–0.99×**；缩小 sT 恢复每 SM 2 个 block 后全面转正。
- 运算的操作数、MMA 顺序和组累加顺序都与原 kernel 相同 → **输出与旧 kernel 逐位一致**（A/B 中每点都用 `torch.equal` 核对）。
- 删除 BM=128 试验 kernel（`tfp8_gemm128_*`、`gemm_prepared(wide=)`、对应门禁）。新门禁 `test_gemm_pipeline_edges`：K=32（单步，无预取）/64/96/160 × G=None/32（每步都是组边界）× M=9/64/65/130，对照 FP64 参考。门禁 **17 项全 PASS**。

**交替 A/B**（`.verify/bench/gpu/ab_gemm_pipeline.py`：旧 kernel 从快照 `tfp8_kernels_prepipe.cu` 在同一进程内 NVRTC 编译，15 轮中位，不含 prep）：

| 形状 | M=16 | 64 | 128 | 256 | 512 | 2048 | M=2048 TF（旧→新） |
|---|---:|---:|---:|---:|---:|---:|---:|
| o 4096² g128 | 1.44× | 1.70× | 1.33× | 1.41× | 1.45× | 1.40× | 12.4 → **17.4** |
| o 4096² perchan | 1.70× | 1.84× | 1.61× | 1.71× | 1.68× | 1.69× | 13.4 → **22.8** |
| gate 11008×4096 g128 | 1.35× | 1.56× | 1.59× | 1.59× | 1.56× | 1.50× | 12.0 → **18.1** |
| DiT 4608×1152 perchan | 1.17× | 1.41× | 1.52× | 1.59× | 1.51× | 1.59× | 13.1 → **20.9** |

**生产基准**（`bench_transcode.py --only gemm`，run12，4096² per-channel，含 prep）：

| M | fused（r11 → **r12**） | chunked r12 | cuBLAS |
|---:|---:|---:|---:|
| 64 | 7.0 → **12.1 TF** | 5.0 | 20.9 |
| 128 | 13.4 → **21.1 TF** | 9.3 | 41.6 |
| 256 | 13.5 → **21.5 TF** | 21.3 | 55.1 |
| 512 | 13.3 → **21.4 TF** | 39.7 | 74.4 |
| 2048 | 13.3 → **21.2 TF** | 53.9 | 76.3 |

- 融合 GEMM 的平台期从 ~13 TF 提到 **~21 TF（per-channel）/ ~17–18 TF（g128）**。g128 每 4 个 K 步要 spill 一次组累加，比 per-channel 慢约 25%，这是下一个可以优化的点（例如把组 scale 预乘进解码后的 B，就不用 spill）—— 已在 §12.6 实现，差距从 25% 缩到约 12%。
- fused 与 chunked 的交点仍在 M≈256（21.5 vs 21.3 TF），默认 `large_m=256` 不用改；M=64–255 走 fused，比改前快 1.4–1.8×。
- 端到端：`demo_image`（64px、dim 512、depth 8，全程走 chunked）PSNR 50.98 dB / SSIM 0.9946、0 次回退；`demo_llm --size tiny` logits cosine 0.99734。

**产物**：`.verify/bench/gpu/ab_gemm_pipeline.py` / `.json` / `.log`、`tfp8_kernels_prepipe.cu`（A/B 基线快照）、`transcode_gpu_run12_gemm.json`、`run12_gemm.log`。

### 12.6 2026-11-08 晚：去掉 g128 的组 spill（组 scale 折进解码后的权重 tile）

**定位**（`.verify/bench/gpu/diag_g128_gap.py`）：同一 kernel、同一份码字，只改 G：g128 码字 G=128 是 1034 µs，G=K 是 788 µs；per-channel 码字 975 / 782 µs。两种码字都没有落到慢解码路径（exp==0 字节 0.01%）。**所以差距来自 G（组边界处理），与码字无关。**

**改动**（`gemm_w8a16`）：
- 每列分解 ws[n,g] = smax[n]·r[n,g]，smax = 该行各组 scale 的最大值，r ≤ 1。r 在解码写 smem 时乘进 B tile（fp32 乘、一次 fp16 舍入），smax[n]、激活行 scale 和 bias 在 epilogue 里只做一次。主循环不再 spill，`racc[32]` 寄存器也删掉了。
- **per-channel（单组）不做折叠**，epilogue 表达式与旧 kernel 相同 → A/B 中每点都 bit-exact。
- 第一版 g128 只有 0.89–1.03×：stash 里那次 `ws` 全局读没有预取，排在 MMA 之后，再加一次 IEEE 除法，正好落在关键路径上。改为在 `fetch` 里与权重一起预取 scale，并在每个 block 预先算 1/smax 后，g128 达到 **1.04–1.12×**。
- 又试了一版：r 是 fp16 正规范围内的 2 的幂时，用 `__hmul2` 代替 fp32 乘。结果逐位相同，但速度没有差别（0.88–1.08×，噪声），已撤回以保持代码简单。剩余约 12% 的 g128/per-channel 差距不在折叠算术上。

**数值**：
- 2 的幂 scale（`quantize_weight` 默认）下折叠是精确的。`diag_fold.py` 显示：组 scale 跨度 2^0…2^-20 时，折叠与 spill kernel 的误差完全一致（误差/行最大值 2.2–3.6e-4，由 fp16 输出主导）；FP32 输出时折叠误差约 2e-6。
- 非 2 的幂 scale（checkpoint 兼容模式）：每个折叠后的权重多一次 fp16 舍入（相对 2^-12，与 chunked 的 dequant 精度相同，比 FP8 自身量化步长细 256×），误差/行最大值 1.3–1.8e-4。逐元素相对误差在近零输出上会放大（抵消），所以门禁用行最大值口径。
- 新门禁 `test_gemm_folded_group_scales`（e4m3fn/e5m2 × 2 的幂 / 非 2 的幂 × M=16/100，组 scale 跨度 2^-20）。**18 项全 PASS**。

**顺带修复**：`_ext._gemm_launch` 用的是 `"f16" if out_dtype == float16 else "bf16"`，**`out_dtype=float32` 的 fused GEMM 由 bf16 kernel 往 float32 缓冲里写，输出是垃圾**。新增 `tfp8_gemm_f32` 入口；`test_kernel_group_scales_formats_dtypes` 的输出 dtype 加上 float32。

**交替 A/B**（`AB_BASE=tfp8_kernels_spill.cu AB_OUT=ab_gemm_fold`，基线为 §12.5 的双缓冲 spill kernel）：

| 形状 | M=16 | 64 | 128 | 256 | 512 | 2048 | M=512 TF（旧→新） |
|---|---:|---:|---:|---:|---:|---:|---:|
| o 4096² g128 | 1.06× | 1.04× | 1.11× | 1.11× | 1.12× | 1.08× | 17.5 → **19.6** |
| gate 11008×4096 g128 | 1.12× | 1.06× | 1.07× | 1.07× | 1.10× | 1.09× | 16.8 → **18.5** |
| o 4096² perchan（bit-exact） | 0.91× | 1.00× | 0.96× | 0.99× | 0.98× | 1.00× | 22.0 → 21.5 |
| DiT 4608×1152 perchan（bit-exact） | 1.21× | 0.91× | 0.93× | 1.01× | 1.03× | 1.05× | 18.7 → 19.3 |

per-channel 走的代码路径不变，±10% 是桌面争用下的噪声（同一点在不同轮次之间也有这么大的抖动）。生产基准 run13（4096² per-channel）fused 20.5–20.9 TF、chunked M=2048 54.3 TF，与 run12 持平。

**产物**：`ab_gemm_fold.json/.log`、`ab_gemm_fold_hmul.json/.log`（撤回的 hmul 试验）、`diag_g128_gap.py`、`diag_fold.py`、`tfp8_kernels_spill.cu` / `tfp8_kernels_fold_f32.cu`（A/B 基线快照）、`transcode_gpu_run13_gemm.json`、`run13_gemm.log`。

### 12.7 run10/11 小形状 GEMV「退化」排查：不是代码回归，是提交地板 / 桌面争用

**背景**：复测分析指出 M=1 小形状 GEMV 比之前慢：5632×2048 64.9/69.0 → 78.1/107.7 µs，4608×1152 70/74 → 78/96 µs，而且两轮之间也不稳定，怀疑 16:38/17:02 的改动引入了退化。**归因更正**：那两批改动（§12.4–12.6：chunked epilogue、GEMM 双缓冲、g128 折叠）是本会话做的，不是外部并行会话；它们都没有碰 GEMV。

**1. 代码层面**：当前 `tfp8_kernels.cu` 的 GEMV 段和 decode 辅助函数与本会话开始前的源码快照（`tfp8_kernels_orig_1108.cu`）**逐字节相同**。`w8a16` 的 M≤8 路径未改；`_launch` 唯一的改动是 float 槽位的类型，而 GEMV 没有 float 参数。

**2. 同进程交替 A/B**（`.verify/bench/gpu/ab_gemv_regress.py`，原始模块与当前模块各自 NVRTC 编译，11 轮中位，M=1 bf16）：

| 形状 | graph 原始 | graph 当前 | graph cuBLAS fp16 | loop 原始 | loop 当前 | loop cuBLAS | loop FP8Linear.forward | 当前 loop 轮间范围 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12288×4096 | 221.6 | 223.1 | 215.5 | 220.8 | 224.8 | 226.5 | 226.0 | 201.7–367.5 |
| 11008×4096 | 195.8 | 204.4 | 204.3 | 193.7 | 208.1 | 212.1 | 203.3 | 188.7–220.3 |
| 4096×4096 | 82.6 | 82.6 | 81.0 | 82.5 | 87.9 | 95.3 | 126.6 | 81.4–122.5 |
| 4096×11008 | 208.9 | 203.9 | 209.9 | 210.9 | 215.1 | 210.6 | 218.9 | 198.5–222.4 |
| 5632×2048 | 64.7 | 63.9 | 60.1 | 75.5 | 74.8 | 75.1 | 124.7 | 64.1–118.3 |
| 4608×1152 | 45.0 | 31.3 | 38.7 | 65.1 | 62.7 | 99.5 | 164.3 | 51.3–97.6 |

（µs；graph = 200 次启动录成 CUDA graph 回放，基本是纯 kernel 时间；loop = 200 次 `_launch` 背靠背，即 bench_transcode GEMV 表的口径。原始与当前输出全部 bit-exact。）

- **纯 kernel 时间原始 = 当前**（差异在噪声内，4608×1152 当前反而更快）→ **没有代码回归**。
- 单轮内当前 kernel 的 loop 时间就在 64–118 µs（5632×2048）、51–98 µs（4608×1152）之间摆动，**run10 与 run11 的 78.1 vs 107.7 µs 就落在这个区间里**。fp16 cuBLAS 也一样（4608×1152 graph 38.7 µs，loop 99.5 µs）。

**3. 提交地板**（同一时刻，wall 时间、异步启动，5 轮）：1 个元素的 `torch add_` **36–56 µs/次**（平时是个位数 µs）；`_ext.w8a16` GEMV 74–114 µs/次（kernel 本身只有 31–45 µs）；`FP8Linear.forward` 131–148 µs/次。当时 `nvidia-smi` 列出的 GPU 客户端有 Wallpaper Engine、GamePP、NVIDIA Overlay、多个 WebView 等。**结论：小形状 GEMV 是受提交 / WDDM 争用限制，不是受 kernel 限制**。kernel 时间 <70 µs 的层，loop 口径测到的主要是这个地板，比值和绝对值都会随桌面负载摆动。

**对后续讨论的影响**：
- 「解除 ALU 瓶颈」应该用 graph 口径（纯 kernel）判断。按 graph 口径，大形状 GEMV 与 fp16 cuBLAS 打平（12288×4096 223 vs 216 µs；11008×4096 204 vs 204 µs），即约 0.97–1.03×；本轮的改动确实没动 GEMV。
- 真实存在、可以优化的是 `FP8Linear.forward` 的 Python 开销：小层上 forward 比裸 `w8a16` 多约 45–100 µs（4608×1152：164 vs 63 µs loop）。对 DiT 这类几百个小层的模型，这比 kernel 本身更重要。单独开一项处理，不应和 GEMV kernel 混在一起。
- 另外更正一点：fused GEMM 现在 21.5 TF ≈ cuBLAS 的 28%，**不是**「每 64 行重解码整层权重」造成的。§12.4.3 的 BM=128（重解码减半）实测反而更慢，§12.5 的双缓冲在同样的重解码量下从 13 提到 21 TF；剩余差距在 WMMA 主循环的效率（BK=32、每步一次 barrier、ldmatrix/寄存器复用不足）。

**产物**：`ab_gemv_regress.py` / `.json` / `.log`、`tfp8_kernels_orig_1108.cu`（会话前源码快照）。

### 12.8 2026-11-08 晚：砍 `FP8Linear.forward` 小层开销（GEMV 路径）+ 修 fp32 输入 GEMV 的错误输出

**开销拆解**（M=1，4608×1152，当时桌面争用较重，CPU 侧数字整体偏高）：`cuLaunchKernel` 约 58 µs；Python 包装从裸 launch 到 `L(x)` 共约 60 µs，其中 `torch.empty` 约 7、`reshape` 每次约 4（输入、输出各一次）、`_launch` 每次逐参数构造类型签名约 3.7、`nn.Module` buffer 属性读取每次约 0.8、`_lut_for` 约 1.8、`w8a16` 里每次新建 tag dict 等。

**改动**：
- `_ext.GemvLaunch`：每层、每个（输入 dtype, 输出 dtype）组合缓存一份**预构建的 ctypes 参数块**。每次调用只写 x 指针、y 指针、M 三个值，然后直接调 `cuLaunchKernel`；静态张量由对象持有引用，不会被释放。
- `FP8Linear._gemv_launch`：用 `_buffers` 的身份比较（dict 查找）校验缓存。buffer 对象被替换（`.to()` / `.cuda()` / 直接赋值）或 out_dtype 变化时重建；`load_state_dict` 是原地 `copy_`，指针不变。输入不在本层设备、当前设备不同、dtype 不支持时走通用路径。
- `forward` 快路径：contiguous 的 CUDA 输入、M ≤ 8、前 3 次 NaN 检查之后，**不做 reshape**（kernel 只需要数据指针和 M），y 直接按最终形状分配；同时校验 `shape[-1] == in_features`（旧路径没有这个检查）。
- `_launch` 新增 `kinds=`，内部调用（prep / dequant / epilogue / GEMM / GEMV）传常量签名，每次省约 3.7 µs。

**顺带发现并修复的 bug**：GEMV 的激活暂存用 `VEC = 16/sizeof(XT) < 8 ? 8 : 16/sizeof(XT)`，对 fp32 输入 VEC 被钳到 8，而一次 128-bit 加载只有 4 个 float → **一半的 x 没写进 shared memory，kernel 读到未初始化内存**。fp32 输入的 GEMV 输出错误（误差为行最大值的 68%）且不确定。这个 bug 在本会话之前就存在（§12.7 证明 GEMV 源码未变），之前的门禁只用 bf16 输入，所以没发现。已改为 `VEC = 16/sizeof(XT)`。

**门禁 19 项全 PASS**。新增 `test_gemv_prepared_launch_tracks_layer_state`：快路径与通用路径逐位一致；f16/bf16/f32 输入 × M=1/3/8 都对照 FP64 参考（覆盖 fp32 bug）；N-d 输入、换 out_dtype、替换 buffer、`load_state_dict`、CPU 往返后都不会留下失效指针。

**交替 A/B**（`.verify/bench/gpu/ab_forward_overhead.py`，old = 旧 forward + 旧 w8a16 GEMV 分支的逐字副本，9 轮中位）：

| 形状（M=1，bf16） | 纯 Python 开销 2-D（launch 换成 no-op） | 纯 Python 3-D | 实际 wall 2-D | 实际 wall 3-D |
|---|---:|---:|---:|---:|
| 4608×1152 | 72.8 → **34.3 µs** | 73.2 → **40.7** | 121.2 → **94.8** | 123.7 → **84.3** |
| 5632×2048 | 67.2 → **36.6** | 72.7 → **34.8** | 112.6 → **84.9** | 127.0 → **89.9** |
| 4096×4096 | 72.1 → **39.4** | 72.6 → **36.4** | 102.3 → **81.5** | 113.5 → **87.8** |

- 每次调用的主机侧开销约减半（**−31 到 −38 µs**），实际 wall 每次省 **21–40 µs**。纯 Python 口径与 GPU 争用无关，最可靠；wall 口径里的 cuLaunchKernel 本身仍然被争用放大到约 50 µs。
- 剩下约 35 µs 中，`nn.Module.__call__` 约 8、`torch.empty` 约 7（争用下），其余是 ctypes 调用本身。再往下压需要 CUDA graph 或 C 扩展，超出本项范围。
- `demo_llm --size tiny`：logits cosine 0.997340 不变；decode 相对 bf16 两次分别为 0.83× / 1.18×。这个 tiny 模型每 token 11–13 ms，被注意力与 Python 主导、噪声极大，**本项改进在 demo 级别无法分辨**，以上面的受控 A/B 为准。

**产物**：`ab_forward_overhead.py` / `.json` / `.log`。

### 12.9 2026-11-08 18:1x：当前代码再次全量复测（run14 / run15）

**代码状态**：`tfp8/_ext.py` / `tfp8/linear.py` 17:55、`tfp8/csrc/tfp8_kernels.cu` 17:52 —— §12.4–12.8 的改动全部落地。`tests/test_tfp8.py` **19 项门禁全 PASS**（`CUDA kernels tested`，TESTS_EXIT=0）。两轮全量 `python -I tests/bench_transcode.py`（18:18:33–18:20:17，各约 52 s）均正常退出，decode/encode/gemm 的 bit-exact 检查全 True；GPU 仍默认 **300 W**，SM boost 2010 MHz、功耗 120–121 W，起点 util 31–36%（桌面有争用）。

**对照口径**：run9（14:2x，grid 修复后首测）是唯一保留的 14:2x 基线（run10 已被 16:59 的重跑覆盖）；run10/run11（16:59/17:00，同一轮优化）作中期参考。本轮所有数字系统性偏低 4–7%（同代码 cuBLAS fp16 1871 vs 1763 µs；run15 起点时钟 540 MHz），按 cuBLAS 归一后与 run10/11 持平——属桌面争用，不是代码回归。

**GEMM 4096²（双轮中位；ratio 为相对 14:2x 基线）**：

| M | fused 旧→新 (µs) | fused TF 旧→新 | chunked 旧→新 (µs) | chunked TF 旧→新 | cuBLAS fp16 (µs) |
|---:|---:|---:|---:|---:|---:|
| 32 | 261.8 → 180.9 | 4.10 → 5.94 | 657.5 → 334.4 | 1.63 → 3.21 | 90.5 |
| 64 | 295.8 → 181.4 | 7.26 → 11.84 | 518.6 → 344.4 | 4.14 → 6.23 | 91.3 |
| 128 | 309.8 → 221.7 | 13.86 → 19.37 | 604.3 → 381.5 | 7.11 → 11.26 | 94.7 |
| 256 | 634.1 → 424.2 | 13.55 → **20.25** | 636.3 → 390.2 | 13.50 → 22.01 | 166.5 |
| 512 | 1260.0 → 841.4 | 13.64 → **20.42** | 581.1 → 442.4 | 29.57 → 38.83 | 233.6 |
| 1024 | 2537.4 → 1655.3 | 13.54 → **20.76** | 879.0 → 714.1 | 39.09 → 48.11 | 462.4 |
| **2048** | 5194.7 → **3258.2** | 13.23 → **21.09** | 1606.0 → **1255.1** | 42.79 → **54.75** | 927.8 |

**chunked 分解 M=2048（µs）**：prep 118.8 → 111.0；dequant 130.2 → 134.9（占比 8.1% → 10.7%）；mm 959.8 → 959.6；**epilogue 325.0 → 127.1（−61%）**；**cast 121.8 → 0（并入 epilogue）**；total 1606.0 → 1255.1（**−21.9%**）。

**GEMV M=1（µs，tfp8 / 同轮 fp16 cuBLAS）**：12288×4096 222.4（0.98×）、11008×4096 205.9（1.00×）、4096² 87.0（1.04×）、4096×11008 213.7（1.03×）、5632×2048 78.2（1.01×）、4608×1152 67.4（0.96×）。大形状与 14:2x 持平、与 fp16 cuBLAS 打平；小形状（5632×2048、4608×1152）比 14:2x 快 26–32%，来自 §12.8 的 forward 开销优化。

**转译吞吐（本轮双轮中位）**：解码 micro 45.1 M 361.0 µs = 124.9 Gelem/s = 374.7 GB/s；生产 dequant qkv g128 361.5 µs = 124.7 Gelem/s = 374.2 GB/s（perchan 356.4 / 379.6，e5m2 358.1 / 377.7），o 4096² 133.1（378.3）、1b 91.8（377.1）、dit 47.0（339.1）；编码 micro f16→e4m3 323.5 µs = 139.4 Gelem/s = 418.1 GB/s；`quantize_weight` 10.2–10.3 Gelem/s；可流带宽 copy u8 381.1 GB/s、torch fill 457.3、sum 433.9。**与 14:2x 相比全部在 ±5–7% 内（争用方向），转译本身仍是带宽受限、无结构性变化。**

**结论**：§12.4–12.8 的改动把它们的目标都兑现了——fused 平台期 **13.2 → 21.1 TF（+59%）**、chunked **42.8 → 54.75 TF（+28%）**、chunked 端到端 **−21.9%**；转译吞吐不变（374–418 GB/s，贴着 381–457 GB/s 的可流带宽），GEMV 与 fp16 cuBLAS 打平（0.96–1.04×）。绝对值受共享桌面 ±5–7% 影响，结论看比值不看单次绝对值。

**产物**：`transcode_gpu_run14.json` / `transcode_gpu_run15.json`、`run14.log` / `run15.log`、`compare_run14b.py`、`run14_run15_compare.json` / `.log`。
