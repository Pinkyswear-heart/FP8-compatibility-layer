# benchlab — 基准测试页面

一个本地网页应用：在本机跑一次 FP8 转译基准，存成一条按显卡归档的记录，然后把任意多条记录
并排对比。没有构建步骤，没有第三方依赖，只用 Python 标准库和一张 HTML 页面。

```bash
python -m benchlab            # http://127.0.0.1:8770
python -m benchlab --open     # 顺便打开浏览器
```

Windows 上如果 `python` 不在 PATH，用完整路径；页面只绑定 `127.0.0.1`，因为 `/api/run`
会启动子进程。

## 它管理什么

每条记录是一个 `tests/bench_transcode.py` 的输出加一层元数据，存在

```
benchlab/records/<gpu-slug>/<yyyymmddThhmmss>.json
```

`benchlab/records/` 在 `.gitignore` 里 —— 记录是本机的测量，不是仓库内容。要分享，把那个
JSON 文件发给对方，对方拖进页面的「导入」面板即可。

## 为什么会需要它

`tests/bench_transcode.py` 每次跑完都覆盖同一个 JSON，于是「这张卡现在比上周快还是慢」
根本无从回答。这个页面把每次运行留档，并且固定以**第一条选中的记录**为基线，
所以「11-13 比 11-08 慢了多少」和「B 卡比 A 卡快多少」是同一个操作。

## 哪些指标参与跨卡排序

只有每张卡都必然能产出的指标才进对比图：

| 指标 | 单位 | 含义 |
|---|---|---|
| 解码 | Gelem/s | 45.1M 权重，读 1 B 写 2 B |
| 编码 | Gelem/s | 45.1M 元素 |
| 拷贝带宽 | GB/s | 256 MB u8 拷贝，双程计 |
| cuBLAS FP16 | TFLOPS | 卡本身的算力，用来把「转译慢」和「卡慢」分开 |
| 融合 GEMM | TFLOPS | 解码放进 WMMA 主循环 |
| chunked GEMM | TFLOPS | 先反量化再交给 cuBLAS |
| 提交地板 | µs | 空 kernel 一轮，含驱动开销 |

GEMV 不在其中：它的绝对时间依赖层形状，而且在这张卡上被 Windows/WDDM 约 28 µs 的提交地板
支配，跨卡排序等于排驱动版本。GEMV 的逐形状数据保留在记录详情里，只是不参与排名。

## 跨卡对比需要什么

页面本身不需要 GPU 也能跑 —— 它只是读 JSON。但**别的显卡的数据必须在那台机器上测出来**：

```bash
# 在被对比的那台机器上
python -I tests/bench_transcode.py --json my_gpu.json
```

然后把 `my_gpu.json` 拖进页面。没有任何一张卡的真实数据时，页面比较的是「同一张卡在两个
时间点」。

## 代码结构

| 文件 | 职责 |
|---|---|
| `server.py` | HTTP 路由、静态文件、API。仅标准库 |
| `runner.py` | 起子进程跑 `tests/bench_transcode.py`，流式收集日志，把结果落盘 |
| `store.py` | 记录格式、指标抽取、对比算术。不 import torch |
| `static/` | 页面：`index.html`、`app.js`（手写 SVG 图表）、`style.css` |

测量本身完全不在这里 —— 它仍然归 `tests/bench_transcode.py` 所有，这样报告里的数字和页面上
的数字永远是同一段代码产出的。
