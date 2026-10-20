# doc

Ascend npugraph_ex 图池占用与多 bs 精度问题的定位任务书与探针。
目录结构与 gitcode 上那份 doc 仓对齐,便于后续增量同步。

## `docs/测试结果/ex单图占10G与多bs精度/`

把一个 MoE + 投机解码(多头 Eagle / MTP)的推理模型改写到 npugraph_ex 上,
**由 ex 自己做 compile + capture**,踩到两件事:

1. 编译一个图就吃掉约 10G device 内存,主模型 + 3 个 draft × 多 bs ⇒ OOM;
2. 调整 `clone_input` / `use_graph_pool` 之后,**多 bs 出现精度问题**。

> **边界**:以下分析只适用于 **ex 持有 capture** 的情形。社区 sglang 自己那条路是
> `force_eager=True` + 由 `NPUGraphRunner` 持有 capture,那种情况下 `clone_input` /
> `clone_output` 根本不生效 —— `acl_graph.py:920-923` 在 `compile()` 之前就 return 了。

### `定位任务书.md`

三条待验机制 + T0~T6 单变量实验 + 决策树:

| | 机制 | 判据 |
| --- | --- | --- |
| A | `clone_input=False` 碰上**未被识别的** input mutation | 存在"地址真的换过、且不在 `captured_mutated_inputs`"的入参 |
| B | `use_graph_pool` 跨 FX 图共享,与链式拓扑(主模型输出喂 draft)冲突 | 单改池就复现,单改 `clone_input` 不复现 |
| C | 图输出被后续 replay 覆盖 | `clone_output=True` 一开精度就恢复 |

**C 排最前**:上游代码自己写着 "The retained memory of acl graph output tensors will be
overwritten by subsequent executions, **which may cause precision issues**",并把
`clone_output=True` 列为解法之一(`acl_graph.py:1659-1672`)。

第二节汇总了八组已读码核实的事实(带 `文件:行`),供执行者直接引用而不必重新推导 ——
包括图池数量(每个 FX 图各开一个,`acl_graph.py:1006`)、克隆发生在 capture 上下文内因此
占图池、`clone_input` 两条路在 replay 期的语义差异、input mutation 识别的四道关卡及各自的漏法。

### 探针

- **`ex_input_probe.py`** —— 主探针,已在 910B3 冒烟。挂在 `AclGraph.capture` 和
  `AclGraph.process_input` 上,输出三张表:

  - **T1 入参普查**:逐入参列出大小,标出哪些会被 `clone_input` 克隆(复刻 `capture` 的筛选条件),
    末尾给出"合计 X GiB = `clone_input=False` 能省掉的上限"
  - **T2 地址稳定性**:每个入参见过几个不同的 `data_ptr`。判据是**不同地址数 > 1**
    且不在 `captured_mutated_inputs` 里 —— 机制 A 的直接判据
  - **T5 图池归因**:按 `segment_pool_id` 分组统计每个私有池的 total,并用
    `池 total / 应克隆 / 差额=激活` 做交叉校验(**不依赖**常常取不到的栈帧)

- **`ex_pool_probe.py`** —— 早一版,只做逐 (FX 图, graph_key) 的 `memory_reserved` 增量与池归因。
  `ex_input_probe.py` 是它的超集,一般用后者。

## 用法

在任何 `torch.compile` 之前插两行:

```python
import sys; sys.path.insert(0, "<本仓>/docs/测试结果/ex单图占10G与多bs精度")
import ex_input_probe; ex_input_probe.install()
```

进程退出时自动打报告,也可手动调 `ex_input_probe.report()`。输出一律 `## ` 前缀,`grep '^##'` 取。

环境变量:`EX_PROBE_STEPS`(跟踪多少步 `process_input`,默认 50)、
`EX_PROBE_DUMP_DIR`(落 capture 前/后的 memory snapshot pickle)、
`EX_PROBE_TOPN`、`EX_PROBE_NOSTACK=1`(不开 `_record_memory_history`)。

## 看内存快照

装的 `torch_npu` 的 `memory_snapshot()` **不接受 `mempool_id` 参数**(源码仓里有,装的包里没有),
所以探针走 `_snapshot()` 取全量再按 `segment["segment_pool_id"]` 过滤。

- `https://pytorch.org/memory_viz` —— 纯前端,拖进去即可,**不上传**。视图选
  `Allocated Memory (incl. Private Pools)`,私有池是灰色条并标 `Private Pool ((0,2)): capacity ...`;
  `Search stack frame...` 框里打 `acl_graph` 可单独筛出克隆
- `python3 -m torch.cuda._memory_viz {stats,segments,memory,trace,compare,trace_plot} *.pickle`
  —— 实测能吃 NPU 产的 pickle,其中 `compare before.pickle after.pickle` 最有用
- 直接 `pickle.load`,结构是 `{'segments': [...], 'device_traces': [...]}`
