# -*- coding: utf-8 -*-
"""ex 自捕获(非 force_eager)下的图池归因探针。

回答三个问题:
  1. 10G 是「单个 FX 图的池」还是「累计」?        -> 逐 (fx图, graph_key) 的 reserved 增量
  2. 这个池里到底是什么?                          -> memory_snapshot(pool_id) 按分配栈聚合
  3. clone_input=False 能省多少?                  -> 单独摘出 acl_graph.capture 的 empty_like

用法(在任何 torch.compile 之前):
    import ex_pool_probe; ex_pool_probe.install()
跑完若干步后(或进程退出时自动):
    ex_pool_probe.report()

环境变量:
    EX_POOL_PROBE_TOPN     每个池打印的 top-N 分配栈, 默认 12
    EX_POOL_PROBE_NOSTACK  设为 1 则不开 _record_memory_history(省开销, 但没有栈归因)

输出一律 '## ' 前缀, grep '^##' 即可。
"""
import os
import sys
import atexit
import traceback
from collections import OrderedDict, defaultdict

_TOPN = int(os.environ.get("EX_POOL_PROBE_TOPN", "12"))
_NOSTACK = os.environ.get("EX_POOL_PROBE_NOSTACK", "0") == "1"

# records[(fx_name, graph_key)] = dict(...)
_records = OrderedDict()
# pools[fx_name] = pool_id tuple
_pools = OrderedDict()
_installed = False


def out(*a):
    print("##", *a)
    sys.stdout.flush()


def _gib(n):
    return (n or 0) / float(1 << 30)


def _mem(torch_npu):
    import torch
    return torch.npu.memory_reserved(), torch.npu.memory_allocated()


def _snapshot(pool_id):
    """按池取快照。失败返回 None 并打印原因, 不影响主流程。"""
    try:
        import torch_npu
        return torch_npu.npu.memory_snapshot(pool_id)
    except Exception as e:
        out(f"snapshot failed for pool={pool_id}: {type(e).__name__}: {e}")
        return None


def _frame_str(f):
    if not isinstance(f, dict):
        return str(f)
    fn = f.get("filename") or f.get("file") or "?"
    return "%s:%s:%s" % (os.path.basename(str(fn)), f.get("line", "?"), f.get("name", "?"))


def _is_clone_frame(f):
    """acl_graph.py 的 capture() 里那次 torch.empty_like, 即 clone_input 的开销。"""
    if not isinstance(f, dict):
        return False
    fn = str(f.get("filename") or "")
    return fn.endswith("acl_graph.py") and f.get("name") == "capture"


def _digest(segments):
    """把快照聚合成: 总量 / 按栈的 top-N / clone 归因。字段名按 CUDA 快照同构, 全部 .get 防御。"""
    total = 0
    active = 0
    by_stack = defaultdict(int)
    clone_bytes = 0
    clone_n = 0
    nblocks = 0
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        total += seg.get("total_size", 0) or 0
        active += seg.get("active_size", seg.get("allocated_size", 0)) or 0
        for blk in seg.get("blocks", []) or []:
            if not isinstance(blk, dict):
                continue
            state = str(blk.get("state", ""))
            if "active" not in state:          # 只归因还活着的块
                continue
            size = blk.get("size", 0) or 0
            nblocks += 1
            frames = blk.get("frames") or []
            if any(_is_clone_frame(f) for f in frames):
                clone_bytes += size
                clone_n += 1
            key = " <- ".join(_frame_str(f) for f in frames[:3]) if frames else "<no frames>"
            by_stack[key] += size
    return {
        "total": total,
        "active": active,
        "nblocks": nblocks,
        "clone_bytes": clone_bytes,
        "clone_n": clone_n,
        "by_stack": by_stack,
    }


def install():
    global _installed
    if _installed:
        out("already installed")
        return
    try:
        from npugraph_ex._acl_concrete_graph.acl_graph import AclGraph
    except Exception as e:
        out(f"cannot import AclGraph, is npugraph_ex on sys.path? {type(e).__name__}: {e}")
        return

    import torch
    import torch_npu

    if not _NOSTACK:
        try:
            torch_npu.npu.memory._record_memory_history(
                enabled="all", context="all", stacks="python", max_entries=1_000_000
            )
            out("memory history recording: ON (stacks=python)")
        except Exception as e:
            out(f"_record_memory_history failed, continue without stacks: {type(e).__name__}: {e}")

    _orig_capture = AclGraph.capture

    def capture_probe(self, graph_key, *args, **kwargs):
        fx_name = getattr(self, "name", "<noname>")
        r_before, a_before = _mem(torch_npu)
        try:
            return _orig_capture(self, graph_key, *args, **kwargs)
        finally:
            r_after, a_after = _mem(torch_npu)
            pool_id = getattr(self, "pool", None)
            _pools.setdefault(fx_name, pool_id)
            seq = len([k for k in _records if k[0] == fx_name])
            rec = {
                "fx": fx_name,
                "key": str(graph_key),
                "seq_in_fx": seq,            # 0 = 该 FX 图的首次捕获
                "pool": pool_id,
                "reserved_delta": r_after - r_before,
                "allocated_delta": a_after - a_before,
                "reserved_after": r_after,
                "n_inputs_total": len(getattr(self, "_user_inputs_mapping", {}) or {}),
                "n_params": len(getattr(self, "_parameter_user_inputs", []) or []),
                "n_mutated": len(getattr(self, "_mutated_user_inputs", []) or []),
            }
            _records[(fx_name, str(graph_key))] = rec
            out(
                "capture fx=%s key=%s seq=%d  reserved+%.3fGiB  allocated+%.3fGiB  "
                "reserved_now=%.3fGiB  pool=%s  inputs=%d(params=%d, mutated=%d)"
                % (
                    fx_name, graph_key, seq,
                    _gib(rec["reserved_delta"]), _gib(rec["allocated_delta"]),
                    _gib(r_after), pool_id,
                    rec["n_inputs_total"], rec["n_params"], rec["n_mutated"],
                )
            )

    AclGraph.capture = capture_probe
    _installed = True
    atexit.register(_safe_report)
    out("installed: AclGraph.capture wrapped")


def _safe_report():
    try:
        report()
    except Exception:
        out("report failed:")
        traceback.print_exc()


def report():
    out("=" * 78)
    out("A. 逐 (FX 图, graph_key) 的捕获增量 —— 判定「单池 10G」还是「累计 10G」")
    out("=" * 78)
    out("%-22s %-14s %4s %12s %12s" % ("fx_graph", "graph_key", "seq", "reserved+GiB", "alloc+GiB"))
    per_fx_first = {}
    per_fx_rest = defaultdict(float)
    for (fx, key), r in _records.items():
        out("%-22s %-14s %4d %12.3f %12.3f"
            % (fx[:22], key[:14], r["seq_in_fx"], _gib(r["reserved_delta"]), _gib(r["allocated_delta"])))
        if r["seq_in_fx"] == 0:
            per_fx_first[fx] = _gib(r["reserved_delta"])
        else:
            per_fx_rest[fx] += _gib(r["reserved_delta"])

    out("")
    out("B. 按 FX 图汇总 —— 首次捕获 vs 其余 bs 档的增量")
    out("%-22s %14s %14s   %s" % ("fx_graph", "first+GiB", "others+GiB", "判读"))
    for fx in per_fx_first:
        first = per_fx_first[fx]
        rest = per_fx_rest.get(fx, 0.0)
        if first <= 0.01:
            verdict = "首次增量近 0, 池可能来自别处"
        elif rest < 0.2 * first:
            verdict = "intra-FX 复用生效(其余 bs 几乎免费)"
        else:
            verdict = "!! 其余 bs 也在涨, intra-FX 复用没生效"
        out("%-22s %14.3f %14.3f   %s" % (fx[:22], first, rest, verdict))

    out("")
    out("C. 每个 FX 图自己的池里有什么 —— memory_snapshot(pool_id) 归因")
    grand_total = 0.0
    grand_clone = 0.0
    for fx, pool_id in _pools.items():
        out("-" * 78)
        out(f"FX 图 {fx}  pool={pool_id}")
        if pool_id is None:
            out("  pool 为 None, 跳过(说明没走 use_graph_pool 且没取到句柄)")
            continue
        segs = _snapshot(pool_id)
        if segs is None:
            continue
        d = _digest(segs)
        grand_total += _gib(d["total"])
        grand_clone += _gib(d["clone_bytes"])
        out("  池 total=%.3fGiB  其中 active=%.3fGiB  活跃块数=%d"
            % (_gib(d["total"]), _gib(d["active"]), d["nblocks"]))
        out("  >>> clone_input 归因(acl_graph.capture 里的 empty_like): %.3fGiB / %d 块"
            % (_gib(d["clone_bytes"]), d["clone_n"]))
        if not d["by_stack"]:
            out("  (没有栈信息。确认 _record_memory_history 开了, 且 EX_POOL_PROBE_NOSTACK 没设)")
        for stack, size in sorted(d["by_stack"].items(), key=lambda kv: -kv[1])[:_TOPN]:
            out("   %8.3fGiB  %s" % (_gib(size), stack))

    out("")
    out("D. 总账")
    out("  所有 ex 图池合计 total = %.3f GiB" % grand_total)
    out("  其中 clone_input 的克隆   = %.3f GiB  <- clone_input=False 能省掉的上限" % grand_clone)
    out("  ex 池数量 = %d (每个 FX 图一个; 不设 use_graph_pool 时 acl_graph.py:1006 各开新池)" % len(_pools))
    out("")
    out("E. 下一步判据")
    out("  * 池数 == 模型数 且各池 total 相近 -> ×N 结构成立, 优先砍 ×N(use_graph_pool 共享)")
    out("  * C 里 top 栈是与 bs 无关的 buffer(如 EP dispatch/combine) -> use_graph_pool 划算")
    out("  * C 里 top 栈随 bs 线性 -> use_graph_pool 会赔(它强制关掉 intra-FX 复用), 改砍 bs 档位")
    out("  * D 的 clone 行 > 1GiB -> 直接上 clone_input=False")
    out("=" * 78)
