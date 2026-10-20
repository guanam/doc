# -*- coding: utf-8 -*-
"""npugraph_ex 自捕获路径的「入参普查 + 地址稳定性 + 图池归因」探针。

专治三个问题:
  T1 入参普查   —— 哪些入参被 clone_input 克隆了, 各多大
  T2 地址稳定性 —— 每步每个入参的 data_ptr 变不变(机制 A 的直接判据)
  T5 图池归因   —— 每个 FX 图的私有池吃了多少, 其中多少是克隆

用法(在任何 torch.compile 之前):
    import ex_input_probe; ex_input_probe.install()
跑完若干步后(进程退出时也会自动打一次):
    ex_input_probe.report()

环境变量:
    EX_PROBE_STEPS      最多跟踪多少次 process_input, 默认 50(0 = 不限)
    EX_PROBE_TOPN       每个池打印的 top-N 分配栈, 默认 12
    EX_PROBE_NOSTACK    =1 则不开 _record_memory_history(省开销, 但没有栈归因)
    EX_PROBE_DUMP_DIR   设了就把 capture 前/后的 snapshot pickle 落到这个目录

输出一律 '## ' 前缀, 用 grep '^##' 取。
"""
import os
import sys
import atexit
import traceback
from collections import OrderedDict, defaultdict

_MAX_STEPS = int(os.environ.get("EX_PROBE_STEPS", "50"))
_TOPN = int(os.environ.get("EX_PROBE_TOPN", "12"))
_NOSTACK = os.environ.get("EX_PROBE_NOSTACK", "0") == "1"
_DUMP_DIR = os.environ.get("EX_PROBE_DUMP_DIR", "")

_census = OrderedDict()      # (fx, key) -> {idx: {...}}
_captures = OrderedDict()    # (fx, key) -> {pool, reserved_delta, ...}
_pools = OrderedDict()       # fx -> pool_id
_ptr_changes = defaultdict(lambda: defaultdict(int))   # (fx,key) -> idx -> 变化次数
_ptr_seen = defaultdict(lambda: defaultdict(set))      # (fx,key) -> idx -> 见过的 data_ptr 集合
_steps = defaultdict(int)
_installed = False
_dumped_before = False


def out(*a):
    print("##", *a)
    sys.stdout.flush()


def _gib(n):
    return (n or 0) / float(1 << 30)


def _mib(n):
    return (n or 0) / float(1 << 20)


def _try_dump(tag):
    if not _DUMP_DIR:
        return
    try:
        import torch_npu
        os.makedirs(_DUMP_DIR, exist_ok=True)
        p = os.path.join(_DUMP_DIR, f"{tag}.pickle")
        torch_npu.npu.memory._dump_snapshot(p)
        out(f"dumped snapshot {p}")
    except Exception as e:
        out(f"dump snapshot {tag} failed: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- T1 入参普查
def _census_one(self, graph_key, args):
    """在 capture 入口做一次普查: 每个 user input 的大小, 以及它会不会被克隆。"""
    import torch
    mapping = getattr(self, "_user_inputs_mapping", {}) or {}
    mutated = set(getattr(self, "_mutated_user_inputs", []) or [])
    params = set(getattr(self, "_parameter_user_inputs", []) or [])
    refout = set((getattr(self, "_userinput_ref_with_output", {}) or {}).keys())
    clone_on = bool(self.config.get("clone_input", True))

    rows = {}
    for name, idx in mapping.items():
        if idx >= len(args):
            continue
        a = args[idx]
        if not isinstance(a, torch.Tensor):
            rows[idx] = dict(name=name, kind="non-tensor", nbytes=0, cloned=False,
                             shape=None, dtype=None, ptr=None)
            continue
        nbytes = a.numel() * a.element_size()
        is_npu = bool(getattr(a, "is_npu", False))
        # 复刻 acl_graph.capture 的筛选: :1346 跳 mutated, :1349 跳 ref_with_output, :1359 要 is_npu
        cloned = clone_on and is_npu and (name not in mutated) and (idx not in refout)
        if name in mutated:
            kind = "mutated(跳过克隆)"
        elif idx in refout:
            kind = "ref_with_output(跳过克隆)"
        elif not is_npu:
            kind = "非 NPU(跳过克隆)"
        else:
            kind = "普通输入"
        rows[idx] = dict(name=name, kind=kind, nbytes=nbytes, cloned=cloned,
                         shape=tuple(a.shape), dtype=str(a.dtype), ptr=a.data_ptr())
    # 参数不在 mapping 里(fx2acl_converter.py:429-437), 单独记个数就行
    rows["_meta"] = dict(n_params=len(params), n_mapping=len(mapping),
                         n_mutated=len(mutated), n_refout=len(refout), clone_on=clone_on)
    return rows


def install():
    global _installed
    if _installed:
        out("already installed")
        return
    try:
        from npugraph_ex._acl_concrete_graph.acl_graph import AclGraph
    except Exception as e:
        out(f"cannot import AclGraph (npugraph_ex 不在 sys.path?): {type(e).__name__}: {e}")
        return

    import torch
    import torch_npu

    if not _NOSTACK:
        try:
            torch_npu.npu.memory._record_memory_history(
                enabled="all", context="all", stacks="python", max_entries=1_000_000)
            out("memory history recording: ON (stacks=python)")
        except Exception as e:
            out(f"_record_memory_history failed, 继续但没有栈归因: {type(e).__name__}: {e}")

    _orig_capture = AclGraph.capture
    _orig_process = AclGraph.process_input

    def capture_probe(self, graph_key, *args, **kwargs):
        global _dumped_before
        fx = getattr(self, "name", "<noname>")
        k = (fx, str(graph_key))
        if not _dumped_before:
            _try_dump("before_first_capture")
            _dumped_before = True
        try:
            _census[k] = _census_one(self, graph_key, args)
        except Exception as e:
            out(f"census failed for {k}: {type(e).__name__}: {e}")
        r0, a0 = torch.npu.memory_reserved(), torch.npu.memory_allocated()
        try:
            return _orig_capture(self, graph_key, *args, **kwargs)
        finally:
            r1, a1 = torch.npu.memory_reserved(), torch.npu.memory_allocated()
            pool = getattr(self, "pool", None)
            _pools.setdefault(fx, pool)
            seq = len([x for x in _captures if x[0] == fx])
            _captures[k] = dict(pool=pool, reserved_delta=r1 - r0,
                                allocated_delta=a1 - a0, reserved_after=r1, seq=seq)
            out("capture fx=%s key=%s seq=%d reserved+%.3fGiB alloc+%.3fGiB now=%.3fGiB pool=%s"
                % (fx, graph_key, seq, _gib(r1 - r0), _gib(a1 - a0), _gib(r1), pool))

    def process_probe(self, graph_key, *args):
        """T2: 记录每个入参的 data_ptr 是否相对捕获期变过。"""
        fx = getattr(self, "name", "<noname>")
        k = (fx, str(graph_key))
        _steps[k] += 1
        if _MAX_STEPS == 0 or _steps[k] <= _MAX_STEPS:
            try:
                # 基线必须用「用户侧」的捕获期 data_ptr(普查时记的 rows[idx]["ptr"])。
                # 不能用 graphs_meta.userinputs_meta[idx].data_ptr —— clone_input=True 时
                # 那是克隆体的地址(acl_graph.py:1360 换完才在 :1374 记 meta), 拿它比永远不等。
                rows = _census.get(k, {})
                for idx, r in rows.items():
                    if idx == "_meta" or not isinstance(idx, int) or idx >= len(args):
                        continue
                    base = r.get("ptr")
                    if base is None:
                        continue
                    a = args[idx]
                    p = a.data_ptr() if hasattr(a, "data_ptr") else None
                    if p is None:
                        continue
                    _ptr_seen[k][idx].add(p)
                    if p != base:
                        _ptr_changes[k][idx] += 1
            except Exception as e:
                if _steps[k] == 1:
                    out(f"ptr tracking failed for {k}: {type(e).__name__}: {e}")
        return _orig_process(self, graph_key, *args)

    AclGraph.capture = capture_probe
    AclGraph.process_input = process_probe
    _installed = True
    atexit.register(_safe_report)
    out("installed: AclGraph.capture + AclGraph.process_input wrapped")


def _safe_report():
    try:
        report()
    except Exception:
        out("report failed:")
        traceback.print_exc()


# ------------------------------------------------------------- T5 池归因
def _is_clone_frame(f):
    if not isinstance(f, dict):
        return False
    return str(f.get("filename") or "").endswith("acl_graph.py") and f.get("name") == "capture"


def _frame_str(f):
    if not isinstance(f, dict):
        return str(f)
    fn = f.get("filename") or "?"
    return "%s:%s:%s" % (os.path.basename(str(fn)), f.get("line", "?"), f.get("name", "?"))


def _pool_digest():
    """全量快照 + 按 segment_pool_id 分组。装的 torch_npu 的 memory_snapshot() 不收 mempool_id,
    所以走 _snapshot() 再自己过滤。"""
    try:
        import torch_npu
        snap = torch_npu.npu.memory._snapshot()
    except Exception as e:
        out(f"_snapshot failed: {type(e).__name__}: {e}")
        return {}
    pools = defaultdict(lambda: dict(total=0, nseg=0, clone=0, clone_n=0,
                                     by_stack=defaultdict(int)))
    for seg in snap.get("segments", []) or []:
        pid = seg.get("segment_pool_id")
        pid = tuple(pid) if isinstance(pid, (list, tuple)) else pid
        d = pools[pid]
        d["total"] += seg.get("total_size", 0) or 0
        d["nseg"] += 1
        for blk in seg.get("blocks", []) or []:
            if "active" not in str(blk.get("state", "")):
                continue
            size = blk.get("size", 0) or 0
            frames = blk.get("frames") or []
            if any(_is_clone_frame(f) for f in frames):
                d["clone"] += size
                d["clone_n"] += 1
            key = " <- ".join(_frame_str(f) for f in frames[:3]) if frames else "<no frames>"
            d["by_stack"][key] += size
    return pools


def report():
    _try_dump("after_capture")

    out("=" * 92)
    out("T1 入参普查 —— 哪些入参被 clone_input 克隆了")
    out("=" * 92)
    total_cloned = 0
    for (fx, key), rows in _census.items():
        meta = rows.get("_meta", {})
        out("-" * 92)
        out("FX 图 %s  key=%s   clone_input=%s   mapping=%d params=%d mutated=%d ref_with_output=%d"
            % (fx, key, meta.get("clone_on"), meta.get("n_mapping", 0),
               meta.get("n_params", 0), meta.get("n_mutated", 0), meta.get("n_refout", 0)))
        out("  %-4s %-28s %12s %-22s %-24s %s"
            % ("idx", "name", "MiB", "shape", "kind", "会被克隆"))
        items = [(i, r) for i, r in rows.items() if i != "_meta"]
        for idx, r in sorted(items, key=lambda kv: -(kv[1]["nbytes"] or 0)):
            if r["cloned"]:
                total_cloned += r["nbytes"]
            out("  %-4s %-28s %12.2f %-22s %-24s %s"
                % (idx, str(r["name"])[:28], _mib(r["nbytes"]), str(r["shape"])[:22],
                   r["kind"], "YES <<<" if r["cloned"] else "-"))
    out("")
    out(">>> 所有 FX 图合计: 会被克隆的入参共 %.3f GiB (clone_input=False 能省掉的上限)"
        % _gib(total_cloned))

    out("")
    out("=" * 92)
    out("T2 地址稳定性 —— 机制 A 的直接判据")
    out("=" * 92)
    out("判读: 某入参 data_ptr 变过, 且它不在 captured_mutated_inputs 里 ->")
    out("      clone_input=False 时图会往捕获期旧地址写, 写出去的东西到不了新张量 -> 精度错。")
    danger = []
    for (fx, key), rows in _census.items():
        seen = _ptr_seen.get((fx, key), {})
        for idx, r in rows.items():
            if idx == "_meta" or not isinstance(idx, int):
                continue
            nptr = len(seen.get(idx, ()))
            if nptr == 0:
                continue
            n = _ptr_changes.get((fx, key), {}).get(idx, 0)
            moved = nptr > 1          # 真的换过地址
            is_mut = "mutated" in str(r.get("kind", ""))
            if not moved and n == 0:
                continue              # 地址恒定, 无风险, 不打
            out("  fx=%-10s key=%-14s idx=%-3s name=%-24s 不同地址=%-3d 偏离基线=%-4d %s %s"
                % (fx, key[:14], idx, str(r.get("name"))[:24], nptr, n,
                   "MOVED" if moved else "stable",
                   "(mutated, 地址变会触发重捕获)" if is_mut else ("!! 不在 mutated" if moved else "")))
            if moved and not is_mut:
                danger.append((fx, key, idx, r.get("name"), nptr))
    if not _ptr_changes:
        out("  (没采到 process_input。若用了 force_eager 则不会走这条路;否则确认跑了 replay 步)")
    out("")
    if danger:
        out(">>> !! 危险集合(地址变过且不在 mutated), 共 %d 个:" % len(danger))
        for fx, key, idx, nm, nptr in danger:
            out("     fx=%s idx=%s name=%s (见过 %d 个不同地址)" % (fx, idx, nm, nptr))
        out(">>> 机制 A 成立。clone_input=False 在这些入参上会丢写入。")
    else:
        out(">>> 没发现「地址真的换过且不在 mutated」的入参 -> 机制 A 在本次采样内不成立。")
    out("    注: 判据是「不同地址数 > 1」。偏离基线但只见过 1 个地址 = 调用方每步换了同一块,")
    out("        不构成机制 A 的风险。")

    out("")
    out("=" * 92)
    out("T5 图池归因")
    out("=" * 92)
    out("ex 的 FX 图 -> 池: %s" % ", ".join("%s=%s" % (k, v) for k, v in _pools.items()))
    pools = _pool_digest()
    ex_pools = {p for p in _pools.values() if p is not None}
    grand = 0.0
    for pid, d in sorted(pools.items(), key=lambda kv: -kv[1]["total"]):
        tag = " <<< ex 图池" if pid in ex_pools else ""
        out("-" * 92)
        out("pool %-12s segments=%-4d total=%.3f GiB%s" % (str(pid), d["nseg"], _gib(d["total"]), tag))
        if pid in ex_pools:
            grand += _gib(d["total"])
        if d["clone"]:
            out("   其中 clone_input 的克隆: %.3f GiB / %d 块" % (_gib(d["clone"]), d["clone_n"]))
        for stack, size in sorted(d["by_stack"].items(), key=lambda kv: -kv[1])[:_TOPN]:
            out("   %8.3f GiB  %s" % (_gib(size), stack))
    out("")
    out(">>> ex 图池合计 %.3f GiB, 池数 %d" % (grand, len(ex_pools)))
    out("    (不设 use_graph_pool 时每个 FX 图各开一个池: acl_graph.py:1006)")
    out("")
    out("    交叉校验(不依赖栈帧, 栈帧常常取不到):")
    out("    %-22s %14s %16s %14s" % ("fx_graph", "池 total GiB", "应克隆 GiB", "差额=激活 GiB"))
    for fx, pid in _pools.items():
        tot = _gib(pools.get(pid, {}).get("total", 0)) if pid in pools else 0.0
        exp = 0.0
        for (f2, k2), rows in _census.items():
            if f2 != fx:
                continue
            for idx, r in rows.items():
                if idx != "_meta" and r.get("cloned"):
                    exp += _gib(r.get("nbytes", 0))
            break      # 同一 FX 图取第一个 graph_key 即可(入参集合相同)
        out("    %-22s %14.3f %16.3f %14.3f" % (fx[:22], tot, exp, tot - exp))
    out("=" * 92)
