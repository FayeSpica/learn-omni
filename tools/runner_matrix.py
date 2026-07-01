#!/usr/bin/env python3
"""
runner_matrix.py — 生成 model_runner 覆盖矩阵(L1)。

把 vllm / vllm-ascend / vllm-omni 四个 runner 的方法集合抽出来,输出一张
Markdown 覆盖矩阵:每个方法在每个 runner 里是「直接 override」还是「继承」。
只输出**存在分叉**的方法(至少被一个子类 override),纯继承的方法折叠计数。

- 结构(override / inherit)是 AST 自动抽的,永不腐烂。
- 语义(⚠️ 分叉、❌ 缺兜底)是**人工**在 index.md 里补的——AST 看不出行为差异。

用法:
    OMNI_SRC=~/git/vllm_omni python3 tools/runner_matrix.py            # 打印到 stdout
    OMNI_SRC=~/git/vllm_omni python3 tools/runner_matrix.py > docs/vllm-omni/runner-compare/_matrix.generated.md

基线 SHA 会自动从各 repo 的 git HEAD 读取,写进输出头部。
"""
from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

SRC = Path(os.environ.get("OMNI_SRC", Path.home() / "git" / "vllm_omni")).expanduser()

# 四(+2 阶段)个 runner。顺序即矩阵列顺序:上游 → omni GPU → ascend NPU → omni NPU → 两个阶段子类。
RUNNERS = [
    # key,          列标题,             repo,          相对路径,                                                   类名
    ("vllm_gpu",    "vllm GPU",         "vllm",        "vllm/v1/worker/gpu_model_runner.py",                       "GPUModelRunner"),
    ("omni_gpu",    "omni GPU",         "vllm-omni",   "vllm_omni/worker/gpu_model_runner.py",                     "OmniGPUModelRunner"),
    ("ascend_npu",  "ascend NPU",       "vllm-ascend", "vllm_ascend/worker/model_runner_v1.py",                    "NPUModelRunner"),
    ("omni_npu",    "omni NPU",         "vllm-omni",   "vllm_omni/platforms/npu/worker/npu_model_runner.py",       "OmniNPUModelRunner"),
    ("omni_npu_ar", "omni NPU·AR",      "vllm-omni",   "vllm_omni/platforms/npu/worker/npu_ar_model_runner.py",    "NPUARModelRunner"),
    ("omni_npu_gen","omni NPU·Gen",     "vllm-omni",   "vllm_omni/platforms/npu/worker/npu_generation_model_runner.py", "NPUGenerationModelRunner"),
]

# 已知的继承边(子 -> 父),用于把「本类没定义」区分为「继承自我们集合内的父类」还是「不在链上」。
PARENTS = {
    "omni_gpu":     ["vllm_gpu"],
    "ascend_npu":   ["vllm_gpu"],
    "omni_npu":     ["omni_gpu", "ascend_npu"],   # 菱形
    "omni_npu_ar":  ["omni_npu"],
    "omni_npu_gen": ["omni_npu"],
}

MARK_OVERRIDE = "🔧"   # 本类直接定义
MARK_INHERIT = "⬆️"    # 继承自集合内某个父类
MARK_ABSENT = "·"      # 该列继承链上无人定义


def sha(repo: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(SRC / repo), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "?"


def methods_of(path: Path, classname: str) -> set[str]:
    """抽 classname 直接定义的方法名(含 async)。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"<!-- 解析失败 {path}: {e} -->")
        return set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == classname:
            return {
                b.name for b in node.body
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not b.name.startswith("__")
            }
    print(f"<!-- 未找到 class {classname} in {path} -->")
    return set()


def ancestors(key: str) -> list[str]:
    out, stack = [], list(PARENTS.get(key, []))
    while stack:
        k = stack.pop(0)
        if k not in out:
            out.append(k)
            stack.extend(PARENTS.get(k, []))
    return out


def cell(key: str, method: str, defined: dict[str, set[str]]) -> str:
    if method in defined[key]:
        return MARK_OVERRIDE
    if any(method in defined[a] for a in ancestors(key)):
        return MARK_INHERIT
    return MARK_ABSENT


def main() -> None:
    defined = {k: methods_of(SRC / repo / rel, cls) for k, _, repo, rel, cls in RUNNERS}
    keys = [k for k, *_ in RUNNERS]
    all_methods = sorted(set().union(*defined.values()))

    # 只保留「有分叉」的方法:被某个子类(非 vllm_gpu 基准)直接 override。
    subclass_defined = set().union(*(defined[k] for k in keys if k != "vllm_gpu"))
    diverging = [m for m in all_methods if m in subclass_defined]
    folded = len(all_methods) - len(diverging)

    repos = {}
    for _, _, repo, *_ in RUNNERS:
        repos.setdefault(repo, sha(repo))

    print("<!-- 本文件由 tools/runner_matrix.py 生成,勿手改;语义标注写在 index.md -->")
    print()
    print("!!! info \"对齐基线(自动读取 git HEAD)\"")
    for repo, s in repos.items():
        print(f"    - `{repo}` @ `{s}`")
    print()
    print(f"> 图例:{MARK_OVERRIDE} 本类直接 override · {MARK_INHERIT} 继承自集合内父类 · "
          f"{MARK_ABSENT} 继承链上无人定义。仅列出**存在 override 分叉**的方法;"
          f"另有 **{folded}** 个方法为纯继承,已折叠。")
    print()

    header = "| 方法 | " + " | ".join(lbl for _, lbl, *_ in RUNNERS) + " |"
    sep = "|---|" + "|".join([":---:"] * len(RUNNERS)) + "|"
    print(header)
    print(sep)
    for m in diverging:
        row = " | ".join(cell(k, m, defined) for k in keys)
        print(f"| `{m}` | {row} |")


if __name__ == "__main__":
    main()
