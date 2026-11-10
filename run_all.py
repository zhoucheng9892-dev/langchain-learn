"""批量回归：依次运行所有章节示例，输出通过/失败汇总。

用法：
    uv run python run_all.py            # 全部
    uv run python run_all.py ch07       # 只跑某章
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    scripts = sorted(ROOT.glob("chapters/ch*/[0-9]*.py"))
    if only:
        scripts = [s for s in scripts if only in str(s)]

    results = []
    for path in scripts:
        rel = path.relative_to(ROOT)
        t0 = time.time()
        proc = subprocess.run(
            [PY, str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        cost = time.time() - t0
        ok = proc.returncode == 0 and "Traceback" not in proc.stderr
        results.append((ok, str(rel), cost, proc.stderr.strip().splitlines()[-1] if not ok and proc.stderr.strip() else ""))
        print(f"{'PASS' if ok else 'FAIL'}  {str(rel):<52} {cost:5.1f}s")
        if not ok:
            print(f"      {results[-1][3][:160]}")

    total = len(results)
    passed = sum(1 for r in results if r[0])
    print()
    print(f"通过 {passed}/{total}，总耗时 {sum(r[2] for r in results):.0f}s")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
