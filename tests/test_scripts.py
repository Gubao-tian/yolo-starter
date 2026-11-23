"""CLI 入口脚本的端到端测试：真实子进程、真实临时目录。"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(script: str, *args: object) -> subprocess.CompletedProcess[str]:
    """在项目根目录以当前解释器运行 scripts/ 下的脚本。"""
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *[str(a) for a in args]],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
    )


class TestScripts(unittest.TestCase):
    def test_check_reports_demo_ready_and_full_hint(self) -> None:
        """check.py 必须给出 demo 可用与 full 依赖安装提示。"""
        process = run_script("check.py")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("demo", process.stdout)
        self.assertIn("requirements.txt", process.stdout)

    def test_run_all_demo_creates_artifacts_in_path_with_spaces(self) -> None:
        """run_all.py 能在含空格路径下产出看板、CSV 与 JSON。"""
        with tempfile.TemporaryDirectory(prefix="yolo run ") as tmp:
            process = run_script("run_all.py", "--mode", "demo", "--root", tmp)
            self.assertEqual(process.returncode, 0, process.stderr)
            out = Path(tmp)
            self.assertTrue((out / "reports" / "index.html").is_file())
            self.assertTrue((out / "outputs" / "comparison.csv").is_file())
            self.assertTrue((out / "outputs" / "results.json").is_file())
            self.assertIn("reports/index.html", process.stdout)


if __name__ == "__main__":
    unittest.main()
