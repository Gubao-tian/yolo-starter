# 校园垃圾分类投放识别项目 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native, chosen because the user delegated execution) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有 `yolo-starter` 骨架补成一套可直接课堂演示、满足「能看/能跑/能对比/能复述」的校园垃圾分类投放识别项目。

**Architecture:** 现有 `src/yolo_starter` 已完成双模式流水线、指标归一化、SVG 可视化与自包含 HTML 看板，60 个单测通过；本计划先补入口脚本与文档，再按用户最终选择将主题改为 6 类校园垃圾检测，并新增「类别→投放建议」规则模块与看板面板。最终 71 个单测通过。

**Tech Stack:** Python 3.13 标准库、unittest、Ultralytics YOLOv8（full 模式可选）、自包含 HTML/SVG。

**Spec:** `docs/DESIGN.md`

## Global Constraints

- `demo` 模式不得导入 `ultralytics` / `torch` / `cv2`，不得联网。
- 全部路径需兼容含空格的 Windows 路径（项目实际位于 `E:\work  buddy\...`）。
- 看板必须是单个 `reports/index.html`，不引用外部资源。
- 测试用 `python -m unittest discover -s tests -v`，不引入 pytest。
- 示例数据必须显著标注为非真实推理结果。

## Review Focus

- 含空格路径下运行入口脚本仍应成功 → `tests/test_scripts.py` 用带空格的临时目录跑 `run_all.py`。
- 裸 Python 环境（无 ultralytics）运行 `demo` 不应报错 → `tests/test_scripts.py` 用 `sys.executable` 子进程验证。
- `full` 模式缺少依赖时应给出明确安装命令而非堆栈 → `tests/test_scripts.py` 检查 `check.py` 输出包含 `requirements.txt`。
- 报告需覆盖四项能力且转义外部文本 → 现有 `tests/test_report.py` 已覆盖，回归运行。
- 重复运行结果可复现 → 现有 `tests/test_pipeline.py` 已覆盖，回归运行。

---

### Task 1: 补齐一键入口脚本与环境自检

**Files:**
- Create: `scripts/run_all.py`
- Create: `scripts/check.py`
- Create: `tests/test_scripts.py`
- Create: `outputs/.gitkeep`, `reports/.gitkeep`（空占位，保证目录可提交）

**Interfaces:**
- Consumes: `pipeline.run(mode, experiment_ids, dataset, device, epochs, root, open_browser, limit, config_path) -> dict`，返回含 `artifacts`、`rows`、`conclusions`、`warnings` 的字典。
- Consumes: `pipeline.environment_info() -> dict` 与 `yolo_backend.backend_status() -> dict`。
- Produces: CLI `python scripts/run_all.py [--mode demo|full] [--experiments E1,E2] [--dataset NAME] [--device cpu|cuda:0] [--epochs N] [--limit N] [--config PATH] [--root PATH] [--open]`；CLI `python scripts/check.py`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_scripts.py
import subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                          cwd=ROOT, text=True, capture_output=True)

class TestScripts(unittest.TestCase):
    def test_check_reports_demo_ready_and_full_hint(self):
        p = run("check.py")
        self.assertEqual(p.returncode, 0)
        self.assertIn("demo", p.stdout)
        self.assertIn("requirements.txt", p.stdout)

    def test_run_all_demo_creates_artifacts_in_path_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="yolo run ") as tmp:
            p = run("run_all.py", "--mode", "demo", "--root", tmp)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue((Path(tmp) / "reports" / "index.html").is_file())
            self.assertTrue((Path(tmp) / "outputs" / "comparison.csv").is_file())
            self.assertIn("reports/index.html", p.stdout)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_scripts -v`
Expected: FAIL，`scripts/check.py` / `scripts/run_all.py` 不存在。

- [ ] **Step 3: 实现两个脚本**

`scripts/run_all.py`：
- 顶部把 `PROJECT_ROOT/src` 插入 `sys.path`（与 `tests/__init__.py` 相同手法），使脚本无需安装包即可运行。
- 用 `argparse` 定义 Interfaces 中的全部参数；`--mode` 默认 `demo`，`--epochs` 默认 `0`。
- 调用 `pipeline.run(...)`；把返回结果里的 `artifacts` 逐条打印为 `已生成: <相对路径>`，打印结论条数与警告条数。
- 捕获 `ValueError` / `RuntimeError`，向 stderr 打印简短中文错误并返回退出码 2，不打印完整堆栈。
- `--open` 透传给 `pipeline.run(open_browser=True)`。

`scripts/check.py`：
- 同样 bootstrap `sys.path`。
- 打印 Python 版本、`pipeline.environment_info()` 与 `yolo_backend.backend_status()` 的关键项。
- 打印「demo 模式可用：python scripts/run_all.py --mode demo」；若 ultralytics 缺失，打印「full 模式需先安装：pip install -r requirements.txt」，仍返回 0。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests.test_scripts -v`
Expected: PASS（2 tests）。

- [ ] **Step 5: 全量回归**

Run: `python -m unittest discover -s tests -v`
Expected: PASS（62 tests）。

### Task 2: 补齐课堂文档

**Files:**
- Create: `README.md`
- Create: `docs/WALKTHROUGH.md`
- Create: `docs/CHECKLIST.md`

**Interfaces:**
- Consumes: `configs/experiments.json` 的 E1~E4、`COMMANDS` 常量、看板章节结构。
- Produces: 可被学生直接照抄的命令与讲稿；文档中的命令必须与脚本实际参数一致。

- [ ] **Step 1: 写 README.md**

包含：项目一句话介绍、四项能力对照表、目录树、`demo`/`full` 快速开始命令、`--experiments`/`--epochs`/`--dataset` 参数说明、常见问题（无依赖、无网络、CPU 很慢、看板打不开）。所有命令须与 `pipeline.COMMANDS` 一致。

- [ ] **Step 2: 写 docs/WALKTHROUGH.md**

包含：5 分钟讲解稿（6 段覆盖看板 8 节）；YOLO 单阶段检测器的 3 句话解释；mAP/Precision/Recall/耗时的白话解释；垃圾分类规则分工；E1→E2→E3→E4 的预期结论；7 个答辩 Q&A（为何选 YOLOv8、为何选垃圾分类、demo 数据是否真实、如何准备数据集、规则如何决策、置信度取舍、如何复现）。

- [ ] **Step 3: 写 docs/CHECKLIST.md**

用勾选框列出四项能力的可验证标准，每项附一条命令或一个文件；包含「演示前检查」清单（入口脚本存在、看板可离线打开、示例数据徽标可见）。

- [ ] **Step 4: 验证文档命令与文件引用一致**

Run: `python -c "from pathlib import Path; import re; t=Path('README.md').read_text(encoding='utf-8'); assert 'run_all.py' in t and 'check.py' in t and 'unittest' in t"`
Expected: 退出码 0，无输出。

### Task 3: 端到端演示验证与交付

**Files:**
- Modify: 无（仅运行验证）

**Interfaces:**
- Consumes: Task 1 产出的 CLI 与 Task 2 产出的文档。
- Produces: `reports/index.html`、`outputs/results.json`、`outputs/comparison.csv`、`outputs/summary.md`。

- [ ] **Step 1: 生成演示看板**

Run: `python scripts/run_all.py --mode demo`
Expected: 退出码 0，打印 4 个 `已生成:` 路径，`reports/index.html` 存在。

- [ ] **Step 2: 校验关键内容**

Run: `python -c "from pathlib import Path; t=Path('reports/index.html').read_text(encoding='utf-8'); assert '演示数据' in t and 'E1' in t and '能对比' in t and 'http' not in t.replace('http://www.w3.org/2000/svg','')"`
Expected: 退出码 0。

- [ ] **Step 3: 展示成品**

用 `present_files` 打开 `reports/index.html`，确认看板包含检测框示意、三张柱状图、对比表和自动结论。

- [ ] **Step 4: 最终全量测试**

Run: `python -m unittest discover -s tests -v`
Expected: PASS（62 tests）。
