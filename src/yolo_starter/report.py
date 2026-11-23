"""生成自包含、离线可看的 HTML 看板（reports/index.html）。"""
from __future__ import annotations

import html
from pathlib import Path

from .paths import ProjectPaths
from .visualize import comparison_table_html


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


CSS = """
:root{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--bg:#f8fafc;--card:#ffffff;
--blue:#2563eb;--amber:#b45309;--amber-bg:#fffbeb;--green:#059669;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.65}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 64px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;
border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:28px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:19px;margin:36px 0 14px;
padding-left:10px;border-left:4px solid var(--blue)}
p{margin:8px 0}.muted{color:var(--muted);font-size:13px}
.badge{display:inline-block;padding:5px 12px;border-radius:999px;font-size:13px;font-weight:600}
.badge.real{background:#ecfdf5;color:var(--green);border:1px solid #a7f3d0}
.badge.demo{background:var(--amber-bg);color:var(--amber);border:1px solid #fde68a}
.notice{background:var(--amber-bg);border:1px solid #fde68a;color:#78350f;
border-radius:10px;padding:12px 16px;margin:16px 0;font-size:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.card b{display:block;font-size:15px;margin-bottom:6px}
.card span{font-size:13px;color:var(--muted)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px;margin:12px 0;overflow-x:auto}
table.grid{border-collapse:collapse;width:100%;font-size:13px;min-width:880px}
th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top}
th{background:#f1f5f9;font-weight:600;white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.strong{font-weight:700;color:var(--blue)}
.id-cell{font-weight:600;white-space:nowrap}
.tag{background:#dbeafe;color:#1d4ed8;border-radius:4px;padding:1px 6px;font-size:11px}
ul{padding-left:22px}li{margin:6px 0}
pre{background:#0f172a;color:#e2e8f0;border-radius:10px;padding:14px 16px;
overflow-x:auto;font-size:12.5px;line-height:1.7}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
img{max-width:100%;border-radius:8px;border:1px solid var(--line)}
footer{margin-top:44px;border-top:1px solid var(--line);padding-top:16px;color:var(--muted);font-size:13px}
"""


def build_report(context: dict) -> str:
    """根据流水线产物构造完整 HTML；所有外部文本均做 HTML 转义。"""
    mode = context.get("mode", "demo")
    is_demo = mode == "demo"
    badge_class = "demo" if is_demo else "real"
    badge_text = "演示数据" if is_demo else "真实实验"
    source_note = context.get("source_note", "")
    rows = context.get("rows", [])
    conclusions = context.get("conclusions", [])
    charts = context.get("charts", {})
    env = context.get("env", {})
    warnings = context.get("warnings", [])
    commands = context.get("commands", [])

    notice = ""
    if is_demo:
        notice = f'<div class="notice"><b>数据说明：</b>{_esc(source_note)}</div>'

    cards = [
        ("能看", "本页即看板：垃圾检测框、投放建议、指标图、对比表与结论全部可视化，离线双击即可打开。"),
        ("能跑", "scripts/run_all.py 一键产出全部文件；scripts/check.py 自检；tests/ 为可重跑单测。"),
        ("能对比", "configs/experiments.json 定义受控实验，输出对比表、柱状图与自动结论。"),
        ("能复述", "README.md + docs/WALKTHROUGH.md 讲稿 + docs/CHECKLIST.md 验收清单。"),
    ]
    cards_html = "".join(
        f'<div class="card"><b>{_esc(title)}</b><span>{_esc(desc)}</span></div>'
        for title, desc in cards
    )

    design_rows = "".join(
        "<tr>"
        f"<td class='id-cell'>{_esc(r.get('id', ''))}</td>"
        f"<td>{_esc(r.get('name', ''))}</td>"
        f"<td>{_esc(r.get('model', ''))} / {_esc(r.get('imgsz', ''))} / conf {_esc(r.get('conf', ''))}</td>"
        f"<td>{_esc(r.get('hypothesis', ''))}</td>"
        "</tr>"
        for r in rows
    )
    design_table = (
        "<table class='grid'><thead><tr><th>编号</th><th>实验</th><th>配置</th>"
        "<th>待验证假设</th></tr></thead><tbody>" + design_rows + "</tbody></table>"
    )

    charts_html = "".join(
        f'<div class="panel"><p class="muted">{_esc(title)}</p>{svg}</div>'
        for title, svg in charts.items()
    )
    conclusions_html = "".join(f"<li>{_esc(item)}</li>" for item in conclusions)
    env_html = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in env.items()
    )
    warnings_html = ""
    if warnings:
        warnings_html = "<div class='notice'><b>运行提示：</b><ul>" + "".join(
            f"<li>{_esc(w)}</li>" for w in warnings
        ) + "</ul></div>"
    commands_html = _esc("\n".join(commands))

    visual_html = context.get("visual_html", "")
    classification = context.get("classification_html", "")
    classification_section = (
        '<h3>分类投放建议</h3><div class="panel">' + classification + "</div>"
        if classification else ""
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>校园垃圾分类投放识别看板 · {_esc(mode)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>校园垃圾分类投放识别 · YOLOv8 实验看板</h1>
    <p class="muted">数据集：{_esc(context.get('dataset', 'datasets/campus_trash/data.yaml'))} ｜ 生成时间：{_esc(context.get('generated_at', ''))} ｜ 模式：{_esc(mode)}</p>
  </div>
  <span class="badge {badge_class}">{badge_text}</span>
</header>

{notice}
{warnings_html}

<h2>一、四项能力总览</h2>
<div class="cards">{cards_html}</div>

<h2>二、实验设计（能对比）</h2>
<p class="muted">同一数据集、同一指标口径，只改变模型 / 输入分辨率 / 置信度阈值，保证差异可归因。</p>
<div class="panel">{design_table}</div>

<h2>三、指标对比表</h2>
<div class="panel">{comparison_table_html(rows)}</div>
<p class="muted">Δ 列以 {_esc(context.get('baseline', 'E1'))} 为基线；mAP 与 P/R 越高越好，单图耗时越低越好。</p>

<h2>四、可视化（能看）</h2>
<div class="charts">{charts_html}</div>
<h2>五、检测效果</h2>
<div class="panel">{visual_html}</div>
{classification_section}

<h2>六、自动结论</h2>
<div class="panel"><ul>{conclusions_html}</ul></div>

<h2>七、环境与复现命令（能跑）</h2>
<div class="panel"><table class="grid"><thead><tr><th>项</th><th>值</th></tr></thead>
<tbody>{env_html}</tbody></table></div>
<pre>{commands_html}</pre>

<h2>八、复述提纲（能复述）</h2>
<div class="panel"><ul>
<li>项目对象是校园垃圾：塑料瓶、纸张、纸板、玻璃、金属、其他垃圾共 6 类，检测后按规则给出投放建议。</li>
<li>YOLO 是单阶段检测器，一次前向传播同时预测框与类别；本项目用 Ultralytics YOLOv8。</li>
<li>流程：加载实验矩阵 → 逐组验证 / 推理 → 指标归一化 → 生成对比表与图 → 输出看板与结论。</li>
<li>评价：mAP@0.50、mAP@0.50:0.95、Precision、Recall、单图耗时；结论由代码从数据自动推导。</li>
<li>取舍：分辨率与模型规模换精度、耗时间上升；提高 conf 换 Precision、Recall 下降。</li>
<li>复现：完整讲稿见 docs/WALKTHROUGH.md，逐项验收见 docs/CHECKLIST.md。</li>
</ul></div>

<footer>数据来源：{_esc(source_note)}<br>本页由 src/yolo_starter/report.py 生成，无外部资源依赖，可离线打开。</footer>
</div>
</body>
</html>"""


def save_report(context: dict, path: str | Path | None = None) -> Path:
    """把看板写入磁盘并返回路径；父目录不存在时自动创建。"""
    target = Path(path) if path is not None else ProjectPaths().report_file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_report(context), encoding="utf-8")
    return target
