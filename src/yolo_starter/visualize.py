"""纯标准库 SVG 可视化：图表、对比表、检测效果图。

不依赖 matplotlib / plotly，也不需要联网加载 JS，保证看板离线可看。
"""
from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path

from . import trash_rules

SVG_NS = "http://www.w3.org/2000/svg"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def bar_svg(values: list[float], labels: list[str], title: str,
            unit: str = "", color: str = "#2563eb", width: int = 640,
            height: int = 260) -> str:
    """生成单系列柱状图 SVG；空数据或全零数据也能正常渲染。"""
    if not values or not labels:
        return (
            f'<svg xmlns="{SVG_NS}" viewBox="0 0 {width} {height}" width="100%" '
            f'role="img" aria-label="{_esc(title)}">'
            f'<text x="24" y="40" font-size="16" fill="#475569">{_esc(title)}</text>'
            f'<text x="24" y="{height // 2}" font-size="14" fill="#94a3b8">暂无数据</text></svg>'
        )

    left, right, top, bottom = 52, 18, 46, 46
    plot_w = width - left - right
    plot_h = height - top - bottom
    peak = max(values) or 1.0
    scale = peak * 1.15 if peak > 0 else 1.0
    count = len(values)
    slot = plot_w / count
    bar_w = min(72.0, slot * 0.56)

    parts = [
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="{_esc(title)}">',
        f'<text x="{left}" y="26" font-size="15" font-weight="600" fill="#0f172a">{_esc(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#cbd5e1"/>',
        f'<text x="8" y="{top + 6}" font-size="11" fill="#64748b">{scale:.2f}</text>',
    ]
    parts.append(f'<text x="8" y="{top + plot_h}" font-size="11" fill="#64748b">0</text>')

    for index, (value, label) in enumerate(zip(values, labels)):
        bar_h = max(1.0, (float(value) / scale) * plot_h) if scale > 0 else 1.0
        x = left + slot * index + (slot - bar_w) / 2
        y = top + plot_h - bar_h
        parts.append(
            f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{bar_h:.1f}" rx="4" fill="{_esc(color)}"/>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
            f'font-size="12" font-weight="600" fill="#0f172a">{float(value):.3f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 20:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#475569">{_esc(label)}</text>'
        )
        if unit:
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 36:.1f}" text-anchor="middle" '
                f'font-size="10" fill="#94a3b8">{_esc(unit)}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def comparison_table_html(rows: list[dict]) -> str:
    """生成指标对比表；差值带符号，便于直接读出涨跌方向。"""
    headers = [
        "实验", "配置", "mAP@0.50", "mAP@0.50:0.95", "Precision", "Recall",
        "耗时 ms/图", "参数量 M", "ΔmAP@0.50", "ΔmAP@0.50:0.95", "Δ耗时",
    ]
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body: list[str] = []
    for row in rows:
        m = row.get("metrics", {})
        config = f"{_esc(row.get('model', ''))} / {_esc(row.get('imgsz', ''))} / conf {_esc(row.get('conf', ''))}"
        badge = ' <span class="tag">基线</span>' if row.get("is_baseline") else ""
        cells = [
            f"<td class='id-cell'>{_esc(row.get('id', ''))}{badge}<br>"
            f"<span class='muted'>{_esc(row.get('name', ''))}</span></td>",
            f"<td>{config}</td>",
            f"<td class='num'>{float(m.get('map50', 0)):.3f}</td>",
            f"<td class='num strong'>{float(m.get('map5095', 0)):.3f}</td>",
            f"<td class='num'>{float(m.get('precision', 0)):.3f}</td>",
            f"<td class='num'>{float(m.get('recall', 0)):.3f}</td>",
            f"<td class='num'>{float(m.get('ms_per_image', 0)):.1f}</td>",
            f"<td class='num'>{float(m.get('params_m', 0)):.1f}</td>",
            f"<td class='num'>{float(row.get('delta_map50', 0)):+.3f}</td>",
            f"<td class='num'>{float(row.get('delta_map5095', 0)):+.3f}</td>",
            f"<td class='num'>{float(row.get('delta_ms_per_image', 0)):+.1f}</td>",
        ]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table class="grid"><thead><tr>' + head + "</tr></thead><tbody>"
        + "".join(body) + "</tbody></table>"
    )


def detection_svg(detections: list[dict], width: int = 640, height: int = 360) -> str:
    """生成检测效果示意图；坐标为 0~1 相对值。"""
    parts = [
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 {width} {height}" width="100%" '
        'role="img" aria-label="目标检测效果图">',
        '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#e0f2fe"/><stop offset="100%" stop-color="#f1f5f9"/>'
        "</linearGradient></defs>",
        f'<rect width="{width}" height="{height}" fill="url(#bg)"/>',
        f'<rect y="{int(height * 0.72)}" width="{width}" height="{int(height * 0.28)}" fill="#cbd5e1" opacity="0.6"/>',
        '<text x="16" y="26" font-size="13" fill="#475569">检测框 = 模型输出；框上文字 = 类别与置信度</text>',
    ]
    for box in detections:
        x, y = float(box["x"]) * width, float(box["y"]) * height
        w, h = float(box["w"]) * width, float(box["h"]) * height
        color = _esc(box.get("color", "#2563eb"))
        label = f"{_esc(box['label'])} {float(box['score']):.2f}"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="3" rx="3"/>'
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{max(0, y - 20):.1f}" width="{min(w, 12 + 8 * len(label)):.1f}" '
            f'height="20" fill="{color}" rx="3"/>'
        )
        parts.append(
            f'<text x="{x + 6:.1f}" y="{max(14, y - 5):.1f}" font-size="12" fill="#ffffff">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def classification_panel_html(detections: list[dict]) -> str:
    """生成垃圾分类建议面板；所有类别与建议文本都做 HTML 转义。"""
    summary = trash_rules.summarize(detections)
    rows = "".join(
        "<tr>"
        f"<td class='id-cell'>{_esc(item['label'])}</td>"
        f"<td class='num'>{int(item['count'])}</td>"
        f"<td>{_esc(item['advice'])}</td>"
        "</tr>"
        for item in summary["items"]
    )
    return (
        '<table class="grid"><thead><tr><th>识别类别</th><th>数量</th><th>投放建议</th>'
        "</tr></thead><tbody>" + rows + "</tbody></table>"
        f'<p><b>{_esc(summary["headline"])}</b></p>'
    )


def caption_html(name: object) -> str:
    """生成安全的文件名说明，避免外部文件名破坏 HTML 结构。"""
    return f'<figcaption class="muted">{_esc(name)}</figcaption>'


def image_data_uri(path: str | Path) -> str:
    """把本地图片转成 base64 data URI；文件不存在时返回空串。"""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    mime = mimetypes.guess_type(file_path.name)[0] or "image/png"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
