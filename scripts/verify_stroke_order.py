#!/usr/bin/env python3
"""
验证骨架笔画书写方向：用「树」字样检查方向归一化后无倒写。

断言（几何启发式，非教材笔画序）：
  - 至少有一条有效笔画
  - 竖向主导笔画：起点 y ≤ 终点 y（上→下）
  - 横向主导笔画：起点 x ≤ 终点 x（左→右）

用法:
  <ENV_PY> scripts/verify_stroke_order.py
  <ENV_PY> scripts/verify_stroke_order.py --char 木 --out-dir examples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import stream_render as sr

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent


def _find_cjk_font(size: int = 220) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """在常见 Windows / 跨平台路径中查找可画汉字的字体。"""
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    raise SystemExit(
        "[err] 未找到可渲染汉字的字体（试过 msyh/simhei/simsun/NotoSansCJK/PingFang）"
    )


def render_char_png(char: str, size: int = 320, pad: int = 40) -> np.ndarray:
    """白底黑字 BGR 图，供骨架追踪。"""
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _find_cjk_font(size=size - pad * 2)
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2 - bbox[0]
    y = (size - th) // 2 - bbox[1]
    draw.text((x, y), char, fill=(20, 20, 20), font=font)
    rgb = np.array(canvas)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def extract_strokes(bgr: np.ndarray, min_points: int = 8, spacing: float = 2.5):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )
    ink = thresh < 10
    if not ink.any():
        # 回退：简单灰度阈值（白底黑字）
        ink = gray < 128
    skel = sr._zhang_suen_skeleton(ink, max_iterations=160)
    raw = sr.trace_8connected(skel, min_points=min_points)
    return sr._finalize_skeleton_strokes(raw, spacing)


def stroke_direction_ok(stroke: list[tuple[int, int]]) -> tuple[bool, str]:
    if len(stroke) < 2:
        return False, "点数不足"
    sx, sy = stroke[0]
    ex, ey = stroke[-1]
    dx = ex - sx
    dy = ey - sy
    if abs(dy) >= abs(dx):
        ok = sy <= ey
        return ok, "vertical" if ok else "vertical_reversed"
    ok = sx <= ex
    return ok, "horizontal" if ok else "horizontal_reversed"


def draw_debug(
    bgr: np.ndarray,
    strokes: list[list[tuple[int, int]]],
    out_path: Path,
) -> None:
    canvas = bgr.copy()
    for i, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue
        pts = np.array(stroke, dtype=np.int32).reshape(-1, 1, 2)
        color = (
            int(40 + (i * 70) % 180),
            int(80 + (i * 40) % 150),
            int(200 - (i * 30) % 150),
        )
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
        sx, sy = stroke[0]
        ex, ey = stroke[-1]
        cv2.circle(canvas, (sx, sy), 4, (0, 180, 0), -1)
        cv2.circle(canvas, (ex, ey), 4, (0, 0, 220), -1)
        cv2.putText(
            canvas,
            str(i + 1),
            (sx + 4, sy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 40, 40),
            2,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise SystemExit(f"[err] 无法写出 debug 图: {out_path}")
    buf.tofile(str(out_path))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="验证骨架笔画书写方向（树字样）")
    p.add_argument("--char", default="树", help="用于验证的汉字（默认：树）")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "examples",
        help="字样与 debug 图输出目录",
    )
    args = p.parse_args(argv)

    char = args.char.strip()
    if not char:
        print("[err] --char 不能为空", file=sys.stderr)
        return 1

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture = out_dir / f"stroke-order-{char}.png"
    debug = out_dir / f"stroke-order-{char}-debug.png"

    bgr = render_char_png(char)
    ok, buf = cv2.imencode(".png", bgr)
    if ok:
        buf.tofile(str(fixture))
        print(f"[ok] 字样: {fixture}")

    strokes = extract_strokes(bgr)
    if not strokes:
        print("[FAIL] 未追踪到任何笔画", file=sys.stderr)
        return 1

    print(f"[info] 笔画数: {len(strokes)}")
    failures: list[str] = []
    for i, stroke in enumerate(strokes):
        sx, sy = stroke[0]
        ex, ey = stroke[-1]
        ok_dir, kind = stroke_direction_ok(stroke)
        status = "OK" if ok_dir else "FAIL"
        print(
            f"  #{i + 1:02d} {status} {kind:20s} "
            f"start=({sx},{sy}) end=({ex},{ey}) pts={len(stroke)}"
        )
        if not ok_dir:
            failures.append(f"笔画#{i + 1} {kind} start=({sx},{sy}) end=({ex},{ey})")

    draw_debug(bgr, strokes, debug)
    print(f"[ok] debug: {debug}")

    if failures:
        print("[FAIL] 方向断言未通过:", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print("[PASS] 所有笔画方向符合上→下 / 左→右")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
