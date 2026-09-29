"""Minimal hand-written SVG primitives (no matplotlib dependency for figures)."""

from __future__ import annotations

from dataclasses import dataclass

PALETTE = ("#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b7791f", "#2c7a7b")


@dataclass(frozen=True)
class Frame:
    width: int = 720
    height: int = 400
    left: int = 70
    right: int = 20
    top: int = 40
    bottom: int = 55

    @property
    def plot_w(self) -> int:
        return self.width - self.left - self.right

    @property
    def plot_h(self) -> int:
        return self.height - self.top - self.bottom


class Canvas:
    def __init__(self, frame: Frame, title: str) -> None:
        self.f = frame
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{frame.width}" height="{frame.height}" '
            f'viewBox="0 0 {frame.width} {frame.height}" font-family="Helvetica, Arial, sans-serif" font-size="12">',
            f'<rect width="{frame.width}" height="{frame.height}" fill="#ffffff"/>',
            f'<text x="{frame.left}" y="22" font-size="15" font-weight="bold" fill="#1a202c">{esc(title)}</text>',
        ]

    def x(self, v: float, lo: float, hi: float) -> float:
        span = (hi - lo) or 1.0
        return self.f.left + (v - lo) / span * self.f.plot_w

    def y(self, v: float, lo: float, hi: float) -> float:
        span = (hi - lo) or 1.0
        return self.f.top + self.f.plot_h - (v - lo) / span * self.f.plot_h

    def axes(self, xlabel: str, ylabel: str) -> None:
        f = self.f
        x0, y0 = f.left, f.top + f.plot_h
        self.parts.append(f'<line x1="{x0}" y1="{f.top}" x2="{x0}" y2="{y0}" stroke="#4a5568"/>')
        self.parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + f.plot_w}" y2="{y0}" stroke="#4a5568"/>')
        self.parts.append(f'<text x="{x0 + f.plot_w / 2:.1f}" y="{f.height - 12}" text-anchor="middle" fill="#4a5568">{esc(xlabel)}</text>')
        self.parts.append(
            f'<text x="16" y="{f.top + f.plot_h / 2:.1f}" text-anchor="middle" fill="#4a5568" '
            f'transform="rotate(-90 16 {f.top + f.plot_h / 2:.1f})">{esc(ylabel)}</text>'
        )

    def yticks(self, lo: float, hi: float, n: int = 5, fmt: str = "{:.2f}") -> None:
        for i in range(n + 1):
            v = lo + (hi - lo) * i / n
            yy = self.y(v, lo, hi)
            self.parts.append(f'<line x1="{self.f.left - 4}" y1="{yy:.1f}" x2="{self.f.left + self.f.plot_w}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
            self.parts.append(f'<text x="{self.f.left - 8}" y="{yy + 4:.1f}" text-anchor="end" fill="#4a5568">{fmt.format(v)}</text>')

    def xtick(self, xv: float, label: str) -> None:
        y0 = self.f.top + self.f.plot_h
        self.parts.append(f'<line x1="{xv:.1f}" y1="{y0}" x2="{xv:.1f}" y2="{y0 + 4}" stroke="#4a5568"/>')
        self.parts.append(f'<text x="{xv:.1f}" y="{y0 + 17}" text-anchor="middle" fill="#4a5568">{esc(label)}</text>')

    def polyline(self, pts: list[tuple[float, float]], color: str, width: float = 2.0, dash: str | None = None) -> None:
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')

    def polygon(self, pts: list[tuple[float, float]], color: str, opacity: float = 0.2) -> None:
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.parts.append(f'<polygon points="{d}" fill="{color}" fill-opacity="{opacity}" stroke="none"/>')

    def rect(self, x: float, y: float, w: float, h: float, color: str) -> None:
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color}"/>')

    def circle(self, x: float, y: float, r: float, color: str) -> None:
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>')

    def legend(self, items: list[tuple[str, str]], x: float | None = None, y: float | None = None) -> None:
        x = self.f.left + 10 if x is None else x
        y = self.f.top + 8 if y is None else y
        for i, (label, color) in enumerate(items):
            yy = y + i * 16
            self.parts.append(f'<rect x="{x}" y="{yy - 8}" width="12" height="10" fill="{color}"/>')
            self.parts.append(f'<text x="{x + 18}" y="{yy + 1}" fill="#2d3748">{esc(label)}</text>')

    def note(self, text: str) -> None:
        self.parts.append(f'<text x="{self.f.left}" y="{self.f.top - 6}" font-size="11" fill="#718096">{esc(text)}</text>')

    def render(self) -> str:
        return "\n".join(self.parts + ["</svg>"])


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
