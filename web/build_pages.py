"""Turn web/app.html into a complete HTML document for GitHub Pages (docs/index.html).

app.html holds only the page content, without <html>/<head>/<body>.
Everything before the first <style> (title, meta, link) goes into <head>, the rest into <body>.

Usage:  python web/build_pages.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "web" / "app.html"
OUT = ROOT / "docs" / "index.html"

SKELETON = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
:root {{ color-scheme: light; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }}
body {{ margin: 0; font: 14px system-ui, sans-serif; }}
img {{ max-width: 100%; }}
[hidden] {{ display: none !important; }}
</style>
{head}
</head>
<body>
{body}
</body>
</html>
"""


def main():
    src = SRC.read_text(encoding="utf-8")
    cut = src.index("<style>")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(SKELETON.format(head=src[:cut].strip(), body=src[cut:].strip()), encoding="utf-8")
    (OUT.parent / ".nojekyll").touch()  # keep GitHub Pages from processing the files
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
