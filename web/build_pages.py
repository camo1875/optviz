"""web/app.html 을 GitHub Pages용 완전한 HTML 문서(docs/index.html)로 만든다.

app.html 은 <html>/<head>/<body> 없이 내용만 담고 있다 (claude.ai 아티팩트로도 그대로 올리기 위해).
여기서 첫 <style> 앞부분(title, meta, link)은 <head>로, 나머지는 <body>로 감싼다.

사용법:  python web/build_pages.py
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
    (OUT.parent / ".nojekyll").touch()  # GitHub Pages가 파일을 가공하지 않도록
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
