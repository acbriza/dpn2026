#!/bin/bash
# Render an SVG in this directory to a vector PDF (via headless Chrome) and a
# raster PNG (via pdftoppm) for visual QA. Usage: ./render.sh <name-without-ext>
set -euo pipefail
cd "$(dirname "$0")"
name="$1"
svg="${name}.svg"
[ -f "$svg" ] || { echo "missing $svg" >&2; exit 1; }

width=$(grep -o 'width="[0-9.]*"' "$svg" | head -1 | grep -o '[0-9.]*')
height=$(grep -o 'height="[0-9.]*"' "$svg" | head -1 | grep -o '[0-9.]*')

html=$(mktemp --suffix=.html)
{
  echo "<html><head><style>"
  echo "@page { size: ${width}px ${height}px; margin: 0; }"
  echo "html,body { margin:0; padding:0; }"
  echo "</style></head><body>"
  cat "$svg"
  echo "</body></html>"
} > "$html"

google-chrome --headless=new --no-sandbox --disable-gpu \
  --print-to-pdf="${name}.pdf" --no-pdf-header-footer \
  "file://${html}" 2>/dev/null

pdftoppm -png -r 150 "${name}.pdf" "${name}_qa"
rm -f "$html"
echo "wrote ${name}.pdf and ${name}_qa-1.png"