"""Bring the Figure 1 schematic in line with the camera-ready: three external cohorts, field faults.

Figure 1 was drawn in diagrams.net and exported to vector PDF; the editable diagram travels inside
the PDF's metadata. There is no diagrams.net renderer in this environment, so the two lines that
changed are patched in place: each old line is redacted and the new one is set at the same baseline,
centred on the same box, in Arial, which is metric-identical to the Liberation Sans the export
embeds. The embedded diagram is updated to match and also written out as ``fig1_framework.drawio``,
so the next edit can start from the source rather than from this patch.

    Input box    "external: ds004504"   ->  "external: 3 cohorts, n = 257"   (A.12)
    Sensor box   "42 conditions:"       ->  "42 conditions + field faults:"  (A.13)
    Ladder strip the raster heads under the Input and Sensor boxes, captioned with internal codes
                 (b19 ... b1_ap), -> the vector strip e29 draws, captioned with the montage names
                 of Figure 2a and Figure A7. The PDF gets the vector PDF in the visible (cropped)
                 area of the old image; the diagram gets the SVG, uncropped, in the same cell.

Idempotent: each step checks whether it has already been applied.

    python experiments/e29_montage_map.py      # writes fig1_ladder.{pdf,svg}
    python tools/patch_framework_figure.py
Out: outputs/figures/fig1_framework.pdf (patched), outputs/figures/fig1_framework.drawio
"""
from __future__ import annotations

import base64
import os
import re
import sys
import urllib.parse
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures" / "fig1_framework.pdf"
DRAWIO = FIG.with_suffix(".drawio")
LADDER = FIG.with_name("fig1_ladder")
#: Pixel size of the old raster ladder, which identifies it among the figure's two images.
OLD_LADDER_PX = (1557, 345)
FONTS = Path(os.environ.get("FONT_DIR", r"C:\Windows\Fonts"))   # Arial, italic and bold-italic
SIZE = 7.4

#: (text of the old line, [(new text, bold?) segments]); both lines are italic, bold segments are
#: bold-italic, as in the original.
PATCHES = [
    ("external: ds004504", [("external: ", False), ("3 cohorts, n = 257", True)]),
    ("42 conditions:", [("42 conditions + field faults:", False)]),
]
XML_EDITS = [
    ("external: &lt;b&gt;ds004504&lt;/b&gt;", "external: &lt;b&gt;3 cohorts, n = 257&lt;/b&gt;"),
    ("42 conditions: &lt;br&gt;", "42 conditions + field faults: &lt;br&gt;"),
]


def _lines(page):
    """Visual lines as (joined text, bbox, first-span origin)."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if spans:
                text = "".join(s["text"] for s in spans).replace("\xa0", " ").strip()
                out.append((text, fitz.Rect(ln["bbox"]), spans[0]["origin"]))
    return out


def _inset(style):
    """The diagram's crop of an image cell, as fractions (top, right, bottom, left)."""
    m = re.search(r"clipPath=inset\\?\(([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\\?\)", style)
    return tuple(float(v) / 100 for v in m.groups()) if m else (0.0, 0.0, 0.0, 0.0)


def patch_ladder(page, xml):
    """Swap the raster ladder for the vector one, in the PDF and in the diagram source."""
    old = [x for x in page.get_images(full=True) if (x[2], x[3]) == OLD_LADDER_PX]
    if not old:
        return xml, False
    cell = re.search(r'style="([^"]*image=data:image/png,[^";]+;[^"]*clipPath=[^"]*)"', xml)
    if not cell:
        sys.exit("ladder cell not found in the diagram source")
    top, right, bottom, left = _inset(cell.group(1))
    xref = old[0][0]
    r = page.get_image_rects(xref)[0]
    shown = fitz.Rect(r.x0 + left * r.width, r.y0 + top * r.height,
                      r.x1 - right * r.width, r.y1 - bottom * r.height)
    page.delete_image(xref)
    strip = fitz.open(LADDER.with_suffix(".pdf"))
    page.show_pdf_page(shown, strip, 0)
    svg = base64.b64encode(LADDER.with_suffix(".svg").read_bytes()).decode("ascii")
    style = re.sub(r"image=data:image/png,[^;]+;", f"image=data:image/svg+xml,{svg};", cell.group(1))
    style = re.sub(r"clipPath=[^;]*;?", "", style)
    return xml.replace(cell.group(1), style), True


def main():
    doc = fitz.open(FIG)
    page = doc[0]
    meta = doc.metadata
    xml = urllib.parse.unquote(meta.get("subject", ""))
    xml, ladder = patch_ladder(page, xml)
    have = {t for t, _, _ in _lines(page)}
    if any("".join(s for s, _ in new) in have for _, new in PATCHES):
        if not ladder:
            print("already patched")
            return 0
        return _save(doc, meta, xml)

    fi = fitz.Font(fontfile=str(FONTS / "ariali.ttf"))
    fbi = fitz.Font(fontfile=str(FONTS / "arialbi.ttf"))
    targets = []
    for old, new in PATCHES:
        hit = [(r, o) for t, r, o in _lines(page) if t == old]
        if len(hit) != 1:
            sys.exit(f"expected one line reading {old!r}, found {len(hit)}")
        rect, origin = hit[0]
        targets.append((rect, origin, new))
        # Tight to the glyphs, no fill: the box colour behind the text must survive.
        page.add_redact_annot(rect + (-0.3, -0.3, 0.3, 0.3), fill=False)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                          graphics=fitz.PDF_REDACT_LINE_ART_NONE)

    page.insert_font(fontname="ArI", fontfile=str(FONTS / "ariali.ttf"))
    page.insert_font(fontname="ArBI", fontfile=str(FONTS / "arialbi.ttf"))
    for rect, origin, new in targets:
        width = sum((fbi if bold else fi).text_length(s, SIZE) for s, bold in new)
        x = (rect.x0 + rect.x1) / 2 - width / 2
        for s, bold in new:
            page.insert_text((x, origin[1]), s, fontsize=SIZE, fontname="ArBI" if bold else "ArI")
            x += (fbi if bold else fi).text_length(s, SIZE)

    for a, b in XML_EDITS:
        if a in xml:
            xml = xml.replace(a, b)
    return _save(doc, meta, xml)


def _save(doc, meta, xml):
    if xml:
        DRAWIO.write_text(xml, encoding="utf-8")
        meta["subject"] = urllib.parse.quote(xml)
        doc.set_metadata(meta)
    # Embed only the glyphs used; the full Arial faces would triple the file for two lines of text.
    doc.subset_fonts()
    tmp = FIG.with_name(FIG.stem + ".tmp.pdf")
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    tmp.replace(FIG)
    print(f"patched {FIG.name}; diagram source -> {DRAWIO.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
