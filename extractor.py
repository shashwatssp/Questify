"""
Core PDF Question Extraction Engine.

Auto-detects PDF format (question numbering, option style, equation fonts,
watermarks, logos, separator lines) and extracts questions into structured
JSON + image assets.

Supports JEE, GATE, NEET, and similar exam formats.
"""

import pymupdf as fitz
import json
import re
import os
import io
import tempfile
import zipfile
from collections import Counter
from typing import Callable, Optional
from PIL import Image


# ── Unicode to LaTeX mapping ─────────────────────────────────────

UNICODE_LATEX_MAP = {
    '\u2192': r'\rightarrow', '\u2190': r'\leftarrow',
    '\u21cc': r'\rightleftharpoons', '\u00d7': r'\times',
    '\u00f7': r'\div', '\u00b1': r'\pm',
    '\u2260': r'\neq', '\u2264': r'\leq', '\u2265': r'\geq',
    '\u221e': r'\infty', '\u00b0': r'^{\circ}',
    '\u0394': r'\Delta', '\u2212': '-',
    '\U0001d461': 't', '\U0001d458': 'k', '\U0001d434': 'A',
    '\U0001d435': 'B', '\U0001d436': 'C', '\U0001d437': 'D',
    '\U0001d438': 'E', '\U0001d443': 'P', '\U0001d445': 'R',
    '\U0001d44e': 'a', '\U0001d44f': 'b', '\U0001d450': 'c',
    '\U0001d45b': 'n', '\U0001d45a': 'm', '\U0001d465': 'x',
    '\U0001d45f': 'r', '\U0001d45d': 'p',
    '\u2082': '_2', '\u2083': '_3', '\u2084': '_4', '\u2085': '_5',
    '\u207a': '^+', '\u207b': '^-',
}

MATH_FONT_KEYWORDS = ("Type3", "CambriaMath", "MathJax", "CMMI", "CMSY",
                       "Symbol", "STIXMath", "Mathematica", "MT-Extra")

# Minimum aspect ratio to consider an image a separator line
LINE_ASPECT_RATIO = 12


def _unicode_to_latex(text: str) -> str:
    result = text
    for uc, ltx in UNICODE_LATEX_MAP.items():
        result = result.replace(uc, ltx)
    if any(c in result for c in [r'\rightarrow', r'\times', r'\Delta',
                                  '_', '^', r'\frac', r'\infty']):
        result = f"${result}$"
    return result


def _is_math_font(font_name: str) -> bool:
    return any(kw in font_name for kw in MATH_FONT_KEYWORDS)


def _is_separator_line(w, h):
    """True if the image dimensions suggest a horizontal/vertical rule."""
    if w == 0 or h == 0:
        return True
    ratio = max(w, h) / max(min(w, h), 1)
    return ratio >= LINE_ASPECT_RATIO


# ── Auto-detection ───────────────────────────────────────────────

def detect_pdf_format(doc):
    """
    Scan the PDF to auto-detect question numbering, option format,
    equation fonts, watermark/logo images, and separator lines.
    """
    # Collect spans from first few pages
    all_spans = []
    for pg in range(min(10, doc.page_count)):
        page = doc[pg]
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                full_line = "".join(s["text"] for s in line["spans"]).strip()
                for span in line["spans"]:
                    all_spans.append({
                        "text": span["text"].strip(),
                        "font": span["font"],
                        "size": span["size"],
                        "line_text": full_line,
                        "page": pg,
                        "y": line["bbox"][1],
                        "bbox": line["bbox"],
                    })

    # --- Question numbering ---
    q_dot_pattern = re.compile(r'^Q\.\s*(\d+)\s*$')
    n_dot_pattern = re.compile(r'^(\d{1,3})\.$')

    q_dot_hits, n_dot_hits = [], []
    q_dot_fonts, n_dot_fonts = Counter(), Counter()

    for s in all_spans:
        m = q_dot_pattern.match(s["text"])
        if m:
            q_dot_hits.append(int(m.group(1)))
            q_dot_fonts[s["font"]] += 1
        m2 = n_dot_pattern.match(s["text"])
        if m2:
            n_dot_hits.append(int(m2.group(1)))
            n_dot_fonts[s["font"]] += 1

    if len(q_dot_hits) >= 3:
        q_style = "Q.N"
        q_regex = re.compile(r'^Q\.\s*(\d+)\s*$')
        q_font = q_dot_fonts.most_common(1)[0][0] if q_dot_fonts else None
    elif len(n_dot_hits) >= 3:
        q_style = "N."
        q_regex = re.compile(r'^(\d{1,3})\.$')
        q_font = n_dot_fonts.most_common(1)[0][0] if n_dot_fonts else None
    else:
        q_line_re = re.compile(r'^Q\.?\s*(\d+)')
        q_line_hits = [s for s in all_spans if q_line_re.match(s["line_text"])]
        if q_line_hits:
            q_style = "Q.N_line"
            q_regex = q_line_re
            q_font = Counter(s["font"] for s in q_line_hits).most_common(1)[0][0]
        else:
            q_style = "N."
            q_regex = re.compile(r'^(\d{1,3})\.$')
            q_font = None

    # --- Option format ---
    # Check for (A)/(B) vs A)/B) vs (1)/(2)
    paren_alpha = sum(1 for s in all_spans
                      if re.match(r'^\([A-D]\)', s["text"]))
    bare_alpha = sum(1 for s in all_spans
                     if re.match(r'^[A-D]\)', s["text"]))
    paren_num = sum(1 for s in all_spans
                    if re.match(r'^\([1-4]\)', s["text"]))

    if paren_num > paren_alpha and paren_num > bare_alpha:
        opt_style = "paren_num"  # (1) (2) (3) (4)
        opt_re = re.compile(r'^\(([1-4])\)\s*(.*)')
        opt_inline_re = re.compile(r'\(([1-4])\)\s*')
    elif paren_alpha >= bare_alpha:
        opt_style = "paren"      # (A) (B) (C) (D)
        opt_re = re.compile(r'^\(([A-D])\)\s*(.*)')
        opt_inline_re = re.compile(r'\(([A-D])\)\s*')
    else:
        opt_style = "bare"       # A) B) C) D)
        opt_re = re.compile(r'^([A-D])\)\s*(.*)')
        opt_inline_re = re.compile(r'([A-D])\)\s*')

    # --- Equation fonts ---
    eq_fonts = {s["font"] for s in all_spans if _is_math_font(s["font"])}

    # --- Watermark / logo / separator detection ---
    # Count how many distinct pages each image size appears on
    img_size_pages = Counter()
    for pg in range(doc.page_count):
        seen = set()
        for img in doc[pg].get_images(full=True):
            key = (img[2], img[3])
            if key not in seen:
                img_size_pages[key] += 1
                seen.add(key)

    # Images on >30% of pages are watermarks/logos; also thin lines
    skip_img_sizes = set()
    threshold = max(3, doc.page_count * 0.3)
    for (w, h), count in img_size_pages.items():
        if count >= threshold or _is_separator_line(w, h):
            skip_img_sizes.add((w, h))

    # Collect xrefs of images to hide in rendered output
    skip_xrefs = set()
    for pg in range(doc.page_count):
        for img in doc[pg].get_images(full=True):
            if (img[2], img[3]) in skip_img_sizes:
                skip_xrefs.add(img[0])

    # --- Answer pattern ---
    # Detect if the PDF contains "Answer (N)" or "Answer (A)" lines
    # Answer lines may have asterisks for disputed answers, e.g. "Answer (4*)"
    answer_re = None
    for s in all_spans:
        lt = s["line_text"]
        if re.match(r'^Answer\s*\([1-4]\*?\)', lt):
            answer_re = re.compile(r'^Answer\s*\(([1-4])\*?\)')
            break
        if re.match(r'^Answer\s*\([A-D]\*?\)', lt):
            answer_re = re.compile(r'^Answer\s*\(([A-D])\*?\)')
            break

    # --- Detect instruction/cover pages to skip ---
    # A page that has "instructions" or is entirely header with no real
    # questions is likely a cover page.
    skip_pages = set()
    for pg in range(min(3, doc.page_count)):
        text = doc[pg].get_text().lower()
        if ("important instruction" in text or "instructions :" in text
                or "instructions:" in text):
            skip_pages.add(pg)

    return {
        "q_style": q_style,
        "q_regex": q_regex,
        "q_font": q_font,
        "opt_style": opt_style,
        "opt_re": opt_re,
        "opt_inline_re": opt_inline_re,
        "eq_fonts": eq_fonts,
        "skip_img_sizes": skip_img_sizes,
        "skip_xrefs": skip_xrefs,
        "answer_re": answer_re,
        "skip_pages": skip_pages,
    }


# ── Extraction helpers ───────────────────────────────────────────

def _detect_header_footer(doc):
    """Detect repeated header/footer text across pages.
    Returns a set of exact strings plus compiled regex patterns."""
    line_counter = Counter()
    # Also track lines by y-position to detect positional headers/footers
    y_lines = Counter()  # (rounded_y, normalized_text) -> count
    all_lines_by_y = {}  # (rounded_y, normalized_text) -> set of actual texts

    sample = min(doc.page_count, 20)
    for pg in range(sample):
        for block in doc[pg].get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                text = "".join(s["text"] for s in line["spans"]).strip()
                if not text or len(text) <= 5:
                    continue
                line_counter[text] += 1
                # Normalize: replace numbers with # to detect pattern repeats
                normed = re.sub(r'\d+', '#', text)
                y_key = (round(line["bbox"][1]), normed)
                y_lines[y_key] += 1
                all_lines_by_y.setdefault(y_key, set()).add(text)

    threshold = max(3, sample * 0.4)
    skip_exact = {text for text, count in line_counter.items()
                  if count >= threshold}

    # Add all variants of pattern-repeated lines at the same y-position
    for (y, normed), count in y_lines.items():
        if count >= threshold:
            skip_exact.update(all_lines_by_y[(y, normed)])

    return skip_exact


def find_question_positions(doc, config):
    """Locate every question number and its (page, y) coordinate."""
    q_regex = config["q_regex"]
    q_font = config["q_font"]
    q_style = config["q_style"]
    skip_lines = config.get("skip_lines", set())
    positions = []

    section_hdr_re = re.compile(
        r'Q\.\s*\d+\s*[–\-]\s*Q\.\s*\d+', re.IGNORECASE)

    skip_pages = config.get("skip_pages", set())

    for page_num in range(doc.page_count):
        if page_num in skip_pages:
            continue
        page = doc[page_num]
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                full_line = "".join(s["text"] for s in line["spans"]).strip()
                # Skip section headers and repeated header/footer text
                if section_hdr_re.search(full_line):
                    continue
                if full_line in skip_lines:
                    continue

                if q_style == "Q.N_line":
                    m = q_regex.match(full_line)
                    if m:
                        positions.append({
                            "id": int(m.group(1)),
                            "page": page_num,
                            "y_start": line["bbox"][1],
                        })
                else:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        m = q_regex.match(text)
                        if not m:
                            continue
                        # For Q.N style, enforce font strictly since
                        # the Q. prefix is distinctive enough.
                        # For N. style, allow any font since different
                        # sections of the same exam may use different fonts.
                        if q_style == "Q.N" and q_font and span["font"] != q_font:
                            continue
                        positions.append({
                            "id": int(m.group(1)),
                            "page": page_num,
                            "y_start": line["bbox"][1],
                        })

    positions.sort(key=lambda p: (p["page"], p["y_start"]))

    # Deduplicate — keep the first occurrence of each question id
    seen = set()
    deduped = []
    for p in positions:
        if p["id"] not in seen:
            deduped.append(p)
            seen.add(p["id"])
    positions = deduped

    # Compute y_end
    for i, pos in enumerate(positions):
        if i + 1 < len(positions) and positions[i + 1]["page"] == pos["page"]:
            pos["y_end"] = positions[i + 1]["y_start"] - 1
        else:
            pos["y_end"] = doc[pos["page"]].rect.height
    return positions


def _collect_spans(doc, page_num, y_start, y_end, config):
    """Collect text spans in a region, filtering noise."""
    page = doc[page_num]
    q_regex = config["q_regex"]
    skip_lines = config.get("skip_lines", set())
    answer_re = config.get("answer_re")
    spans = []

    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        if block["bbox"][3] < y_start or block["bbox"][1] > y_end:
            continue
        for line in block["lines"]:
            ly = line["bbox"][1]
            if ly < y_start or ly > y_end:
                continue
            full_line = "".join(s["text"] for s in line["spans"]).strip()
            # Skip section headers (GATE style)
            if re.match(r'^Q\.\s*\d+\s*[–\-]\s*Q\.\s*\d+', full_line):
                continue
            # Skip section labels (PHYSICS, CHEMISTRY, SECTION-A, etc.)
            if re.match(r'^(SECTION[-\s]?[AB]|PHYSICS|CHEMISTRY|'
                        r'BIOLOGY|BOTANY|ZOOLOGY)\s*$',
                        full_line, re.IGNORECASE):
                continue
            # Skip page-number markers like "- 2 -"
            if re.match(r'^-\s*\d+\s*-$', full_line):
                continue
            # Skip repeated header/footer
            if full_line in skip_lines:
                continue
            # Skip "Answer (X)" lines
            if answer_re and answer_re.match(full_line):
                continue
            for span in line["spans"]:
                if q_regex.match(span["text"].strip()):
                    continue
                spans.append({
                    "text": span["text"],
                    "font": span["font"],
                    "size": span["size"],
                    "y": ly,
                    "x0": span["bbox"][0],
                    "x1": span["bbox"][2],
                    "bbox": span["bbox"],
                })
    return spans


def _merge_spans_into_lines(spans, y_tolerance=3.0):
    if not spans:
        return []
    spans.sort(key=lambda s: (s["y"], s["x0"]))
    lines = []
    current = [spans[0]]
    for s in spans[1:]:
        if abs(s["y"] - current[0]["y"]) <= y_tolerance:
            current.append(s)
        else:
            lines.append(current)
            current = [s]
    lines.append(current)
    return lines


def extract_text_and_equations(doc, page_num, y_start, y_end, config):
    eq_fonts = config["eq_fonts"]
    spans = _collect_spans(doc, page_num, y_start, y_end, config)
    merged = _merge_spans_into_lines(spans)

    text_lines, eq_frags = [], []
    for line_spans in merged:
        line_text, eq_text, has_eq = "", "", False
        ordered = sorted(line_spans, key=lambda s: s["x0"])
        for i, s in enumerate(ordered):
            if i > 0:
                gap = s["x0"] - ordered[i - 1]["x1"]
                if gap > 3:
                    line_text += " "
            line_text += s["text"]
            if s["font"] in eq_fonts or _is_math_font(s["font"]):
                has_eq = True
                eq_text += s["text"]
        stripped = line_text.strip()
        if stripped:
            text_lines.append(stripped)
        if has_eq and eq_text.strip():
            eq_frags.append(eq_text.strip())
    return "\n".join(text_lines), eq_frags


def parse_options(text: str, config) -> dict:
    opt_re = config["opt_re"]
    opt_inline_re = config["opt_inline_re"]
    options = {}
    lines = text.split("\n")

    # Strategy 1: inline — multiple options on one line
    for line in lines:
        stripped = line.strip()
        markers = list(opt_inline_re.finditer(stripped))
        if len(markers) >= 2:
            for idx, m in enumerate(markers):
                key = m.group(1)
                start = m.end()
                end = (markers[idx + 1].start()
                       if idx + 1 < len(markers) else len(stripped))
                options[key] = stripped[start:end].strip()

    # Strategy 2: one option per line
    if not options:
        for line in lines:
            m = opt_re.match(line.strip())
            if m:
                options[m.group(1)] = m.group(2).strip()

    # Strategy 3: letter alone, content on next lines
    if options and any(v == "" for v in options.values()):
        cur, buf = None, []
        for line in lines:
            m = opt_re.match(line.strip())
            if m:
                if cur is not None:
                    options[cur] = " ".join(buf).strip()
                cur = m.group(1)
                rest = m.group(2).strip()
                buf = [rest] if rest else []
            elif cur is not None:
                buf.append(line.strip())
        if cur is not None:
            options[cur] = " ".join(buf).strip()

    return options


def extract_figures(doc, page_num, y_start, y_end, q_id, img_dir, config):
    """Save meaningful images (no watermarks, logos, or lines)."""
    page = doc[page_num]
    skip_sizes = config["skip_img_sizes"]
    figures, saved = [], set()

    # Image blocks in region
    region_blocks = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] == 1:
            by, by2 = block["bbox"][1], block["bbox"][3]
            if by2 >= y_start and by <= y_end:
                region_blocks.append(block)
    if not region_blocks:
        return figures

    for img_info in page.get_images(full=True):
        xref = img_info[0]
        sz = (img_info[2], img_info[3])
        if xref in saved or sz in skip_sizes:
            continue
        try:
            raw = doc.extract_image(xref)
            pil = Image.open(io.BytesIO(raw["image"]))
            iw, ih = pil.size
            if iw <= 0 or ih <= 0:
                continue
            for rb in region_blocks:
                fname = f"q{q_id}_fig{len(figures)+1}.png"
                pil.save(os.path.join(img_dir, fname))
                figures.append({
                    "path": fname,
                    "page": page_num + 1,
                    "bbox": [round(b, 1) for b in rb["bbox"]],
                })
                saved.add(xref)
                region_blocks.remove(rb)
                break
        except Exception:
            continue
    return figures


def render_question_image(doc, pdf_path, page_num, y_start, y_end,
                          q_id, img_dir, config):
    """
    Render the question region at 2x as a clean PNG.
    - Replaces watermarks/logos/separator images with a transparent pixel.
    - Redacts "Answer (N)" text lines so they don't appear in the render.
    Works on a temporary copy to avoid mutating the original document.
    """
    skip_img_sizes = config.get("skip_img_sizes", set())
    answer_re = config.get("answer_re")

    needs_tmp = bool(skip_img_sizes) or bool(answer_re)

    if needs_tmp:
        tmp_doc = fitz.open(pdf_path)
        page = tmp_doc[page_num]

        # Remove unwanted images (watermarks, logos, separator lines)
        if skip_img_sizes:
            tiny = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
            buf = io.BytesIO()
            tiny.save(buf, format="PNG")
            tiny_bytes = buf.getvalue()

            replaced = set()
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in replaced:
                    continue
                if (img_info[2], img_info[3]) in skip_img_sizes:
                    try:
                        page.replace_image(xref, stream=tiny_bytes)
                        replaced.add(xref)
                    except Exception:
                        pass

        # Redact answer lines (e.g. "Answer (2)")
        if answer_re:
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    ly = line["bbox"][1]
                    if ly < y_start or ly > y_end:
                        continue
                    full = "".join(s["text"] for s in line["spans"]).strip()
                    if answer_re.match(full):
                        page.add_redact_annot(fitz.Rect(line["bbox"]),
                                              fill=(1, 1, 1))
            page.apply_redactions()
    else:
        tmp_doc = None
        page = doc[page_num]

    clip = fitz.Rect(page.rect.x0, max(0, y_start - 5),
                     page.rect.x1, min(page.rect.height, y_end + 5))
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
    fname = f"q{q_id}_rendered.png"
    pix.save(os.path.join(img_dir, fname))

    if tmp_doc:
        tmp_doc.close()
    return fname


def classify_question(text, options, config):
    low = text.lower()
    if options and len(options) >= 2:
        return "MCQ"
    if any(kw in low for kw in ["nearest integer", "______", "___",
                                  "nat type", "numerical answer"]):
        return "Numerical"
    opt_inline_re = config["opt_inline_re"]
    if opt_inline_re.search(text):
        return "MCQ"
    return "Numerical"


def extract_metadata(doc):
    # Scan first 2 pages for metadata
    raw_text = ""
    for pg in range(min(2, doc.page_count)):
        raw_text += doc[pg].get_text() + "\n"

    meta = {
        "exam": "", "subject": "", "topic": "",
        "date": "", "total_marks": "", "duration": "",
    }
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

    for line in lines[:30]:
        low = line.lower()
        if not meta["exam"]:
            if "jee" in low:
                meta["exam"] = "JEE Mains"
            elif "gate" in low:
                meta["exam"] = "GATE"
            elif "neet" in low:
                meta["exam"] = "NEET"
            elif "organizing institute" in low:
                meta["exam"] = "GATE"
            elif ("institute" in low or "academy" in low
                    or "coaching" in low):
                meta["exam"] = line
        if low.startswith("subjects") or low.startswith("subject"):
            meta["subject"] = line.split(":")[-1].strip()
        elif not meta["subject"] and "(" in line and ")" in line:
            m = re.match(r'^(.+?)\s*\((\w+)\)\s*$', line)
            if m:
                meta["subject"] = m.group(1).strip()
        if "m.m" in low or "total marks" in low:
            m = re.search(r'(\d+)', line)
            if m:
                meta["total_marks"] = m.group(1)
        if low.startswith("date"):
            meta["date"] = line.split(":")[-1].strip()
        if low.startswith("hours") or "duration" in low:
            val = line.split(":")[-1].strip() if ":" in line else line
            if not meta["duration"]:
                meta["duration"] = val
        if "time" in low and ":" in line and not meta["duration"]:
            val = line.split(":", 1)[-1].strip()
            if val and any(c.isdigit() for c in val):
                meta["duration"] = val
        if "page" in low and "of" in low:
            m = re.search(r'page\s+\d+\s+of\s+(\d+)', low)
            if m:
                meta["total_pages"] = m.group(1)
        if not meta["topic"]:
            skip_kws = ("subject", "total", "date", "hour", "duration",
                        "page", "institute", "section", "q.", "general",
                        "organizing", "mark", "important", "instruction",
                        "corporate", "question", "answer", "time",
                        "m.m", "booklet", "code", "academy", "coaching",
                        "office", "phone", "ph.", "for")
            # Skip page number markers like "- 1 -"
            if re.match(r'^-\s*\d+\s*-$', line):
                continue
            if (not any(k in low for k in skip_kws)
                    and not line.isdigit() and 3 < len(line) < 80):
                meta["topic"] = line
    return meta


# ── Public API ───────────────────────────────────────────────────

def process_pdf(
    pdf_path: str,
    output_dir: str | None = None,
    on_progress: Optional[Callable[[dict], None]] = None,
):
    """Full extraction pipeline.  Auto-detects PDF format.

    If `on_progress` is provided, it is called with progress event dicts
    ({\"event\": \"progress-start\"|\"progress-question\", \"data\": {...}}) so an
    SSE endpoint can stream extraction progress to the client.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="qextract_")

    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    doc = fitz.open(pdf_path)

    # Step 1: auto-detect
    config = detect_pdf_format(doc)
    config["skip_lines"] = _detect_header_footer(doc)
    meta = extract_metadata(doc)

    # Step 2: find questions
    positions = find_question_positions(doc, config)
    if not positions:
        doc.close()
        raise ValueError(
            "Could not find any questions in this PDF. "
            "Supported formats: Q.1/Q.2/... or 1./2./... numbering."
        )

    # Announce the target counts up-front so the client can render a real
    # progress bar instead of an indeterminate spinner.
    if on_progress is not None:
        on_progress({
            "event": "progress-start",
            "data": {
                "total_pages": doc.page_count,
                "total_questions": len(positions),
            },
        })

    # Step 3: extract each question
    questions = []
    opt_re = config["opt_re"]
    answer_re = config.get("answer_re")

    for i, pos in enumerate(positions, start=1):
        qid, pg = pos["id"], pos["page"]
        ys, ye = pos["y_start"], pos["y_end"]

        full_text, eq_raw = extract_text_and_equations(
            doc, pg, ys, ye, config)
        options = parse_options(full_text, config)
        q_type = classify_question(full_text, options, config)

        equations = [{"raw_text": r, "latex": _unicode_to_latex(r)}
                     for r in eq_raw]
        figures = extract_figures(doc, pg, ys, ye, qid, img_dir, config)
        rendered = render_question_image(doc, pdf_path, pg, ys, ye, qid,
                                         img_dir, config)

        # Clean question text: drop options and answer lines
        lines = full_text.split("\n")
        clean, hit_opts = [], False
        for ln in lines:
            if opt_re.match(ln.strip()):
                hit_opts = True
            if answer_re and answer_re.match(ln.strip()):
                continue
            if not hit_opts:
                clean.append(ln)
        question_text = "\n".join(clean).strip() or full_text.strip()

        questions.append({
            "id": qid,
            "text": question_text,
            "type": q_type,
            "options": options,
            "equations": equations,
            "figures": figures,
            "rendered_image": rendered,
            "raw_text": full_text,
            "page": pg + 1,
        })

        # Report per-question progress as each question is completed.
        if on_progress is not None:
            on_progress({
                "event": "progress-question",
                "data": {
                    "index": i,
                    "count": i,
                    "total": len(positions),
                    "page": pg + 1,
                },
            })

    doc.close()

    data = {
        **meta,
        "source_pdf": os.path.basename(pdf_path),
        "total_questions": len(questions),
        "questions": questions,
    }

    json_path = os.path.join(output_dir, "questions_output.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    zip_path = os.path.join(output_dir, "extracted_questions.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, "questions_output.json")
        for fname in os.listdir(img_dir):
            zf.write(os.path.join(img_dir, fname), f"images/{fname}")

    return {
        "json_path": json_path,
        "img_dir": img_dir,
        "zip_path": zip_path,
        "data": data,
    }
