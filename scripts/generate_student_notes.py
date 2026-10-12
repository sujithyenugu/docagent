#!/usr/bin/env python3
"""Generate synthetic Indian student note images for Use Case A evaluation.

Pipeline
--------
1. Generate curriculum text content in Hindi (Devanagari) and English using an
   LLM-style template system (no external LLM required for the base version).
2. Render each page as a PNG using Pillow with appropriate fonts.
3. Apply distortions: ruled-paper background, rotation ±5°, perspective warp,
   Gaussian blur, paper texture.
4. Write ground truth JSON alongside each image.

Output structure
----------------
benchmarks/datasets/student_notes/
  train/
    page_0000.png
    page_0000.json       # {"text": "...", "script": "hindi|english|mixed"}
    ...
  eval/
    page_0400.png
    page_0400.json
    ...

Usage
-----
    python scripts/generate_student_notes.py --output benchmarks/datasets/student_notes \
        --count 500 --seed 42

Requirements (install separately — not in main requirements.txt):
    pip install pillow trdg faker
    # Fonts: Noto Sans Devanagari, Hind (download from Google Fonts)
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curriculum content templates
# ---------------------------------------------------------------------------

_HINDI_TOPICS = [
    ("विज्ञान", [
        "प्रकाश एक प्रकार की ऊर्जा है जो सीधी रेखाओं में चलती है।",
        "ध्वनि तरंगें माध्यम के कणों के कंपन से उत्पन्न होती हैं।",
        "पानी का रासायनिक सूत्र H₂O है।",
        "कोशिका जीवन की मूल इकाई है।",
        "पृथ्वी सूर्य के चारों ओर 365 दिनों में एक चक्कर लगाती है।",
        "गुरुत्वाकर्षण वह बल है जो वस्तुओं को पृथ्वी की ओर खींचता है।",
        "प्रकाश संश्लेषण की क्रिया में पौधे सूर्य के प्रकाश का उपयोग करते हैं।",
        "रक्त में लाल रक्त कणिकाएं ऑक्सीजन का परिवहन करती हैं।",
    ]),
    ("गणित", [
        "त्रिभुज के तीनों कोणों का योग 180° होता है।",
        "वृत्त की परिधि = 2πr",
        "पाइथागोरस प्रमेय: a² + b² = c²",
        "1 से 10 तक के प्राकृत संख्याओं का योग = 55",
        "किसी संख्या का वर्गमूल उस संख्या को आधी घात पर उठाने के बराबर है।",
        "समांतर श्रेढ़ी का nth पद = a + (n-1)d",
        "दो सम संख्याओं का गुणनफल सम होता है।",
    ]),
    ("इतिहास", [
        "भारत को 15 अगस्त 1947 को स्वतंत्रता मिली।",
        "महात्मा गांधी को 'राष्ट्रपिता' कहा जाता है।",
        "मुगल साम्राज्य की स्थापना बाबर ने 1526 में की थी।",
        "अशोक ने कलिंग युद्ध के बाद बौद्ध धर्म अपना लिया।",
        "सिंधु घाटी सभ्यता विश्व की प्राचीनतम सभ्यताओं में से एक है।",
    ]),
    ("भूगोल", [
        "भारत की सबसे लंबी नदी गंगा है।",
        "हिमालय पर्वत एशिया का सबसे ऊंचा पर्वत है।",
        "भारत में कुल 28 राज्य और 8 केंद्र शासित प्रदेश हैं।",
        "थार मरुस्थल राजस्थान में स्थित है।",
        "भारत का क्षेत्रफल लगभग 32.87 लाख वर्ग किलोमीटर है।",
    ]),
]

_ENGLISH_TOPICS = [
    ("Science", [
        "Photosynthesis is the process by which plants make food using sunlight.",
        "Newton's First Law: An object at rest stays at rest unless acted upon by a force.",
        "The human body has 206 bones in an adult skeleton.",
        "Water boils at 100°C and freezes at 0°C at standard pressure.",
        "DNA carries the genetic information of living organisms.",
        "The speed of light in vacuum is approximately 3×10⁸ m/s.",
        "Acids have a pH less than 7; bases have a pH greater than 7.",
    ]),
    ("Mathematics", [
        "The area of a circle = πr²",
        "Simple Interest = (P × R × T) / 100",
        "Quadratic formula: x = (-b ± √(b²-4ac)) / 2a",
        "Probability of an event = Favorable outcomes / Total outcomes",
        "Sum of angles in a polygon = (n-2) × 180°",
        "The HCF of two numbers divides both numbers exactly.",
    ]),
    ("English Grammar", [
        "A noun is a word that names a person, place, thing, or idea.",
        "A verb expresses action or a state of being.",
        "Adjectives modify nouns; adverbs modify verbs, adjectives, or other adverbs.",
        "Active voice: The dog bit the man. Passive: The man was bitten by the dog.",
        "Direct speech uses quotation marks to show exact words spoken.",
    ]),
    ("Social Studies", [
        "The Indian Constitution came into effect on 26 January 1950.",
        "The Parliament of India consists of Lok Sabha and Rajya Sabha.",
        "The Green Revolution of the 1960s increased agricultural production in India.",
        "UNICEF works to protect children's rights worldwide.",
        "The Prime Meridian passes through Greenwich, England.",
    ]),
]

# Mixed (code-switched) page headers — Hindi heading + English body
_MIXED_HEADERS = [
    "विज्ञान Notes — Chapter 3",
    "गणित — Algebra Basics",
    "इतिहास — Modern India",
    "भूगोल — Indian Rivers",
    "Biology — कोशिका विज्ञान",
    "Physics — गति के नियम",
    "Chemistry — रासायनिक अभिक्रियाएं",
    "Economics — बाज़ार और कीमत",
]


# ---------------------------------------------------------------------------
# Page content generators
# ---------------------------------------------------------------------------

def _generate_hindi_page(rng: random.Random) -> tuple[str, str]:
    """Return (text, script='hindi')."""
    subject, lines = rng.choice(_HINDI_TOPICS)
    n = rng.randint(4, 8)
    selected = rng.sample(lines, min(n, len(lines)))
    date = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2025"
    text = f"{subject}\nदिनांक: {date}\n\n" + "\n\n".join(
        f"{i + 1}. {line}" for i, line in enumerate(selected)
    )
    return text, "hindi"


def _generate_english_page(rng: random.Random) -> tuple[str, str]:
    """Return (text, script='english')."""
    subject, lines = rng.choice(_ENGLISH_TOPICS)
    n = rng.randint(4, 8)
    selected = rng.sample(lines, min(n, len(lines)))
    date = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2025"
    text = f"{subject}\nDate: {date}\n\n" + "\n\n".join(
        f"{i + 1}. {line}" for i, line in enumerate(selected)
    )
    return text, "english"


def _generate_mixed_page(rng: random.Random) -> tuple[str, str]:
    """Return code-switched page (Hindi heading + English body)."""
    header = rng.choice(_MIXED_HEADERS)
    _, en_lines = rng.choice(_ENGLISH_TOPICS)
    _, hi_lines = rng.choice(_HINDI_TOPICS)
    en_selected = rng.sample(en_lines, min(3, len(en_lines)))
    hi_selected = rng.sample(hi_lines, min(2, len(hi_lines)))
    date = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2025"
    text = (
        f"{header}\nDate: {date}\n\n"
        + "\n\n".join(f"{i + 1}. {line}" for i, line in enumerate(en_selected))
        + "\n\n"
        + "\n\n".join(f"→ {line}" for line in hi_selected)
    )
    return text, "mixed"


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

def _render_page(text: str, script: str, rng: random.Random, font_dir: Path | None) -> bytes:
    """Render *text* as a notebook page PNG and return bytes.

    Falls back to a white image with PIL's default font if custom fonts are
    unavailable (allows running without extra dependencies).
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageFont
    except ImportError:
        logger.error("Pillow is required: pip install pillow")
        sys.exit(1)

    width, height = 794, 1123  # A4 at 96 dpi

    # Ruled paper background
    img = Image.new("RGB", (width, height), color=(255, 253, 240))
    draw = ImageDraw.Draw(img)
    for y in range(60, height, 28):
        draw.line([(50, y), (width - 30, y)], fill=(180, 200, 230), width=1)
    # Margin line
    draw.line([(90, 0), (90, height)], fill=(220, 150, 150), width=2)

    # Font selection
    font_size = 18
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        if font_dir:
            if script in ("hindi", "mixed"):
                font_path = font_dir / "NotoSansDevanagari-Regular.ttf"
            else:
                font_path = font_dir / "Hind-Regular.ttf"
            if not font_path.exists():
                font_path = font_dir / "NotoSansDevanagari-Regular.ttf"
            font = ImageFont.truetype(str(font_path), font_size)
        else:
            raise OSError("no font dir")
    except (OSError, AttributeError):
        font = ImageFont.load_default()
        font_size = 12

    # Draw text
    x, y = 100, 70
    line_height = font_size + 10
    for line in text.split("\n"):
        if y + line_height > height - 30:
            break
        draw.text((x, y), line, fill=(10, 10, 60), font=font)
        y += line_height

    # Distortions
    angle = rng.uniform(-5, 5)
    img = img.rotate(angle, fillcolor=(255, 253, 240), expand=False)

    blur_radius = rng.uniform(0.2, 0.8)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Perspective warp (simple crop simulation)
    if rng.random() < 0.4:
        margin = rng.randint(5, 20)
        img = img.crop((margin, margin, width - margin, height - margin))
        img = img.resize((width, height), Image.LANCZOS)

    # Return PNG bytes
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(output_dir: Path, count: int, seed: int) -> None:
    rng = random.Random(seed)

    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Font directory — user can set DA_FONT_DIR env var
    font_dir_env = os.environ.get("DA_FONT_DIR")
    font_dir = Path(font_dir_env) if font_dir_env else None

    # Target distribution: ~250 Hindi, ~150 English, ~100 mixed
    ratios = [("hindi", 0.50), ("english", 0.30), ("mixed", 0.20)]
    scripts = []
    for script, ratio in ratios:
        scripts.extend([script] * int(count * ratio))
    while len(scripts) < count:
        scripts.append("hindi")
    rng.shuffle(scripts)

    train_count = int(count * 0.8)
    generated = 0

    for idx, script in enumerate(scripts):
        split = "train" if idx < train_count else "eval"
        out_dir = train_dir if split == "train" else eval_dir
        stem = f"page_{idx:04d}"

        if script == "hindi":
            text, _ = _generate_hindi_page(rng)
        elif script == "english":
            text, _ = _generate_english_page(rng)
        else:
            text, _ = _generate_mixed_page(rng)

        png_bytes = _render_page(text, script, rng, font_dir)
        (out_dir / f"{stem}.png").write_bytes(png_bytes)
        (out_dir / f"{stem}.json").write_text(
            json.dumps({"text": text, "script": script, "index": idx}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        generated += 1
        if generated % 50 == 0:
            logger.info("Generated %d / %d pages", generated, count)

    logger.info(
        "Done. %d pages → %s (train: %d, eval: %d)",
        count,
        output_dir,
        train_count,
        count - train_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic student notes dataset")
    parser.add_argument(
        "--output",
        default="benchmarks/datasets/student_notes",
        help="Output directory (default: benchmarks/datasets/student_notes)",
    )
    parser.add_argument("--count", type=int, default=500, help="Total pages to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    generate(Path(args.output), args.count, args.seed)


if __name__ == "__main__":
    main()
