#!/usr/bin/env python3
"""Generate synthetic Indian student assignment and exam texts for teacher grading evaluation.

Pipeline
--------
1. Generate answer texts in Hindi (Devanagari), English, and mixed (code-switched).
2. Include realistic student errors: spelling mistakes, partial answers, struck-through
   corrections (represented as [STRIKETHROUGH: word]).
3. Each page has a paired answer key for grading evaluation.
4. Render each page as a PNG using Pillow with ruled-paper background and distortions.
5. Write ground truth JSON (text, answer_key, subject, grade_level, marks_possible)
   alongside each image.

Output structure
----------------
benchmarks/datasets/student_assignments/
  train/
    page_0000.png
    page_0000.json   # {text, script, subject, grade_level, answer_key, marks_possible}
    ...
  eval/
    page_0400.png
    page_0400.json
    ...

Usage
-----
    python scripts/generate_student_assignments.py \
        --output benchmarks/datasets/student_assignments \
        --count 500 --seed 42

    # Custom font directory (set DA_FONT_DIR env var or pass --font-dir)
    python scripts/generate_student_assignments.py --font-dir /path/to/fonts

Requirements (install separately):
    pip install pillow faker
    # Fonts: Noto Sans Devanagari, Hind (from Google Fonts)
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
# Assignment content — (question, correct_answer_keywords, model_answer)
# ---------------------------------------------------------------------------

_ENGLISH_ASSIGNMENTS = [
    {
        "subject": "Science",
        "grade_level": "Class 8",
        "questions": [
            {
                "question": "What is photosynthesis?",
                "answer_keywords": ["sunlight", "chlorophyll", "glucose", "oxygen", "carbon dioxide"],
                "model_answer": "Photosynthesis is the process by which green plants make food using sunlight, water, and carbon dioxide. Chlorophyll absorbs sunlight. The products are glucose and oxygen.",
                "marks": 4,
            },
            {
                "question": "State Newton's First Law of Motion.",
                "answer_keywords": ["rest", "motion", "external force", "inertia"],
                "model_answer": "An object at rest remains at rest, and an object in motion continues in motion with the same speed and direction, unless acted upon by an external unbalanced force. This property is called inertia.",
                "marks": 3,
            },
            {
                "question": "What is the chemical formula for water?",
                "answer_keywords": ["H2O", "hydrogen", "oxygen"],
                "model_answer": "The chemical formula for water is H2O. It consists of two hydrogen atoms and one oxygen atom.",
                "marks": 2,
            },
        ],
    },
    {
        "subject": "Mathematics",
        "grade_level": "Class 10",
        "questions": [
            {
                "question": "Find the roots of x² - 5x + 6 = 0.",
                "answer_keywords": ["x=2", "x=3", "factors", "(x-2)(x-3)"],
                "model_answer": "x² - 5x + 6 = 0\n(x-2)(x-3) = 0\nx = 2 or x = 3\nThe roots are x = 2 and x = 3.",
                "marks": 4,
            },
            {
                "question": "What is the area of a circle with radius 7 cm?",
                "answer_keywords": ["πr²", "154", "22/7", "area"],
                "model_answer": "Area = πr² = (22/7) × 7 × 7 = 154 cm²",
                "marks": 3,
            },
            {
                "question": "Define the Pythagorean theorem.",
                "answer_keywords": ["right triangle", "hypotenuse", "a²+b²=c²", "square"],
                "model_answer": "In a right-angled triangle, the square of the hypotenuse (c) is equal to the sum of the squares of the other two sides. a² + b² = c²",
                "marks": 3,
            },
        ],
    },
    {
        "subject": "History",
        "grade_level": "Class 9",
        "questions": [
            {
                "question": "When did India gain independence?",
                "answer_keywords": ["1947", "August 15", "British", "independence"],
                "model_answer": "India gained independence on August 15, 1947, from British colonial rule.",
                "marks": 2,
            },
            {
                "question": "Who is called the Father of the Nation?",
                "answer_keywords": ["Mahatma Gandhi", "Gandhi", "non-violence", "Bapu"],
                "model_answer": "Mahatma Gandhi is called the Father of the Nation. He led the non-violent independence movement against British rule.",
                "marks": 2,
            },
            {
                "question": "What was the Dandi March?",
                "answer_keywords": ["salt", "Gandhi", "1930", "tax", "civil disobedience"],
                "model_answer": "The Dandi March was a 240-mile march led by Mahatma Gandhi in 1930 to protest the British salt tax. It was a key act of civil disobedience in India's independence movement.",
                "marks": 4,
            },
        ],
    },
    {
        "subject": "English",
        "grade_level": "Class 7",
        "questions": [
            {
                "question": "What is a noun? Give two examples.",
                "answer_keywords": ["person", "place", "thing", "idea", "examples"],
                "model_answer": "A noun is a word that names a person, place, thing, or idea. Examples: book, Delhi.",
                "marks": 3,
            },
            {
                "question": "Convert to passive voice: 'The teacher praised the student.'",
                "answer_keywords": ["was praised", "by the teacher", "passive"],
                "model_answer": "The student was praised by the teacher.",
                "marks": 2,
            },
        ],
    },
]

_HINDI_ASSIGNMENTS = [
    {
        "subject": "विज्ञान",
        "grade_level": "कक्षा 8",
        "questions": [
            {
                "question": "प्रकाश संश्लेषण क्या है?",
                "answer_keywords": ["सूर्य प्रकाश", "क्लोरोफिल", "ग्लूकोज", "ऑक्सीजन", "कार्बन डाइऑक्साइड"],
                "model_answer": "प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा हरे पौधे सूर्य प्रकाश, जल और कार्बन डाइऑक्साइड का उपयोग करके भोजन बनाते हैं। क्लोरोफिल सूर्य प्रकाश को अवशोषित करता है। उत्पाद ग्लूकोज और ऑक्सीजन हैं।",
                "marks": 4,
            },
            {
                "question": "गुरुत्वाकर्षण बल क्या है?",
                "answer_keywords": ["आकर्षण", "पृथ्वी", "द्रव्यमान", "गिरना"],
                "model_answer": "गुरुत्वाकर्षण बल वह बल है जो दो वस्तुओं के बीच उनके द्रव्यमान के कारण उत्पन्न होता है। पृथ्वी का गुरुत्वाकर्षण बल वस्तुओं को अपनी ओर खींचता है।",
                "marks": 3,
            },
        ],
    },
    {
        "subject": "गणित",
        "grade_level": "कक्षा 10",
        "questions": [
            {
                "question": "वृत्त का क्षेत्रफल का सूत्र लिखिए।",
                "answer_keywords": ["πr²", "त्रिज्या", "पाई", "क्षेत्रफल"],
                "model_answer": "वृत्त का क्षेत्रफल = πr² जहाँ r वृत्त की त्रिज्या है।",
                "marks": 2,
            },
            {
                "question": "समांतर श्रेढ़ी का सार्वान्तर क्या है?",
                "answer_keywords": ["सार्वान्तर", "अगला पद", "पहला पद", "d"],
                "model_answer": "समांतर श्रेढ़ी में प्रत्येक पद और उसके पिछले पद के बीच का अंतर समान होता है। इसे सार्वान्तर (d) कहते हैं। d = a₂ - a₁",
                "marks": 3,
            },
        ],
    },
    {
        "subject": "इतिहास",
        "grade_level": "कक्षा 9",
        "questions": [
            {
                "question": "भारतीय स्वतंत्रता संग्राम में महात्मा गांधी की भूमिका क्या थी?",
                "answer_keywords": ["अहिंसा", "सत्याग्रह", "असहयोग आंदोलन", "दांडी"],
                "model_answer": "महात्मा गांधी ने अहिंसा और सत्याग्रह के मार्ग से भारतीय स्वतंत्रता संग्राम का नेतृत्व किया। उन्होंने असहयोग आंदोलन और दांडी मार्च जैसे आंदोलन चलाए।",
                "marks": 5,
            },
        ],
    },
]

_MIXED_ASSIGNMENTS = [
    {
        "subject": "Science / विज्ञान",
        "grade_level": "Class 8 / कक्षा 8",
        "questions": [
            {
                "question": "Define photosynthesis. / प्रकाश संश्लेषण की परिभाषा दीजिए।",
                "answer_keywords": ["sunlight", "chlorophyll", "glucose", "सूर्य प्रकाश", "ग्लूकोज"],
                "model_answer": "Photosynthesis is the process by which plants make food.\nप्रकाश संश्लेषण से पौधे भोजन बनाते हैं।\nIngredients: sunlight, CO2, water → glucose + oxygen",
                "marks": 4,
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Student error simulation
# ---------------------------------------------------------------------------

_SPELLING_ERRORS = {
    "photosynthesis": ["photosinthesis", "photosinthasis", "fotosynthesis"],
    "chlorophyll": ["clorophyll", "chlorofill", "chlorophyl"],
    "nitrogen": ["nitogen", "nitrgen", "nitrogn"],
    "inertia": ["inertia", "inercia", "inertea"],
    "hypothesis": ["hypothisis", "hypothesys", "hypothes"],
    "atmosphere": ["atmospher", "atmoshpere", "atmosphire"],
    "photosynthesis": ["photsynthesis", "photosyntesis"],
    "independence": ["independance", "independnce", "indpendence"],
    "parliament": ["parliment", "parlaiment", "parliment"],
}


def _apply_student_errors(text: str, rng: random.Random, error_rate: float = 0.15) -> str:
    """Apply realistic student errors at a given rate."""
    words = text.split()
    result = []
    for word in words:
        lower = word.lower().rstrip(".,;:?!")
        punct = word[len(lower):]
        if lower in _SPELLING_ERRORS and rng.random() < error_rate:
            misspelled = rng.choice(_SPELLING_ERRORS[lower])
            # Preserve capitalisation
            if word[0].isupper():
                misspelled = misspelled.capitalize()
            result.append(misspelled + punct)
        elif rng.random() < error_rate * 0.3:
            # Randomly drop a character
            if len(lower) > 4:
                idx = rng.randint(1, len(lower) - 2)
                typo = lower[:idx] + lower[idx + 1:]
                if word[0].isupper():
                    typo = typo.capitalize()
                result.append(typo + punct)
            else:
                result.append(word)
        else:
            result.append(word)
    return " ".join(result)


def _add_strikethroughs(text: str, rng: random.Random, rate: float = 0.05) -> str:
    """Add occasional struck-through corrections."""
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        if rng.random() < rate and i + 1 < len(words):
            # Replace a word with a struck-through version then the correct next word
            result.append(f"[STRIKETHROUGH: {words[i]}]")
            result.append(words[i + 1])
            i += 2
        else:
            result.append(words[i])
            i += 1
    return " ".join(result)


# ---------------------------------------------------------------------------
# Page text generation
# ---------------------------------------------------------------------------

def _build_page_text(assignment: dict, rng: random.Random) -> tuple[str, str, dict, float]:
    """Build student answer text for one assignment.

    Returns: (text, script, answer_key, marks_possible)
    """
    questions = rng.sample(assignment["questions"], min(len(assignment["questions"]), rng.randint(2, 3)))
    subject = assignment["subject"]
    grade = assignment["grade_level"]

    # Detect script
    has_devanagari = any(ord(c) > 0x0900 for c in subject)
    if has_devanagari and "/" not in subject:
        script = "hindi"
    elif "/" in subject:
        script = "mixed"
    else:
        script = "english"

    date_str = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2025"
    header = (
        f"{subject}\n{grade}\n"
        + (f"Date: {date_str}" if script == "english" else f"दिनांक: {date_str}")
        + "\n\n"
    )

    answer_text_parts = []
    answer_key = {}
    total_marks = 0.0
    error_rate = rng.uniform(0.05, 0.25)

    for idx, q in enumerate(questions, 1):
        q_num = f"Q{idx}." if script == "english" else f"प्रश्न {idx}."
        answer_text_parts.append(f"{q_num} {q['question']}")

        # Student may answer partially or fully
        completeness = rng.random()
        if completeness < 0.15:
            # Blank or very short non-answer
            student_answer = "I don't know." if script == "english" else "पता नहीं।"
        elif completeness < 0.40:
            # Partial: only first sentence of model answer
            full = q["model_answer"]
            student_answer = full.split(".")[0] + "."
            student_answer = _apply_student_errors(student_answer, rng, error_rate)
        else:
            # Mostly complete with errors
            student_answer = _apply_student_errors(q["model_answer"], rng, error_rate)
            student_answer = _add_strikethroughs(student_answer, rng, rate=0.04)

        answer_text_parts.append(f"Ans: {student_answer}\n")
        answer_key[f"q{idx}"] = {
            "keywords": q["answer_keywords"],
            "marks": q["marks"],
        }
        total_marks += q["marks"]

    text = header + "\n".join(answer_text_parts)
    return text, script, answer_key, total_marks


def _generate_page(rng: random.Random) -> tuple[str, str, dict, float]:
    """Pick a random assignment and generate one student page."""
    pool = _ENGLISH_ASSIGNMENTS + _HINDI_ASSIGNMENTS + _MIXED_ASSIGNMENTS
    assignment = rng.choice(pool)
    return _build_page_text(assignment, rng)


# ---------------------------------------------------------------------------
# Image rendering (Pillow)
# ---------------------------------------------------------------------------

def _render_page(text: str, script: str, rng: random.Random, font_dir: Path | None) -> bytes:
    """Render student answer text as a ruled-paper PNG.

    Falls back to PIL's default bitmap font if custom fonts are unavailable.
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageFont
    except ImportError:
        logger.error("Pillow is required: pip install pillow")
        sys.exit(1)

    import io as _io

    width, height = 794, 1123  # A4 at 96 dpi

    # Ruled paper background (slightly yellowed)
    img = Image.new("RGB", (width, height), color=(255, 252, 235))
    draw = ImageDraw.Draw(img)
    for y in range(60, height, 28):
        draw.line([(50, y), (width - 30, y)], fill=(180, 200, 230), width=1)
    # Red margin line
    draw.line([(95, 0), (95, height)], fill=(220, 140, 140), width=2)

    font_size = 17
    try:
        if font_dir:
            font_name = (
                "NotoSansDevanagari-Regular.ttf"
                if script in ("hindi", "mixed")
                else "Hind-Regular.ttf"
            )
            font_path = font_dir / font_name
            if not font_path.exists():
                font_path = next(font_dir.glob("*.ttf"), None)
            if font_path and font_path.exists():
                from PIL import ImageFont as _IF
                font = _IF.truetype(str(font_path), font_size)
            else:
                raise OSError("font not found")
        else:
            raise OSError("no font dir")
    except (OSError, AttributeError):
        from PIL import ImageFont as _IF
        font = _IF.load_default()
        font_size = 11

    # Draw text
    x, y = 105, 65
    line_height = font_size + 11
    for line in text.split("\n"):
        if y + line_height > height - 30:
            break
        draw.text((x, y), line, fill=(15, 15, 60), font=font)
        y += line_height

    # Distortions
    angle = rng.uniform(-4, 4)
    img = img.rotate(angle, fillcolor=(255, 252, 235), expand=False)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.7)))

    # Perspective crop (phone photo simulation, 40% of images)
    if rng.random() < 0.4:
        m = rng.randint(8, 25)
        img = img.crop((m, m, width - m, height - m)).resize((width, height), Image.LANCZOS)

    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(output_dir: Path, count: int, seed: int, font_dir: Path | None) -> None:
    rng = random.Random(seed)

    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_count = int(count * 0.8)

    for idx in range(count):
        split = "train" if idx < train_count else "eval"
        out_dir = train_dir if split == "train" else eval_dir
        stem = f"page_{idx:04d}"

        text, script, answer_key, marks_possible = _generate_page(rng)

        # Determine subject from text header (first line)
        subject_line = text.split("\n")[0]

        png_bytes = _render_page(text, script, rng, font_dir)
        (out_dir / f"{stem}.png").write_bytes(png_bytes)
        (out_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "text": text,
                    "script": script,
                    "subject": subject_line,
                    "grade_level": text.split("\n")[1] if len(text.split("\n")) > 1 else "",
                    "answer_key": answer_key,
                    "marks_possible": marks_possible,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if (idx + 1) % 50 == 0:
            logger.info("Generated %d / %d pages", idx + 1, count)

    logger.info(
        "Done. %d pages → %s (train: %d, eval: %d)",
        count,
        output_dir,
        train_count,
        count - train_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic student assignment dataset for teacher grading"
    )
    parser.add_argument(
        "--output",
        default="benchmarks/datasets/student_assignments",
        help="Output directory (default: benchmarks/datasets/student_assignments)",
    )
    parser.add_argument("--count", type=int, default=500, help="Total pages to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--font-dir",
        default=os.environ.get("DA_FONT_DIR"),
        help="Directory containing TTF fonts (overrides DA_FONT_DIR env var)",
    )
    args = parser.parse_args()

    font_dir = Path(args.font_dir) if args.font_dir else None
    generate(Path(args.output), args.count, args.seed, font_dir)


if __name__ == "__main__":
    main()
