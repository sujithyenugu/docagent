#!/usr/bin/env python3
"""Generate 800 synthetic Indian student document images for teacher grading pipeline testing.

200 students × 4 variations × 5 document types (but capped at --count).

Document types
--------------
admit_card          — student admit card (board exam / school exam)
roll_number_sheet   — class roll number sheet (teacher reference document)
answer_booklet      — answer booklet cover page
cbse_marksheet      — CBSE Class 10/12 mark sheet
school_id_card      — school ID card

Variations
----------
clean       — clean flat scan
photo       — perspective warp + blur (phone photo)
lowlight    — darkened exposure
glare       — lamination glare overlay

All identity data is faker-generated — NO real PII.

Usage
-----
    python scripts/generate_synthetic_student_docs.py \
        --count 800 --seed 42

    python scripts/generate_synthetic_student_docs.py \
        --output benchmarks/datasets/synthetic_student_docs \
        --count 200 --seed 0

Output
------
benchmarks/datasets/synthetic_student_docs/
  eval/
    stu_000_admit_card_clean.png
    stu_000_admit_card_clean.json   # all field values as ground truth
    stu_000_admit_card_photo.png
    ...

Requirements (install separately):
    pip install pillow faker
    # For valid Aadhaar checksums: included inline (Verhoeff algorithm)
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verhoeff checksum for Aadhaar numbers
# ---------------------------------------------------------------------------

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _verhoeff_checksum(number: str) -> int:
    """Return the Verhoeff check digit for a digit string."""
    c = 0
    for i, d in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(d)]]
    return _VERHOEFF_INV[c]


def _generate_aadhaar(rng: random.Random) -> str:
    """Generate a valid-format 12-digit Aadhaar number (Verhoeff checksum)."""
    # First digit must not be 0 or 1
    digits = [str(rng.randint(2, 9))] + [str(rng.randint(0, 9)) for _ in range(10)]
    check = _verhoeff_checksum("".join(digits))
    digits.append(str(check))
    num = "".join(digits)
    return f"{num[:4]} {num[4:8]} {num[8:]}"


def _generate_pan(rng: random.Random) -> str:
    """Generate a PAN-format number: AAAAA0000A."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(rng.choices(letters, k=5))
        + "".join(str(rng.randint(0, 9)) for _ in range(4))
        + rng.choice(letters)
    )


def _generate_roll_number(rng: random.Random, board: str = "CBSE") -> str:
    if board == "CBSE":
        return f"{rng.randint(10, 99)}{rng.randint(10000000, 99999999):08d}"
    return f"{rng.randint(100000, 999999)}"


def _generate_admission_number(rng: random.Random) -> str:
    return f"ADM{rng.randint(2020, 2025)}{rng.randint(1000, 9999)}"


# ---------------------------------------------------------------------------
# Fake student identity data
# ---------------------------------------------------------------------------

_FIRST_NAMES_MALE = [
    "Aarav", "Aditya", "Arjun", "Dhruv", "Ishaan", "Karan", "Mohit",
    "Naman", "Rahul", "Rohan", "Sarthak", "Shivam", "Tanmay", "Vivek",
    "Ankit", "Deepak", "Gaurav", "Harshit", "Kunal", "Lakshya",
]
_FIRST_NAMES_FEMALE = [
    "Aanya", "Aditi", "Ananya", "Diya", "Ishita", "Kavya", "Mahima",
    "Nidhi", "Pooja", "Priya", "Ritika", "Shreya", "Simran", "Tanvi",
    "Akanksha", "Bhavna", "Charu", "Deepika", "Garima", "Heena",
]
_LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Singh", "Kumar", "Gupta", "Agarwal",
    "Joshi", "Mishra", "Yadav", "Mehta", "Shah", "Nair", "Reddy",
    "Iyer", "Pillai", "Menon", "Krishnan", "Bose", "Chatterjee",
]
_SCHOOLS = [
    "Delhi Public School", "Kendriya Vidyalaya", "DAV Public School",
    "Army Public School", "St. Xavier's School", "Bal Bharati School",
    "Ryan International School", "Birla Vidya Mandir", "The Heritage School",
    "Modern School", "Springdales School", "Apeejay School",
]
_SUBJECTS_CLASS10 = {
    "English": (0, 100),
    "Hindi": (0, 100),
    "Mathematics": (0, 100),
    "Science": (0, 100),
    "Social Science": (0, 100),
}
_EXAM_CENTRES = [
    "Examination Centre No. 1011, New Delhi",
    "Examination Centre No. 2045, Mumbai",
    "Examination Centre No. 3072, Bengaluru",
    "Examination Centre No. 4018, Chennai",
    "Examination Centre No. 5033, Kolkata",
    "Examination Centre No. 6021, Hyderabad",
]


def _make_student(rng: random.Random, student_id: int) -> dict:
    """Generate a complete fake student record."""
    gender = rng.choice(["M", "F"])
    first = rng.choice(_FIRST_NAMES_MALE if gender == "M" else _FIRST_NAMES_FEMALE)
    last = rng.choice(_LAST_NAMES)
    name = f"{first} {last}"

    dob_day = rng.randint(1, 28)
    dob_month = rng.randint(1, 12)
    dob_year = rng.randint(2006, 2010)
    dob = f"{dob_day:02d}/{dob_month:02d}/{dob_year}"

    school = rng.choice(_SCHOOLS)
    aadhaar = _generate_aadhaar(rng)
    roll_no = _generate_roll_number(rng)
    adm_no = _generate_admission_number(rng)

    # Subject marks — realistic distribution with occasional 0s and 100s
    marks = {}
    for subj, (lo, hi) in _SUBJECTS_CLASS10.items():
        if rng.random() < 0.02:
            m = 0
        elif rng.random() < 0.03:
            m = 100
        else:
            m = rng.randint(35, 99)
        marks[subj] = m

    total = sum(marks.values())
    percentage = round(total / (len(marks) * 100) * 100, 1)

    if percentage >= 90:
        grade = "A1"
    elif percentage >= 80:
        grade = "A2"
    elif percentage >= 70:
        grade = "B1"
    elif percentage >= 60:
        grade = "B2"
    elif percentage >= 50:
        grade = "C1"
    elif percentage >= 40:
        grade = "C2"
    else:
        grade = "D"

    return {
        "student_id": student_id,
        "name": name,
        "gender": gender,
        "dob": dob,
        "school": school,
        "aadhaar": aadhaar,
        "roll_number": roll_no,
        "admission_number": adm_no,
        "marks": marks,
        "total_marks": total,
        "percentage": percentage,
        "grade": grade,
        "exam_centre": rng.choice(_EXAM_CENTRES),
        "exam_year": "2025",
    }


# ---------------------------------------------------------------------------
# Document text builders
# ---------------------------------------------------------------------------

def _admit_card_text(s: dict) -> tuple[str, dict]:
    text = f"""CENTRAL BOARD OF SECONDARY EDUCATION
ADMIT CARD — CLASS X EXAMINATION {s['exam_year']}

Student Name   : {s['name']}
Roll Number    : {s['roll_number']}
Date of Birth  : {s['dob']}
School         : {s['school']}
Exam Centre    : {s['exam_centre']}
Father's Name  : (as per school records)
Aadhaar No.    : {s['aadhaar']}

IMPORTANT: Candidate must carry this admit card and a valid photo ID to the
examination centre on all days of the examination.
"""
    fields = {
        "student_name": s["name"],
        "roll_number": s["roll_number"],
        "dob": s["dob"],
        "school_name": s["school"],
        "exam_name": "CBSE Class X Examination",
        "exam_date": "March 2025",
        "exam_centre": s["exam_centre"],
    }
    return text, fields


def _roll_number_sheet_text(students: list[dict], class_section: str) -> tuple[str, dict]:
    header = f"""CLASS ROLL NUMBER SHEET
Class: {class_section}   Academic Year: 2024-25
School: {students[0]['school']}

S.No.  Admission No.   Student Name          Roll Number
----   ------------    ------------          -----------
"""
    rows = ""
    for i, s in enumerate(students[:10], 1):
        rows += f"{i:3d}.   {s['admission_number']:<14}  {s['name']:<22} {s['roll_number']}\n"

    text = header + rows
    # Ground truth: first student on the sheet
    fields = {
        "student_name": students[0]["name"],
        "roll_number": students[0]["roll_number"],
        "class_section": class_section,
        "school_name": students[0]["school"],
    }
    return text, fields


def _answer_booklet_text(s: dict) -> tuple[str, dict]:
    text = f"""ANSWER BOOKLET
Subject: {list(s['marks'].keys())[0]}
Class X — Board Examination {s['exam_year']}

Student Name   : {s['name']}
Roll Number    : {s['roll_number']}
Set Number     : A
Invigilator    : ________________________

Examiner's Use Only
Total Marks Awarded: ___________  /  80
Signature: ________________________
"""
    fields = {
        "student_name": s["name"],
        "roll_number": s["roll_number"],
        "class_grade": "Class X",
        "exam_name": f"CBSE Class X — {list(s['marks'].keys())[0]}",
    }
    return text, fields


def _cbse_marksheet_text(s: dict) -> tuple[str, dict]:
    marks_lines = "\n".join(
        f"  {subj:<22} {marks:>3} / 100"
        for subj, marks in s["marks"].items()
    )
    text = f"""CENTRAL BOARD OF SECONDARY EDUCATION
STATEMENT OF MARKS — CLASS X ({s['exam_year']})

Student Name   : {s['name']}
Roll Number    : {s['roll_number']}
Date of Birth  : {s['dob']}
School         : {s['school']}

MARKS OBTAINED:
{marks_lines}
                         ──────────────
Total                  : {s['total_marks']} / 500
Percentage             : {s['percentage']}%
Grade                  : {s['grade']}
Result                 : {'PASS' if s['percentage'] >= 33 else 'FAIL'}
"""
    fields = {
        "student_name": s["name"],
        "roll_number": s["roll_number"],
        "dob": s["dob"],
        "school_name": s["school"],
        "class_grade": "Class X",
        "exam_year": s["exam_year"],
        "english_marks": str(s["marks"]["English"]),
        "hindi_marks": str(s["marks"]["Hindi"]),
        "mathematics_marks": str(s["marks"]["Mathematics"]),
        "science_marks": str(s["marks"]["Science"]),
        "social_science_marks": str(s["marks"]["Social Science"]),
        "total_marks": str(s["total_marks"]),
        "percentage": str(s["percentage"]),
        "grade": s["grade"],
        "result": "PASS" if s["percentage"] >= 33 else "FAIL",
    }
    return text, fields


def _school_id_text(s: dict) -> tuple[str, dict]:
    text = f"""{s['school'].upper()}
STUDENT IDENTITY CARD
Academic Year: 2024-25

Name    : {s['name']}
Class   : X-A
Roll No.: {s['roll_number']}
Adm No. : {s['admission_number']}
DOB     : {s['dob']}

[Photo]

If found, please return to school office.
"""
    fields = {
        "student_name": s["name"],
        "roll_number": s["roll_number"],
        "admission_number": s["admission_number"],
        "dob": s["dob"],
        "school_name": s["school"],
        "class_grade": "Class X",
    }
    return text, fields


_DOC_BUILDERS = {
    "admit_card": _admit_card_text,
    "answer_booklet": _answer_booklet_text,
    "cbse_marksheet": _cbse_marksheet_text,
    "school_id_card": _school_id_text,
}


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

def _render_document(
    text: str,
    doc_type: str,
    variation: str,
    rng: random.Random,
    font_dir: Path | None,
) -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageFont
    except ImportError:
        logger.error("Pillow is required: pip install pillow")
        sys.exit(1)

    import io as _io

    width, height = 794, 1123  # A4 at 96 dpi
    bg_color = (250, 250, 255) if doc_type == "cbse_marksheet" else (255, 255, 255)
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Border for official documents
    draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(0, 0, 128), width=2)

    font_size = 16
    try:
        if font_dir:
            font_path = next(font_dir.glob("Hind*.ttf"), None) or next(font_dir.glob("*.ttf"), None)
            if font_path and font_path.exists():
                from PIL import ImageFont as _IF
                font = _IF.truetype(str(font_path), font_size)
                title_font = _IF.truetype(str(font_path), font_size + 4)
            else:
                raise OSError()
        else:
            raise OSError()
    except (OSError, AttributeError):
        from PIL import ImageFont as _IF
        font = _IF.load_default()
        title_font = font
        font_size = 11

    x, y = 40, 50
    line_height = font_size + 10
    for line in text.split("\n"):
        if y + line_height > height - 40:
            break
        f = title_font if y < 90 else font
        draw.text((x, y), line, fill=(10, 10, 10), font=f)
        y += line_height

    # Apply variation
    if variation == "photo":
        angle = rng.uniform(-3, 3)
        img = img.rotate(angle, fillcolor=bg_color, expand=False)
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.2)))
        m = rng.randint(10, 30)
        img = img.crop((m, m, width - m, height - m)).resize((width, height), Image.LANCZOS)

    elif variation == "lowlight":
        from PIL import ImageEnhance
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.45, 0.65))
        img = ImageEnhance.Contrast(img).enhance(0.8)

    elif variation == "glare":
        overlay = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw2 = ImageDraw.Draw(overlay)
        gx = rng.randint(width // 4, width * 3 // 4)
        gy = rng.randint(height // 4, height * 3 // 4)
        r = rng.randint(80, 180)
        draw2.ellipse([(gx - r, gy - r), (gx + r, gy + r)], fill=(255, 255, 240))
        img = Image.blend(img, overlay, alpha=rng.uniform(0.25, 0.45))

    else:  # clean — minor blur only
        img = img.filter(ImageFilter.GaussianBlur(radius=0.15))

    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

VARIATIONS = ["clean", "photo", "lowlight", "glare"]
DOC_TYPES = ["admit_card", "roll_number_sheet", "answer_booklet", "cbse_marksheet", "school_id_card"]


def generate(output_dir: Path, count: int, seed: int, font_dir: Path | None) -> None:
    rng = random.Random(seed)
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    n_students = max(1, count // (len(VARIATIONS) * len(DOC_TYPES)))
    students = [_make_student(rng, i) for i in range(max(n_students, 40))]

    generated = 0
    for student_idx, student in enumerate(students):
        if generated >= count:
            break

        # For roll_number_sheet, pick 10 consecutive students for context
        sheet_students = students[student_idx: student_idx + 10] or [student]
        class_section = f"X-{rng.choice('ABCDE')}"

        for doc_type in DOC_TYPES:
            if generated >= count:
                break
            for variation in VARIATIONS:
                if generated >= count:
                    break

                if doc_type == "roll_number_sheet":
                    text, fields = _roll_number_sheet_text(sheet_students, class_section)
                else:
                    text, fields = _DOC_BUILDERS[doc_type](student)

                stem = f"stu_{student_idx:03d}_{doc_type}_{variation}"
                png_bytes = _render_document(text, doc_type, variation, rng, font_dir)
                (eval_dir / f"{stem}.png").write_bytes(png_bytes)
                (eval_dir / f"{stem}.json").write_text(
                    json.dumps(
                        {
                            "doc_type": doc_type,
                            "variation": variation,
                            "student_id": student_idx,
                            "fields": fields,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                generated += 1
                if generated % 100 == 0:
                    logger.info("Generated %d / %d documents", generated, count)

    logger.info("Done. %d documents in %s", generated, eval_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic student document dataset for teacher grading tests"
    )
    parser.add_argument(
        "--output",
        default="benchmarks/datasets/synthetic_student_docs",
        help="Output directory",
    )
    parser.add_argument("--count", type=int, default=800, help="Total documents to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--font-dir",
        default=os.environ.get("DA_FONT_DIR"),
        help="Directory containing TTF fonts",
    )
    args = parser.parse_args()

    font_dir = Path(args.font_dir) if args.font_dir else None
    generate(Path(args.output), args.count, args.seed, font_dir)


if __name__ == "__main__":
    main()
