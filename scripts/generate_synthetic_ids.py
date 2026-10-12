#!/usr/bin/env python3
"""Generate synthetic Indian ID document images for Use Case B evaluation.

Documents generated (200 unique identities × 4 variations = 800 images):
  - Aadhaar card (front)
  - PAN card
  - Driving Licence
  - Voter ID (EPIC)
  - Passport (first page)

Variations per document:
  1. clean     — flat scan simulation
  2. photo     — perspective warp + blur (phone photo)
  3. lowlight  — darkened + noise
  4. glare     — lamination glare overlay

Privacy note
------------
All data is entirely fabricated using the ``faker`` library.  Aadhaar numbers
pass the Verhoeff checksum algorithm but are NOT registered with UIDAI.  PAN
numbers follow the ``AAAAA0000A`` pattern.  No real person's data is used.
No UIDAI / Government of India branding, logos, or official seals are reproduced.

Output structure
----------------
benchmarks/datasets/synthetic_ids/
  train/     (160 identities × 4 variations × 5 doc types = 3200 images)
  eval/      (40 identities × 4 variations × 5 doc types = 800 images)
  Each image: {doc_type}_{identity_id:04d}_{variation}.png
  Each GT:    {doc_type}_{identity_id:04d}_{variation}.json

Usage
-----
    python scripts/generate_synthetic_ids.py --output benchmarks/datasets/synthetic_ids \
        --identities 200 --seed 42

Requirements (install separately):
    pip install pillow faker
"""

import argparse
import io
import json
import logging
import random
import re
import string
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verhoeff algorithm for Aadhaar checksum
# ---------------------------------------------------------------------------

_VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9], [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6], [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8], [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2], [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4], [9,8,7,6,5,4,3,2,1,0],
]
_VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2], [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0], [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5], [7,0,4,6,9,1,3,2,5,8],
]
_VERHOEFF_INV = [0,4,3,2,1,9,8,7,6,5]


def _verhoeff_generate(number: str) -> str:
    """Append a Verhoeff check digit to *number* and return the full string."""
    c = 0
    digits = [int(d) for d in reversed(number + "0")]
    for i, d in enumerate(digits):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return number + str(_VERHOEFF_INV[c])


def _random_aadhaar(rng: random.Random) -> str:
    """Return a 12-digit Aadhaar number with valid Verhoeff checksum."""
    base = "".join(str(rng.randint(0, 9)) for _ in range(11))
    # First digit must not be 0 or 1
    while base[0] in "01":
        base = str(rng.randint(2, 9)) + base[1:]
    return _verhoeff_generate(base)


def _random_pan(rng: random.Random, name: str) -> str:
    """Return a PAN card number in AAAAA0000A format."""
    # First 3 chars: issuing AO series (arbitrary uppercase)
    part1 = "".join(rng.choices(string.ascii_uppercase, k=3))
    # 4th char: taxpayer type (P = individual)
    part2 = "P"
    # 5th char: first letter of surname
    surname_initial = name.split()[-1][0].upper() if name else "K"
    # 4 digits
    digits = "".join(str(rng.randint(0, 9)) for _ in range(4))
    # Last char: check letter (arbitrary)
    last = rng.choice(string.ascii_uppercase)
    return f"{part1}{part2}{surname_initial}{digits}{last}"


# ---------------------------------------------------------------------------
# Fake identity generator
# ---------------------------------------------------------------------------

def _make_identity(idx: int, rng: random.Random) -> dict:
    """Generate a fake but plausible Indian identity record."""
    try:
        from faker import Faker
        fake_hi = Faker("hi_IN")
        fake_en = Faker("en_IN")
    except ImportError:
        logger.error("faker is required: pip install faker")
        sys.exit(1)

    # Seed faker for reproducibility
    Faker.seed(rng.randint(0, 2**31))

    name_en = fake_en.name()
    name_hi = fake_hi.name()
    dob = fake_en.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d/%m/%Y")
    gender = rng.choice(["Male", "Female"])
    address_en = fake_en.address().replace("\n", ", ")
    aadhaar = _random_aadhaar(rng)
    pan = _random_pan(rng, name_en)

    # Driving licence number: SS-RRRR-YYYYNNNNNNN (state code + RTO + year + serial)
    state_codes = ["MH", "DL", "KA", "TN", "UP", "WB", "GJ", "RJ", "AP", "TS"]
    state = rng.choice(state_codes)
    rto = f"{rng.randint(1, 99):02d}"
    year = rng.randint(2000, 2023)
    serial = "".join(str(rng.randint(0, 9)) for _ in range(7))
    dl_number = f"{state}-{rto}-{year}-{serial}"

    # Voter ID (EPIC): 3 uppercase letters + 7 digits
    epic = (
        "".join(rng.choices(string.ascii_uppercase, k=3))
        + "".join(str(rng.randint(0, 9)) for _ in range(7))
    )

    # Passport: 1 letter + 7 digits
    passport_number = rng.choice(string.ascii_uppercase) + "".join(
        str(rng.randint(0, 9)) for _ in range(7)
    )

    vehicle_classes = ["LMV", "MCWG", "LMV-TR", "HMV", "MCWOG"]
    blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

    return {
        "identity_id": idx,
        "name_en": name_en,
        "name_hi": name_hi,
        "dob": dob,
        "gender": gender,
        "address": address_en,
        "aadhaar_number": aadhaar,
        "pan_number": pan,
        "dl_number": dl_number,
        "epic_number": epic,
        "passport_number": passport_number,
        "vehicle_class": rng.choice(vehicle_classes),
        "blood_group": rng.choice(blood_groups),
        "father_name": fake_en.name_male(),
    }


# ---------------------------------------------------------------------------
# Document renderers
# ---------------------------------------------------------------------------

def _base_card(width: int, height: int, bg_color: tuple) -> "Image":
    from PIL import Image
    return Image.new("RGB", (width, height), color=bg_color)


def _draw_field(draw, x: int, y: int, label: str, value: str, font, label_font) -> int:
    """Draw a label:value pair and return new y position."""
    draw.text((x, y), label + ":", fill=(80, 80, 80), font=label_font)
    draw.text((x + 130, y), value, fill=(10, 10, 10), font=font)
    return y + 22


def _render_aadhaar(identity: dict, rng: random.Random) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    w, h = 680, 430
    img = _base_card(w, h, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header bar (blue placeholder — no UIDAI branding)
    draw.rectangle([(0, 0), (w, 60)], fill=(0, 70, 160))
    draw.text((20, 15), "SAMPLE ID CARD — NOT OFFICIAL", fill=(255, 255, 255))
    draw.text((20, 35), "Unique Identification Sample Authority", fill=(200, 220, 255))

    try:
        font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    except Exception:
        font = label_font = None

    y = 80
    y = _draw_field(draw, 20, y, "Name", identity["name_en"], font, label_font)
    y = _draw_field(draw, 20, y, "DOB", identity["dob"], font, label_font)
    y = _draw_field(draw, 20, y, "Gender", identity["gender"], font, label_font)
    y += 5
    draw.text((20, y), "Address:", fill=(80, 80, 80), font=label_font)
    y += 22
    # Wrap address
    addr = identity["address"]
    for chunk in [addr[i:i+60] for i in range(0, min(len(addr), 180), 60)]:
        draw.text((20, y), chunk, fill=(10, 10, 10), font=font)
        y += 20
    y += 10
    # Aadhaar number (masked except last 4 for front display)
    aadhaar = identity["aadhaar_number"]
    display_aadhaar = f"XXXX XXXX {aadhaar[-4:]}"
    draw.text((20, y), f"Sample ID: {display_aadhaar}", fill=(30, 30, 30), font=font)
    draw.rectangle([(0, h - 20), (w, h)], fill=(0, 70, 160))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_pan(identity: dict, rng: random.Random) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    w, h = 640, 400
    img = _base_card(w, h, (252, 248, 230))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (w, 55)], fill=(180, 30, 30))
    draw.text((20, 15), "SAMPLE INCOME TAX CARD — NOT OFFICIAL", fill=(255, 255, 255))
    draw.text((20, 35), "Department of Revenue — Sample", fill=(255, 200, 200))

    try:
        font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    except Exception:
        font = label_font = None

    y = 75
    y = _draw_field(draw, 20, y, "Name", identity["name_en"], font, label_font)
    y = _draw_field(draw, 20, y, "Father's Name", identity["father_name"], font, label_font)
    y = _draw_field(draw, 20, y, "Date of Birth", identity["dob"], font, label_font)
    y = _draw_field(draw, 20, y, "PAN", identity["pan_number"], font, label_font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_dl(identity: dict, rng: random.Random) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    w, h = 680, 430
    img = _base_card(w, h, (240, 248, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (w, 55)], fill=(0, 100, 50))
    draw.text((20, 15), "SAMPLE DRIVING LICENCE — NOT OFFICIAL", fill=(255, 255, 255))

    try:
        font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    except Exception:
        font = label_font = None

    y = 75
    y = _draw_field(draw, 20, y, "Name", identity["name_en"], font, label_font)
    y = _draw_field(draw, 20, y, "DOB", identity["dob"], font, label_font)
    y = _draw_field(draw, 20, y, "DL Number", identity["dl_number"], font, label_font)
    y = _draw_field(draw, 20, y, "Vehicle Class", identity["vehicle_class"], font, label_font)
    y = _draw_field(draw, 20, y, "Blood Group", identity["blood_group"], font, label_font)
    y = _draw_field(draw, 20, y, "Address", identity["address"][:80], font, label_font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_voter_id(identity: dict, rng: random.Random) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    w, h = 640, 400
    img = _base_card(w, h, (255, 252, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (w, 55)], fill=(255, 153, 0))
    draw.text((20, 15), "SAMPLE VOTER ID CARD — NOT OFFICIAL", fill=(255, 255, 255))
    draw.text((20, 35), "Election Commission of India — Sample", fill=(255, 240, 200))

    try:
        font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    except Exception:
        font = label_font = None

    y = 75
    y = _draw_field(draw, 20, y, "Name", identity["name_en"], font, label_font)
    y = _draw_field(draw, 20, y, "Father/Husband", identity["father_name"], font, label_font)
    y = _draw_field(draw, 20, y, "DOB", identity["dob"], font, label_font)
    y = _draw_field(draw, 20, y, "EPIC No.", identity["epic_number"], font, label_font)
    y = _draw_field(draw, 20, y, "Address", identity["address"][:80], font, label_font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_passport(identity: dict, rng: random.Random) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    w, h = 680, 480
    img = _base_card(w, h, (240, 245, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (w, 55)], fill=(0, 40, 120))
    draw.text((20, 15), "SAMPLE TRAVEL DOCUMENT — NOT OFFICIAL", fill=(255, 255, 255))

    try:
        font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    except Exception:
        font = label_font = None

    y = 75
    y = _draw_field(draw, 20, y, "Surname", identity["name_en"].split()[-1], font, label_font)
    y = _draw_field(draw, 20, y, "Given Name", " ".join(identity["name_en"].split()[:-1]), font, label_font)
    y = _draw_field(draw, 20, y, "Nationality", "Indian", font, label_font)
    y = _draw_field(draw, 20, y, "Date of Birth", identity["dob"], font, label_font)
    y = _draw_field(draw, 20, y, "Passport No.", identity["passport_number"], font, label_font)
    y = _draw_field(draw, 20, y, "Gender", identity["gender"], font, label_font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_RENDERERS = {
    "aadhaar": _render_aadhaar,
    "pan": _render_pan,
    "dl": _render_dl,
    "voter_id": _render_voter_id,
    "passport": _render_passport,
}

_GT_FIELDS = {
    "aadhaar": ["name_en", "dob", "gender", "address", "aadhaar_number"],
    "pan": ["name_en", "father_name", "dob", "pan_number"],
    "dl": ["name_en", "dob", "dl_number", "vehicle_class", "blood_group"],
    "voter_id": ["name_en", "father_name", "dob", "epic_number"],
    "passport": ["name_en", "dob", "gender", "passport_number"],
}


# ---------------------------------------------------------------------------
# Variation applicators
# ---------------------------------------------------------------------------

def _apply_variation(img_bytes: bytes, variation: str, rng: random.Random) -> bytes:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np

    img = Image.open(io.BytesIO(img_bytes))

    if variation == "clean":
        pass  # no modification

    elif variation == "photo":
        angle = rng.uniform(-8, 8)
        img = img.rotate(angle, fillcolor=(200, 200, 200), expand=False)
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
        w, h = img.size
        margin = rng.randint(10, 40)
        img = img.crop((margin, margin, w - margin, h - margin))
        img = img.resize((w, h), Image.LANCZOS)

    elif variation == "lowlight":
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(rng.uniform(0.3, 0.55))
        arr = np.array(img).astype(float)
        noise = rng.uniform(8, 20)
        arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, noise, arr.shape)
        arr = arr.clip(0, 255).astype("uint8")
        img = Image.fromarray(arr)

    elif variation == "glare":
        from PIL import ImageDraw
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        w, h = img.size
        cx = rng.randint(w // 4, 3 * w // 4)
        cy = rng.randint(h // 4, 3 * h // 4)
        radius = rng.randint(80, 180)
        d.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill=(255, 255, 255, rng.randint(80, 140)),
        )
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(output_dir: Path, num_identities: int, seed: int) -> None:
    rng = random.Random(seed)

    train_count = int(num_identities * 0.8)
    variations = ["clean", "photo", "lowlight", "glare"]
    doc_types = list(_RENDERERS.keys())

    generated = 0
    total = num_identities * len(doc_types) * len(variations)

    for identity_id in range(num_identities):
        split = "train" if identity_id < train_count else "eval"
        out_dir = output_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        identity = _make_identity(identity_id, rng)

        for doc_type in doc_types:
            renderer = _RENDERERS[doc_type]
            base_bytes = renderer(identity, rng)
            gt_fields = {k: identity[k] for k in _GT_FIELDS[doc_type]}

            for variation in variations:
                stem = f"{doc_type}_{identity_id:04d}_{variation}"
                final_bytes = _apply_variation(base_bytes, variation, rng)
                (out_dir / f"{stem}.png").write_bytes(final_bytes)
                (out_dir / f"{stem}.json").write_text(
                    json.dumps(
                        {
                            "identity_id": identity_id,
                            "doc_type": doc_type,
                            "variation": variation,
                            "fields": gt_fields,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                generated += 1

        if (identity_id + 1) % 20 == 0:
            logger.info("Processed %d / %d identities (%d images)", identity_id + 1, num_identities, generated)

    logger.info("Done. %d images written to %s", generated, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Indian ID document dataset")
    parser.add_argument(
        "--output",
        default="benchmarks/datasets/synthetic_ids",
        help="Output directory",
    )
    parser.add_argument(
        "--identities",
        type=int,
        default=200,
        help="Number of unique identities to generate (default: 200)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    generate(Path(args.output), args.identities, args.seed)


if __name__ == "__main__":
    main()
