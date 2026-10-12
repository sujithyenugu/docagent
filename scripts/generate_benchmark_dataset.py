#!/usr/bin/env python3
"""Generate synthetic Indian document benchmark dataset.

Creates 20 documents (5 per category) across:
  1. GST Invoices          — structured tables, key-value pairs
  2. Handwritten ledgers   — semi-structured, mix of printed headers + handwriting
  3. Government forms      — ration card / municipal applications
  4. ID-style documents    — Aadhaar-format and PAN-format cards

Each document is rendered as a PNG with an accompanying JSON ground truth file
containing all extractable key-value fields.

Output layout
-------------
    benchmarks/dataset/
        invoices/
            invoice_001.png  invoice_001.json
            ...
        ledgers/
            ledger_001.png   ledger_001.json
            ...
        govt_forms/
            form_001.png     form_001.json
            ...
        id_docs/
            id_001.png       id_001.json

Usage
-----
    python scripts/generate_benchmark_dataset.py
    python scripts/generate_benchmark_dataset.py --out-dir benchmarks/dataset --seed 42
"""

import argparse
import json
import math
import random
import textwrap
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DOCS_PER_CATEGORY = 5
_IMG_W, _IMG_H = 794, 1123  # A4 at 96 dpi

_COMPANY_NAMES = [
    "Sharma Trading Co.", "Mehta Enterprises Pvt Ltd", "Patel Goods & Services",
    "Kumar Brothers Ltd", "Singh Auto Parts",
]
_CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
           "Jaipur", "Kolkata", "Ahmedabad", "Pune", "Surat"]
_STATES = ["Maharashtra", "Delhi", "Karnataka", "Telangana", "Tamil Nadu",
           "Rajasthan", "West Bengal", "Gujarat", "Maharashtra", "Gujarat"]
_ITEMS = [
    ("Office Chairs", 2499),
    ("Printer Paper (500 sheets)", 349),
    ("USB Cables", 199),
    ("Monitor Stand", 1299),
    ("Laptop Bag", 899),
    ("Stapler Set", 249),
    ("Whiteboard Markers", 179),
    ("External Hard Drive", 3499),
    ("Webcam HD", 1799),
    ("Desk Lamp", 599),
]
_NAMES = [
    "Rajesh Kumar", "Priya Sharma", "Amit Patel", "Sunita Devi", "Vikram Singh",
    "Anita Gupta", "Suresh Nair", "Kavita Rao", "Deepak Verma", "Meena Krishnan",
    "Ravi Joshi", "Pooja Mehta", "Arun Iyer", "Sita Ram", "Gopal Das",
    "Lakshmi Narayanan", "Rohit Bose", "Nisha Kapoor", "Mahesh Tiwari", "Sarla Bajaj",
]
_GOODS = [
    ("Rice (kg)", 42, "kg"),
    ("Wheat Flour (kg)", 35, "kg"),
    ("Sugar (kg)", 48, "kg"),
    ("Kerosene (L)", 85, "L"),
    ("Dal (kg)", 95, "kg"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _gst_number(rng: random.Random) -> str:
    state_code = rng.randint(1, 36)
    pan = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
    num = "".join(str(rng.randint(0, 9)) for _ in range(4))
    letter = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    suffix = str(rng.randint(1, 9))
    check = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return f"{state_code:02d}{pan}{num}{letter}{suffix}Z{check}"


def _pan(rng: random.Random) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(rng.choices(letters, k=5))
        + "".join(str(rng.randint(0, 9)) for _ in range(4))
        + rng.choice(letters)
    )


def _aadhaar(rng: random.Random) -> str:
    digits = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(11)]
    return f"{digits[0]}{digits[1]}{digits[2]}{digits[3]} " \
           f"{digits[4]}{digits[5]}{digits[6]}{digits[7]} " \
           f"{digits[8]}{digits[9]}{digits[10]}{digits[11]}"


def _date(rng: random.Random, year_range=(2022, 2025)) -> str:
    y = rng.randint(*year_range)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{d:02d}/{m:02d}/{y}"


def _font(size: int = 14):
    """Return a PIL font — falls back to default if TrueType not available."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except (IOError, OSError):
        return ImageFont.load_default()


def _font_bold(size: int = 14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (IOError, OSError):
        return ImageFont.load_default()


def _draw_ruled_bg(draw: "ImageDraw.Draw", w: int, h: int) -> None:
    """Draw faint horizontal ruled lines."""
    draw.rectangle([0, 0, w, h], fill=(255, 255, 245))
    for y in range(40, h, 24):
        draw.line([(20, y), (w - 20, y)], fill=(200, 210, 220), width=1)


# ---------------------------------------------------------------------------
# Category 1 — GST Invoices
# ---------------------------------------------------------------------------

def _make_invoice(idx: int, rng: random.Random) -> tuple[Image.Image, dict]:
    img = Image.new("RGB", (_IMG_W, _IMG_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    company = _COMPANY_NAMES[idx % len(_COMPANY_NAMES)]
    city_idx = rng.randint(0, len(_CITIES) - 1)
    city, state = _CITIES[city_idx], _STATES[city_idx]
    gst = _gst_number(rng)
    inv_no = f"INV-{2024 * 1000 + rng.randint(1, 999):07d}"
    inv_date = _date(rng)
    due_date = _date(rng, (2024, 2025))

    buyer_idx = rng.randint(0, len(_NAMES) - 1)
    buyer = _NAMES[buyer_idx]
    buyer_city_idx = rng.randint(0, len(_CITIES) - 1)

    # Pick 2-4 line items
    n_items = rng.randint(2, 4)
    items = rng.sample(_ITEMS, n_items)
    line_items = []
    for name, unit_price in items:
        qty = rng.randint(1, 10)
        total = qty * unit_price
        line_items.append({"description": name, "qty": qty, "unit_price": unit_price, "total": total})

    subtotal = sum(li["total"] for li in line_items)
    cgst = round(subtotal * 0.09, 2)
    sgst = round(subtotal * 0.09, 2)
    grand_total = subtotal + cgst + sgst

    # ---- render ----
    fb = _font_bold(18)
    f = _font(13)
    fs = _font(11)

    draw.rectangle([0, 0, _IMG_W, 60], fill=(30, 58, 95))
    draw.text((20, 15), company, font=_font_bold(20), fill=(255, 255, 255))
    draw.text((_IMG_W - 140, 15), "TAX INVOICE", font=_font_bold(14), fill=(255, 230, 100))

    y = 70
    draw.text((20, y), f"GST No: {gst}", font=fs, fill=(60, 60, 60))
    draw.text((20, y + 16), f"Address: {city}, {state}", font=fs, fill=(60, 60, 60))

    draw.text((_IMG_W - 280, y), f"Invoice No: {inv_no}", font=f, fill=(30, 30, 30))
    draw.text((_IMG_W - 280, y + 18), f"Date: {inv_date}", font=f, fill=(30, 30, 30))
    draw.text((_IMG_W - 280, y + 36), f"Due Date: {due_date}", font=f, fill=(30, 30, 30))

    y += 70
    draw.line([(20, y), (_IMG_W - 20, y)], fill=(200, 200, 200), width=1)
    y += 10
    draw.text((20, y), f"Bill To: {buyer}", font=fb, fill=(30, 30, 30))
    draw.text((20, y + 18), f"City: {_CITIES[buyer_city_idx]}", font=f, fill=(60, 60, 60))

    y += 50
    draw.rectangle([20, y, _IMG_W - 20, y + 24], fill=(220, 230, 245))
    col_x = [20, 260, 350, 460, 560]
    headers = ["Description", "Qty", "Unit Price", "Total", ""]
    for cx, h in zip(col_x, headers):
        draw.text((cx + 4, y + 4), h, font=_font_bold(12), fill=(30, 30, 30))

    y += 28
    for li in line_items:
        draw.text((col_x[0] + 4, y + 2), li["description"], font=f, fill=(40, 40, 40))
        draw.text((col_x[1] + 4, y + 2), str(li["qty"]), font=f, fill=(40, 40, 40))
        draw.text((col_x[2] + 4, y + 2), f"Rs.{li['unit_price']:,}", font=f, fill=(40, 40, 40))
        draw.text((col_x[3] + 4, y + 2), f"Rs.{li['total']:,}", font=f, fill=(40, 40, 40))
        y += 22
        draw.line([(20, y), (_IMG_W - 20, y)], fill=(230, 230, 230), width=1)

    y += 16
    for label, val in [("Subtotal", subtotal), ("CGST (9%)", cgst), ("SGST (9%)", sgst)]:
        draw.text((_IMG_W - 280, y), label, font=f, fill=(60, 60, 60))
        draw.text((_IMG_W - 110, y), f"Rs.{val:,.2f}", font=f, fill=(60, 60, 60))
        y += 20

    draw.rectangle([_IMG_W - 290, y, _IMG_W - 20, y + 26], fill=(30, 58, 95))
    draw.text((_IMG_W - 280, y + 4), "Grand Total", font=_font_bold(13), fill=(255, 255, 255))
    draw.text((_IMG_W - 150, y + 4), f"Rs.{grand_total:,.2f}", font=_font_bold(13), fill=(255, 230, 100))

    ground_truth = {
        "category": "invoice",
        "fields": {
            "company_name": company,
            "gst_number": gst,
            "city": city,
            "invoice_number": inv_no,
            "invoice_date": inv_date,
            "due_date": due_date,
            "buyer_name": buyer,
            "subtotal": f"Rs.{subtotal:,.2f}",
            "cgst": f"Rs.{cgst:,.2f}",
            "sgst": f"Rs.{sgst:,.2f}",
            "grand_total": f"Rs.{grand_total:,.2f}",
        },
        "line_items": line_items,
    }
    return img, ground_truth


# ---------------------------------------------------------------------------
# Category 2 — Handwritten ledgers
# ---------------------------------------------------------------------------

def _make_ledger(idx: int, rng: random.Random) -> tuple[Image.Image, dict]:
    img = Image.new("RGB", (_IMG_W, _IMG_H), (255, 255, 245))
    draw = ImageDraw.Draw(img)
    _draw_ruled_bg(draw, _IMG_W, _IMG_H)

    shop = f"{_NAMES[idx % len(_NAMES)].split()[0]}'s General Store"
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    month = month_names[rng.randint(0, 11)]
    year = rng.randint(2022, 2025)

    fb = _font_bold(16)
    f = _font(13)
    fh = _font(11)

    draw.text((200, 15), shop, font=fb, fill=(20, 20, 80))
    draw.text((250, 38), f"Daily Ledger — {month} {year}", font=f, fill=(60, 60, 60))
    draw.line([(20, 60), (_IMG_W - 20, 60)], fill=(100, 100, 100), width=2)

    # Column headers
    y = 70
    col_x = [20, 90, 280, 440, 580, 680]
    for cx, h in zip(col_x, ["Date", "Description", "Debit (Rs.)", "Credit (Rs.)", "Balance"]):
        draw.text((cx, y), h, font=_font_bold(12), fill=(30, 30, 80))
    draw.line([(20, y + 18), (_IMG_W - 20, y + 18)], fill=(100, 100, 200), width=1)

    y += 26
    entries = []
    balance = rng.randint(5000, 15000)
    for day in range(1, rng.randint(8, 16)):
        date_str = f"{day:02d}/{month_names.index(month)+1:02d}/{year}"
        is_credit = rng.random() > 0.4
        amount = rng.randint(200, 5000)
        desc_choices = [
            "Sale of goods", "Purchase payment", "Utility bill", "Salary paid",
            "Goods received", "Customer payment", "Rent paid", "Misc expenses",
        ]
        desc = rng.choice(desc_choices)
        if is_credit:
            balance += amount
            entries.append({"date": date_str, "description": desc, "debit": "", "credit": str(amount), "balance": str(balance)})
            draw.text((col_x[0], y), date_str, font=fh, fill=(40, 40, 40))
            draw.text((col_x[1], y), desc, font=fh, fill=(40, 40, 40))
            draw.text((col_x[2], y), "", font=fh, fill=(40, 40, 40))
            draw.text((col_x[3], y), str(amount), font=fh, fill=(20, 120, 20))
            draw.text((col_x[4], y), str(balance), font=fh, fill=(40, 40, 40))
        else:
            balance -= amount
            entries.append({"date": date_str, "description": desc, "debit": str(amount), "credit": "", "balance": str(balance)})
            draw.text((col_x[0], y), date_str, font=fh, fill=(40, 40, 40))
            draw.text((col_x[1], y), desc, font=fh, fill=(40, 40, 40))
            draw.text((col_x[2], y), str(amount), font=fh, fill=(180, 20, 20))
            draw.text((col_x[3], y), "", font=fh, fill=(40, 40, 40))
            draw.text((col_x[4], y), str(balance), font=fh, fill=(40, 40, 40))
        y += 22

    draw.line([(20, y + 4), (_IMG_W - 20, y + 4)], fill=(80, 80, 80), width=2)
    draw.text((20, y + 10), f"Closing Balance: Rs. {balance}", font=fb, fill=(20, 20, 80))

    # Simulate handwriting-like noise (slight rotation + blur)
    img = img.rotate(rng.uniform(-1.0, 1.0), expand=False, fillcolor=(255, 255, 245))
    img = img.filter(ImageFilter.GaussianBlur(0.4))

    ground_truth = {
        "category": "ledger",
        "fields": {
            "shop_name": shop,
            "month": month,
            "year": str(year),
            "closing_balance": str(balance),
            "num_entries": str(len(entries)),
        },
        "entries": entries,
    }
    return img, ground_truth


# ---------------------------------------------------------------------------
# Category 3 — Government forms
# ---------------------------------------------------------------------------

_FORM_TITLES = [
    "Ration Card Application Form",
    "Municipal Birth Certificate Application",
    "Caste Certificate Application",
    "Domicile Certificate Application",
    "Income Certificate Application",
]
_PURPOSES = [
    "To avail subsidised food grains under NFSA",
    "For official record and proof of birth",
    "For educational and employment reservations",
    "To prove permanent residence in the state",
    "To avail income-linked government schemes",
]


def _make_govt_form(idx: int, rng: random.Random) -> tuple[Image.Image, dict]:
    img = Image.new("RGB", (_IMG_W, _IMG_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title = _FORM_TITLES[idx % len(_FORM_TITLES)]
    purpose = _PURPOSES[idx % len(_PURPOSES)]
    state_idx = rng.randint(0, len(_STATES) - 1)
    state = _STATES[state_idx]

    applicant = _NAMES[rng.randint(0, len(_NAMES) - 1)]
    dob = _date(rng, (1960, 2005))
    gender = rng.choice(["Male", "Female", "Other"])
    city = _CITIES[state_idx]
    pin = f"{rng.randint(100000, 999999)}"
    district = rng.choice(["Central", "East", "West", "North", "South"]) + " " + city.split()[0]
    ward_no = f"Ward No. {rng.randint(1, 50)}"
    uid = _aadhaar(rng)
    phone = f"+91 {rng.randint(7000000000, 9999999999)}"
    app_date = _date(rng, (2023, 2025))

    draw.rectangle([0, 0, _IMG_W, 80], fill=(220, 235, 255))
    draw.text((20, 10), f"Government of {state}", font=_font_bold(16), fill=(20, 40, 100))
    draw.text((20, 35), title, font=_font_bold(18), fill=(20, 40, 100))
    draw.text((20, 60), f"Purpose: {purpose}", font=_font(11), fill=(60, 60, 120))

    f = _font(13)
    fb = _font_bold(13)
    y = 100

    def field_row(label: str, value: str) -> None:
        nonlocal y
        draw.text((20, y), label + ":", font=fb, fill=(40, 40, 40))
        draw.text((260, y), value, font=f, fill=(20, 20, 80))
        draw.line([(260, y + 16), (_IMG_W - 40, y + 16)], fill=(180, 180, 180), width=1)
        y += 26

    field_row("Applicant Name", applicant)
    field_row("Date of Birth", dob)
    field_row("Gender", gender)
    field_row("State", state)
    field_row("District", district)
    field_row("City / Town", city)
    field_row("Ward / Village", ward_no)
    field_row("PIN Code", pin)
    field_row("Mobile Number", phone)
    field_row("Aadhaar Number", uid)
    field_row("Application Date", app_date)

    y += 20
    draw.text((20, y), "Declaration: I hereby declare that all information furnished above is true.", font=_font(11), fill=(80, 80, 80))
    y += 40
    draw.text((20, y), "Signature: ________________", font=f, fill=(40, 40, 40))
    draw.text((_IMG_W - 240, y), f"Date: {app_date}", font=f, fill=(40, 40, 40))

    ground_truth = {
        "category": "govt_form",
        "fields": {
            "form_title": title,
            "state": state,
            "applicant_name": applicant,
            "date_of_birth": dob,
            "gender": gender,
            "district": district,
            "city": city,
            "ward": ward_no,
            "pin_code": pin,
            "mobile_number": phone,
            "aadhaar_number": uid,
            "application_date": app_date,
        },
    }
    return img, ground_truth


# ---------------------------------------------------------------------------
# Category 4 — ID-style documents
# ---------------------------------------------------------------------------

def _make_id_doc(idx: int, rng: random.Random) -> tuple[Image.Image, dict]:
    is_pan = idx % 2 == 0
    img_w, img_h = 640, 400
    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    name = _NAMES[rng.randint(0, len(_NAMES) - 1)]
    dob = _date(rng, (1960, 2005))
    father_name = _NAMES[rng.randint(0, len(_NAMES) - 1)]

    if is_pan:
        pan_no = _pan(rng)
        draw.rectangle([0, 0, img_w, img_h], fill=(255, 250, 220))
        draw.rectangle([0, 0, img_w, 55], fill=(0, 100, 0))
        draw.text((20, 10), "INCOME TAX DEPARTMENT — GOVT OF INDIA", font=_font_bold(13), fill=(255, 255, 255))
        draw.text((20, 30), "PERMANENT ACCOUNT NUMBER CARD", font=_font(11), fill=(200, 255, 200))

        draw.text((20, 75), "Name:", font=_font_bold(12), fill=(40, 40, 40))
        draw.text((120, 75), name.upper(), font=_font(14), fill=(20, 20, 80))

        draw.text((20, 105), "Father's Name:", font=_font_bold(12), fill=(40, 40, 40))
        draw.text((180, 105), father_name.upper(), font=_font(14), fill=(20, 20, 80))

        draw.text((20, 135), "Date of Birth:", font=_font_bold(12), fill=(40, 40, 40))
        draw.text((180, 135), dob, font=_font(14), fill=(20, 20, 80))

        draw.text((20, 175), pan_no, font=_font_bold(22), fill=(10, 10, 10), spacing=6)
        draw.text((20, 210), "Signature", font=_font(11), fill=(100, 100, 100))
        draw.line([(20, 230), (200, 230)], fill=(80, 80, 80), width=1)

        draw.rectangle([img_w - 130, 70, img_w - 20, 200], outline=(100, 100, 100), width=2)
        draw.text((img_w - 100, 120), "PHOTO", font=_font(14), fill=(180, 180, 180))

        ground_truth = {
            "category": "id_doc",
            "doc_type": "PAN",
            "fields": {
                "name": name.upper(),
                "fathers_name": father_name.upper(),
                "date_of_birth": dob,
                "pan_number": pan_no,
            },
        }
    else:
        uid = _aadhaar(rng)
        gender = rng.choice(["MALE", "FEMALE"])
        city_idx = rng.randint(0, len(_CITIES) - 1)
        address = f"{rng.randint(1, 999)}, {rng.choice(['MG Road', 'Nehru Street', 'Gandhi Nagar', 'Subhash Chowk'])}, {_CITIES[city_idx]}"

        draw.rectangle([0, 0, img_w, img_h], fill=(248, 248, 255))
        draw.rectangle([0, 0, img_w, 55], fill=(255, 100, 0))
        draw.text((20, 10), "GOVERNMENT OF INDIA", font=_font_bold(15), fill=(255, 255, 255))
        draw.text((20, 32), "UNIQUE IDENTIFICATION AUTHORITY OF INDIA", font=_font(10), fill=(255, 220, 180))

        draw.text((20, 70), name, font=_font_bold(18), fill=(20, 20, 80))
        draw.text((20, 98), f"DOB: {dob}  |  {gender}", font=_font(13), fill=(60, 60, 60))
        draw.text((20, 124), address, font=_font(12), fill=(60, 60, 60))

        draw.text((20, 160), uid, font=_font_bold(24), fill=(10, 10, 10))
        draw.text((20, 200), "VID: " + " ".join(str(rng.randint(1000, 9999)) for _ in range(4)), font=_font(11), fill=(80, 80, 80))

        draw.rectangle([img_w - 130, 60, img_w - 20, 180], outline=(100, 100, 100), width=2)
        draw.text((img_w - 100, 110), "PHOTO", font=_font(14), fill=(180, 180, 180))

        ground_truth = {
            "category": "id_doc",
            "doc_type": "Aadhaar",
            "fields": {
                "name": name,
                "date_of_birth": dob,
                "gender": gender,
                "address": address,
                "aadhaar_number": uid,
            },
        }

    # Slight perspective distortion simulation
    img = img.rotate(rng.uniform(-0.5, 0.5), expand=False, fillcolor=(240, 240, 240))

    # Resize to A4 proportions for consistent benchmark
    img_full = Image.new("RGB", (_IMG_W, _IMG_H), (245, 245, 245))
    paste_x = (_IMG_W - img_w) // 2
    paste_y = 60
    img_full.paste(img, (paste_x, paste_y))

    return img_full, ground_truth


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

_GENERATORS = {
    "invoices": _make_invoice,
    "ledgers": _make_ledger,
    "govt_forms": _make_govt_form,
    "id_docs": _make_id_doc,
}


def generate(out_dir: Path, seed: int = 42) -> None:
    if not PIL_AVAILABLE:
        print("ERROR: Pillow not installed. Run: pip install Pillow")
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for category, fn in _GENERATORS.items():
        cat_dir = out_dir / category
        cat_dir.mkdir(exist_ok=True)
        for i in range(1, _DOCS_PER_CATEGORY + 1):
            rng = _rng(seed + hash(category) + i)
            img, gt = fn(i - 1, rng)
            stem = f"{category[:-1]}_{i:03d}"  # e.g. invoice_001
            img_path = cat_dir / f"{stem}.png"
            gt_path = cat_dir / f"{stem}.json"
            img.save(img_path)
            gt_path.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  wrote {img_path.relative_to(out_dir.parent)}")
            total += 1
    print(f"\nGenerated {total} documents ({_DOCS_PER_CATEGORY} per category) in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark dataset of Indian documents")
    parser.add_argument(
        "--out-dir",
        default="benchmarks/dataset",
        help="Output directory (default: benchmarks/dataset)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    generate(Path(args.out_dir), seed=args.seed)


if __name__ == "__main__":
    main()
