"""Generate the synthetic sample dataset (10 images + ground_truth.json).

Every line of text that is drawn is also recorded, so the exact ground truth
is known for the evaluation. Real photographs can be added to data/samples/
as well; they are processed normally but only images listed in
ground_truth.json are scored by the evaluation script.

Usage:  python scripts/generate_samples.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples"
FONT_DIR = Path("C:/Windows/Fonts") if sys.platform == "win32" else Path("/usr/share/fonts/truetype")
RNG = np.random.default_rng(42)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for candidate in (FONT_DIR / name, Path(name)):
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


class Doc:
    """A canvas that remembers every text line drawn on it (the ground truth)."""

    def __init__(self, w: int, h: int, bg=(255, 255, 255)):
        self.img = Image.new("RGB", (w, h), bg)
        self.draw = ImageDraw.Draw(self.img)
        self.lines: list[str] = []

    def text(self, x, y, s, f, fill=(25, 25, 25), gt=True, center_w=None):
        if center_w is not None:
            x = x + (center_w - self.draw.textlength(s, font=f)) / 2
        self.draw.text((x, y), s, font=f, fill=fill)
        if gt:
            self.lines.append(s)

    def row(self, y, cells, f, fill=(25, 25, 25)):
        """Several columns on one line; ground truth joins them with spaces."""
        for x, s in cells:
            self.draw.text((x, y), s, font=f, fill=fill)
        self.lines.append(" ".join(s for _, s in cells))

    def para(self, x, y, lines, f, gap, fill=(25, 25, 25)):
        for line in lines:
            self.text(x, y, line, f, fill)
            y += gap
        return y

    @property
    def ground_truth(self) -> str:
        return "\n".join(self.lines)


# ------------------------------------------------------------ degradations
def add_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    arr += RNG.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_speckles(img: Image.Image, amount: float) -> Image.Image:
    arr = np.asarray(img).copy()
    mask = RNG.random(arr.shape[:2])
    arr[mask < amount / 2] = 40
    arr[mask > 1 - amount / 2] = 255
    return Image.fromarray(arr)


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def illumination(img: Image.Image, strength: float) -> Image.Image:
    """Uneven lighting: darken from one corner towards the other."""
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    grad = 1.0 - strength * ((xx / w) * 0.6 + (yy / h) * 0.4)
    arr = np.asarray(img).astype(np.float32) * grad[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ------------------------------------------------------------ the samples
def printed_document():
    d = Doc(1240, 1500)
    h1, body, bold = font("timesbd.ttf", 38), font("times.ttf", 27), font("timesbd.ttf", 27)
    d.text(100, 90, "Nimbus Office Supplies Pvt. Ltd.", h1)
    d.text(100, 145, "42 Lake View Road, Bengaluru 560001", body)
    d.text(100, 230, "Date: 14 March 2026", body)
    d.text(100, 280, "Subject: Non-Disclosure Agreement for Project Falcon", bold)
    d.text(100, 360, "Dear Ms. Priya Menon,", body)
    y = d.para(100, 420, [
        "This letter confirms that the attached non-disclosure agreement must be",
        "signed before any confidential material is shared with your team. Please",
        "read the terms and conditions carefully and return the signed copy by the",
        "due date of 21/03/2026. The signature must match the one on our records.",
        "For questions, write to legal@nimbus-supplies.example.com or call",
        "+91 98765 43210. Further details are available on our website at",
        "www.nimbus-supplies.example.com/legal.",
    ], body, 46)
    y = d.para(100, y + 30, ["This matter is urgent and is marked confidential.",
                             "Thank you for your cooperation."], body, 46)
    d.para(100, y + 60, ["Yours sincerely,", "Rahul Verma", "Legal Department"], body, 46)
    return d.img, d.ground_truth


def invoice():
    d = Doc(1240, 1450)
    big, f, b = font("arialbd.ttf", 44), font("arial.ttf", 25), font("arialbd.ttf", 25)
    d.draw.rectangle([0, 0, 1240, 130], fill=(232, 240, 250))
    d.text(80, 40, "TAX INVOICE", big, fill=(20, 50, 110))
    d.text(80, 170, "Nimbus Office Supplies Pvt. Ltd.", b)
    d.text(80, 210, "GSTIN: 29ABCDE1234F1Z5", f)
    d.row(270, [(80, "Invoice No: INV-2026-0147"), (780, "Date: 05/04/2026")], f)
    d.text(80, 315, "Bill To: Arjun Traders, 18 MG Road, Pune 411001", f)
    d.draw.rectangle([70, 390, 1170, 440], fill=(40, 70, 130))
    d.row(400, [(90, "Item"), (560, "Qty"), (700, "Rate"), (930, "Amount")], b, fill=(255, 255, 255))
    rows = [("A4 Paper Ream", "10", "Rs. 250.00", "Rs. 2,500.00"),
            ("Stapler Heavy Duty", "2", "Rs. 450.00", "Rs. 900.00"),
            ("Printer Toner", "1", "Rs. 3,200.00", "Rs. 3,200.00")]
    y = 465
    for item, qty, rate, amt in rows:
        d.row(y, [(90, item), (570, qty), (700, rate), (930, amt)], f)
        d.draw.line([70, y + 45, 1170, y + 45], fill=(200, 200, 200), width=1)
        y += 60
    d.row(y + 20, [(700, "Subtotal"), (930, "Rs. 6,600.00")], f)
    d.row(y + 65, [(700, "GST 18%"), (930, "Rs. 1,188.00")], f)
    d.row(y + 110, [(700, "Grand Total"), (930, "Rs. 7,788.00")], b)
    y += 210
    y = d.para(80, y, ["Payment by bank transfer to Account Number 50100234567891",
                       "Due Date: 20/04/2026"], f, 45)
    d.para(80, y + 40, ["Thank you for your business!",
                        "Queries: accounts@nimbus-supplies.example.com"], f, 45)
    return d.img, d.ground_truth


def receipt():
    d = Doc(600, 1060, bg=(246, 244, 238))
    m, mb = font("consola.ttf", 23), font("consolab.ttf", 26) if (FONT_DIR / "consolab.ttf").exists() else font("courbd.ttf", 26)
    d.text(0, 40, "FRESHMART SUPERSTORE", mb, center_w=600)
    d.text(0, 85, "Shop 12, Sector 15, Noida", m, center_w=600)
    d.text(0, 120, "Tel: (0120) 456 7890", m, center_w=600)
    d.text(40, 185, "Receipt No: RCPT-88213", m)
    d.text(40, 220, "Date: 18/04/2026 Time: 19:42", m)
    d.text(40, 260, "-" * 38, m, gt=False)
    y = 300
    for name, qty, price in [("Milk 1L", "x2", "Rs.120.00"), ("Bread Brown", "x1", "Rs.45.00"),
                             ("Eggs (12)", "x1", "Rs.84.00"), ("Rice 5kg", "x1", "Rs.399.00")]:
        d.row(y, [(40, name), (300, qty), (380, price)], m)
        y += 42
    d.row(y + 10, [(40, "Discount (10%)"), (380, "-Rs.64.80")], m)
    d.text(40, y + 50, "-" * 38, m, gt=False)
    d.row(y + 90, [(40, "TOTAL"), (380, "Rs.583.20")], mb)
    d.para(40, y + 160, ["Paid by UPI", "No refund without receipt", "Thank you, visit again!"], m, 40)
    img = add_speckles(d.img, 0.004)
    img = add_noise(img, 6)
    return img, d.ground_truth


def screenshot_email():
    d = Doc(1280, 800, bg=(255, 255, 255))
    ui, uib = font("segoeui.ttf", 22), font("segoeuib.ttf", 24)
    d.draw.rectangle([0, 0, 1280, 50], fill=(32, 33, 36))
    d.text(20, 12, "Inbox - Mail", font("segoeui.ttf", 20), fill=(235, 235, 235))
    d.draw.rectangle([0, 50, 230, 800], fill=(241, 243, 246))
    for i, item in enumerate(["Inbox", "Sent", "Drafts", "Spam"]):
        d.text(30, 90 + i * 50, item, ui, fill=(70, 70, 70))
    x = 270
    d.text(x, 80, "From: security@bank-alerts.example.com", ui, fill=(90, 90, 90))
    d.text(x, 125, "Subject: URGENT: Password Verification Required", uib)
    d.draw.line([x, 175, 1240, 175], fill=(220, 220, 220), width=2)
    y = d.para(x, 200, [
        "Dear Customer,",
        "We noticed a sign-in attempt from a new device on 02/05/2026.",
        "To keep your account active, complete the verification at",
        "https://secure-login.example.com/verify within 24 hours.",
        "Never share your password or OTP with anyone.",
        "Warning: unverified accounts will be suspended.",
        "Support: +91 91234 56789",
    ], ui, 42, fill=(40, 40, 40))
    d.draw.rounded_rectangle([x, y + 25, x + 200, y + 80], radius=8, fill=(26, 115, 232))
    d.text(x + 38, y + 37, "Verify Now", uib, fill=(255, 255, 255))
    return d.img, d.ground_truth


def poster_sign():
    w, h = 1200, 900
    yy = np.linspace(0, 1, h)[:, None]
    grad = np.zeros((h, w, 3), np.uint8)
    grad[..., 0] = (200 + 40 * yy).astype(np.uint8)
    grad[..., 1] = (40 + 90 * yy).astype(np.uint8)
    grad[..., 2] = 40
    d = Doc(w, h)
    d.img = Image.fromarray(grad)
    d.draw = ImageDraw.Draw(d.img)
    d.text(0, 60, "MEGA SUMMER SALE", font("impact.ttf", 110), fill=(255, 255, 255), center_w=w)
    d.text(0, 210, "FLAT 50% OFF", font("impact.ttf", 130), fill=(255, 236, 80), center_w=w)
    d.draw.rounded_rectangle([330, 390, 870, 470], radius=14, fill=(30, 30, 30))
    d.text(0, 400, "LIMITED OFFER", font("arialbd.ttf", 52), fill=(255, 255, 255), center_w=w)
    f = font("arialbd.ttf", 36)
    d.text(0, 530, "On all electronics and home appliances", f, fill=(255, 255, 255), center_w=w)
    d.text(0, 600, "Valid till 31/05/2026", f, fill=(255, 255, 255), center_w=w)
    d.text(0, 690, "Visit www.megamart.example.in", f, fill=(255, 250, 200), center_w=w)
    d.text(0, 760, "Call +91 99887 76655", f, fill=(255, 250, 200), center_w=w)
    return d.img, d.ground_truth


def skewed_scan():
    d = Doc(1240, 1000, bg=(250, 248, 242))
    h, f = font("georgia.ttf", 34), font("georgia.ttf", 26)
    d.text(100, 80, "INTERNAL MEMO", h)
    d.para(100, 160, ["Date: 10 June 2026", "To: All Department Heads",
                      "Subject: Budget Approval Deadline"], f, 44)
    d.para(100, 340, [
        "The revised budget has been approved by the finance",
        "committee. All department heads must submit their final",
        "expense reports before the deadline of 25/06/2026.",
        "Late submissions will not be processed for payment.",
        "Contact finance@nimbus-supplies.example.com for help.",
        "Reference: BILL-2026-0932",
    ], f, 46)
    img = d.img.rotate(6, resample=Image.BICUBIC, expand=True, fillcolor=(205, 205, 200))
    img = illumination(img, 0.15)
    img = add_noise(img, 5)
    return img, d.ground_truth


def low_quality_scan():
    d = Doc(820, 640)
    f, b = font("arial.ttf", 19), font("arialbd.ttf", 23)
    d.text(60, 40, "PAYMENT RECEIPT", b)
    d.para(60, 100, [
        "Receipt No: RCPT-55120",
        "Date: 22/06/2026",
        "Received from: Kavya Iyer",
        "Amount: Rs. 12,450.00",
        "Mode: Bank Transfer",
        "Account Number: 30214567890",
        "Purpose: Annual maintenance charges",
        "Authorised Signature",
        "This is a computer generated receipt.",
    ], f, 50)
    img = d.img.filter(ImageFilter.GaussianBlur(1.3))
    img = add_noise(img, 16)
    img = jpeg_roundtrip(img, 22)
    return img, d.ground_truth


def low_contrast_notice():
    d = Doc(1200, 760, bg=(196, 196, 190))
    tone = (150, 150, 146)
    d.text(0, 60, "NOTICE", font("arialbd.ttf", 64), fill=tone, center_w=1200)
    d.text(0, 170, "Warning: Restricted Area", font("arialbd.ttf", 40), fill=tone, center_w=1200)
    f = font("arial.ttf", 30)
    y = 270
    for line in ["Only approved personnel may enter beyond this point.",
                 "Visitors must register at the reception desk",
                 "and carry a valid ID card at all times.",
                 "Emergency contact: +91 90000 12345",
                 "Issued on 01/07/2026 by the Facilities Team"]:
        d.text(0, y, line, f, fill=tone, center_w=1200)
        y += 70
    img = illumination(d.img, 0.45)
    img = add_noise(img, 4)
    return img, d.ground_truth


def id_card():
    d = Doc(1000, 640, bg=(250, 251, 253))
    d.draw.rectangle([0, 0, 1000, 120], fill=(22, 72, 140))
    d.text(0, 18, "STUDENT IDENTITY CARD", font("arialbd.ttf", 40), fill=(255, 255, 255), center_w=1000)
    d.text(0, 72, "Nimbus Institute of Technology", font("arial.ttf", 26), fill=(220, 230, 245), center_w=1000)
    d.draw.rectangle([40, 160, 250, 420], outline=(120, 120, 120), width=3, fill=(225, 228, 232))
    d.text(100, 275, "PHOTO", font("arial.ttf", 24), fill=(130, 130, 130), gt=False)
    f = font("calibri.ttf", 28)
    d.para(290, 160, [
        "Name: Ananya Rao",
        "Date of Birth: 12/08/2004",
        "Roll No: NIT-2023-0456",
        "Course: B.Tech Computer Science",
        "Valid Till: 31/07/2027",
        "Email: ananya.rao@nimbus-institute.example.edu",
        "Address: 7 Park Street, Kolkata 700016",
    ], f, 44)
    d.text(40, 480, "Signature", font("calibri.ttf", 24), fill=(80, 80, 80))
    d.draw.line([40, 470, 250, 470], fill=(80, 80, 80), width=2)
    d.text(0, 570, "SPECIMEN - NOT A REAL DOCUMENT", font("arialbd.ttf", 26), fill=(200, 40, 40), center_w=1000)
    return d.img, d.ground_truth


def small_font_form():
    d = Doc(1100, 760)
    t, f = font("verdanab.ttf", 24) if (FONT_DIR / "verdanab.ttf").exists() else font("arialbd.ttf", 24), font("verdana.ttf", 16)
    d.text(60, 40, "Customer Feedback Form", t)
    d.row(110, [(60, "Customer Name: Rohan Gupta"), (600, "Phone: +91 98111 22334")], f)
    d.row(150, [(60, "Order Reference: INV-2026-0388"), (600, "Order Date: 2026-07-15")], f)
    d.draw.line([60, 195, 1040, 195], fill=(180, 180, 180), width=1)
    d.para(60, 220, [
        "Feedback: The parcel was addressed correctly but arrived late.",
        "Payments were invoiced twice; the totals did not match.",
        "Request: refund of the duplicate amount of $49.99 via card.",
        "Discounts promised in the email were not applied.",
    ], f, 38)
    d.draw.rectangle([60, 400, 78, 418], outline=(60, 60, 60), width=2)
    d.draw.line([63, 409, 69, 415], fill=(60, 60, 60), width=2)
    d.draw.line([69, 415, 76, 402], fill=(60, 60, 60), width=2)
    d.text(90, 400, "Approved for review", f)
    d.draw.rectangle([360, 400, 378, 418], outline=(60, 60, 60), width=2)
    d.text(390, 400, "Rejected", f)
    d.text(60, 470, "Signed: R. Gupta", f)
    return d.img, d.ground_truth


SAMPLES = [
    ("01_printed_document.png", "printed document", printed_document),
    ("02_invoice.png", "invoice", invoice),
    ("03_receipt.jpeg", "receipt", receipt),
    ("04_screenshot_email.png", "screenshot", screenshot_email),
    ("05_poster_sign.png", "poster / sign", poster_sign),
    ("06_skewed_scan.png", "skewed scan", skewed_scan),
    ("07_low_quality_scan.jpg", "low-quality scan", low_quality_scan),
    ("08_low_contrast_notice.tiff", "low-contrast photo", low_contrast_notice),
    ("09_id_card.bmp", "ID card", id_card),
    ("10_small_font_form.png", "small-font form", small_font_form),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    truth = {}
    for filename, category, fn in SAMPLES:
        img, text = fn()
        path = OUT / filename
        if path.suffix in {".jpg", ".jpeg"}:
            img.save(path, quality=90 if "receipt" in filename else 60)
        else:
            img.save(path)
        truth[filename] = {"category": category, "text": text}
        print(f"  wrote {path.relative_to(ROOT)}  ({img.size[0]}x{img.size[1]})")
    gt_path = ROOT / "data" / "ground_truth.json"
    gt_path.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {gt_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
