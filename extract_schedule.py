#!/usr/bin/env python3
from PIL import Image
import pytesseract
import cv2
import pandas as pd
import numpy as np
import os
import shutil

# ─── Configuration ─────────────────────────────────────────────────────────────
# Default Windows Tesseract path; system install will be preferred if available
default_tesseract_path = r"C:\Users\dylan.thach\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
INPUT_IMAGE    = "rooftop_schedule.png"
FLAT_IMAGE     = "flat_schedule.png"
DEBUG_BW       = "debug_bw.png"
DEBUG_CLEAN    = "debug_clean.png"
OUTPUT_CSV     = "rooftop_schedule.csv"
UPSCALE_FACTOR = 2

# ─── Setup ────────────────────────────────────────────────────────────────────
def setup_tesseract():
    """Locate the tesseract executable: prefer system install over Windows fallback."""
    # Prefer system tesseract if on PATH
    tess_cmd = shutil.which("tesseract")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
    # Otherwise use Windows-default path if it exists
    elif os.path.exists(default_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = default_tesseract_path
    else:
        raise FileNotFoundError(
            "Tesseract executable not found. Install it or update default_tesseract_path."
        )

# ─── Image Preprocessing ─────────────────────────────────────────────────────
def flatten_image(input_path, flat_path):
    """Remove any alpha channel by compositing onto a white background."""
    im = Image.open(input_path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255,255,255))
        bg.paste(im, mask=im.split()[3])
        bg.save(flat_path)
        return cv2.imread(flat_path)
    return cv2.imread(input_path)


def upscale_image(img, factor):
    """Double the resolution for sharper OCR."""
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def threshold_image(gray):
    """Adaptive threshold for high-contrast binary image."""
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )
    cv2.imwrite(DEBUG_BW, bw)
    return bw

# ─── Grid-Line Removal ───────────────────────────────────────────────────────
def remove_grid_lines(bw, img_shape):
    """Strip away table lines via morphological opening."""
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_shape[1]//30, 1))
    vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_shape[0]//30))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    vert  = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel,  iterations=2)
    clean = cv2.bitwise_xor(bw, cv2.bitwise_or(horiz, vert))
    cv2.imwrite(DEBUG_CLEAN, clean)
    return clean

# ─── Cell Detection ──────────────────────────────────────────────────────────
def dilate_image(clean):
    """Merge text into connected blobs for contour detection."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    return cv2.dilate(clean, kernel, iterations=1)


def find_cells(dilated):
    """Detect and sort bounding boxes for each table cell."""
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w > 30 and h > 15:
            cells.append((y, x, w, h))
    cells.sort(key=lambda b: (b[0], b[1]))
    return cells

# ─── OCR & Table Assembly ────────────────────────────────────────────────────
def ocr_cells(cells, gray):
    """Perform OCR on each cell and build a 2D list of text rows."""
    if not cells:
        raise RuntimeError("No cells detected — check debug images.")
    table, current_y, row = [], cells[0][0], []
    for y,x,w,h in cells:
        if abs(y - current_y) > h//2:
            table.append(row)
            row = []
            current_y = y
        cell = gray[y:y+h, x:x+w]
        cell = 255 - cell  # invert for OCR
        text = pytesseract.image_to_string(cell, config="--psm 7").strip().replace("\n"," ")
        row.append(text or "")
    table.append(row)
    return table

# ─── Table Normalization & Save ──────────────────────────────────────────────
def normalize_and_save(table, output_csv):
    """Pad rows to equal length and save as CSV."""
    col_count = max(len(r) for r in table)
    header = table[0] + [f"col_{i}" for i in range(len(table[0]), col_count)]
    data = [r + [""]*(col_count - len(r)) for r in table[1:]]
    pd.DataFrame(data, columns=header).to_csv(output_csv, index=False)
    print(f"✅ Saved {output_csv} with {col_count} columns.")

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    setup_tesseract()
    img = flatten_image(INPUT_IMAGE, FLAT_IMAGE)
    img = upscale_image(img, UPSCALE_FACTOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = threshold_image(gray)
    clean = remove_grid_lines(bw, img.shape)
    dil = dilate_image(clean)
    cells = find_cells(dil)
    table = ocr_cells(cells, gray)
    normalize_and_save(table, OUTPUT_CSV)

if __name__ == "__main__":
    main()
