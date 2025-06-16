#!/usr/bin/env python3
from PIL import Image
import pytesseract
import cv2
import pandas as pd
import numpy as np
import shutil

# ─── Configuration ─────────────────────────────────────────────────────────────
INPUT_IMAGE    = "rooftop_schedule.png"
FLAT_IMAGE     = "flat_schedule.png"
DEBUG_BW       = "debug_bw.png"
DEBUG_CLEAN    = "debug_clean.png"
OUTPUT_CSV     = "rooftop_schedule.csv"
UPSCALE_FACTOR = 2

# ─── Setup ────────────────────────────────────────────────────────────────────
def setup_tesseract():
    """Ensure Tesseract is available and configure pytesseract."""
    tess_cmd = shutil.which("tesseract")
    if not tess_cmd:
        raise FileNotFoundError(
            "Tesseract executable not found. Please install tesseract-ocr via 'sudo apt-get install tesseract-ocr'."
        )
    pytesseract.pytesseract.tesseract_cmd = tess_cmd

# ─── Image Preprocessing ─────────────────────────────────────────────────────
def flatten_image(input_path, flat_path):
    im = Image.open(input_path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255,255,255))
        bg.paste(im, mask=im.split()[3])
        bg.save(flat_path)
        return cv2.imread(flat_path)
    return cv2.imread(input_path)

def upscale_image(img, factor):
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

def threshold_image(gray):
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
    horiz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (img_shape[1]//30, 1)),
        iterations=2
    )
    vert  = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_shape[0]//30)),
        iterations=2
    )
    clean = cv2.bitwise_xor(bw, cv2.bitwise_or(horiz, vert))
    cv2.imwrite(DEBUG_CLEAN, clean)
    return clean

# ─── Cell Detection ──────────────────────────────────────────────────────────
def dilate_image(clean):
    return cv2.dilate(
        clean,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),
        iterations=1
    )

def find_cells(dilated):
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cells = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w > 30 and h > 15:
            cells.append((y, x, w, h))
    cells.sort(key=lambda b: (b[0], b[1]))
    return cells

# ─── OCR & Table Assembly ────────────────────────────────────────────────────
def ocr_cells(cells, gray):
    if not cells:
        raise RuntimeError("No cells detected. Check debug images.")
    table = []
    current_y, row = cells[0][0], []
    for y,x,w,h in cells:
        if abs(y - current_y) > h//2:
            table.append(row)
            row = []
            current_y = y
        cell_img = 255 - gray[y:y+h, x:x+w]
        text = pytesseract.image_to_string(cell_img, config="--psm 7").strip().replace("\n"," ")
        row.append(text)
    table.append(row)
    return table

# ─── Normalize & Save ────────────────────────────────────────────────────────
def normalize_and_save(table, output_csv):
    max_cols = max(len(r) for r in table)
    header = table[0] + [f"col_{i}" for i in range(len(table[0]), max_cols)]
    rows = [r + [""]*(max_cols - len(r)) for r in table[1:]]
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {output_csv} with {max_cols} columns.")

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    setup_tesseract()
    img = flatten_image(INPUT_IMAGE, FLAT_IMAGE)
    img = upscale_image(img, UPSCALE_FACTOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = threshold_image(gray)
    clean = remove_grid_lines(bw, img.shape)
    dilated = dilate_image(clean)
    cells = find_cells(dilated)
    table = ocr_cells(cells, gray)
    normalize_and_save(table, OUTPUT_CSV)

if __name__ == "__main__":
    main()
