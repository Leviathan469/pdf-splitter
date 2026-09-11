#!/usr/bin/env python3
"""
PDF Stamp Splitter - GUI Version
User-friendly interface for splitting PDFs by stamp detection.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import subprocess
import sys
import threading
from pathlib import Path

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    print("ERROR: pypdf not installed. Run: pip install pypdf")
    sys.exit(1)


class PDFSplitterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Stamp Splitter")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Variables
        self.pdf_path = tk.StringVar()
        self.template_path = tk.StringVar(value=str(Path(__file__).parent / "stamp_template.png"))
        self.threshold = tk.DoubleVar(value=0.30)
        self.dpi = tk.IntVar(value=300)
        self.running = False
        
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="PDF Stamp Splitter", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15))
        
        # PDF File
        ttk.Label(main_frame, text="PDF File:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(main_frame, textvariable=self.pdf_path).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_pdf).grid(row=1, column=2)
        
        # Template File
        ttk.Label(main_frame, text="Template:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(main_frame, textvariable=self.template_path).grid(row=2, column=1, sticky="ew", padx=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_template).grid(row=2, column=2)
        
        # Threshold
        ttk.Label(main_frame, text="Threshold:").grid(row=3, column=0, sticky="w", pady=5)
        
        threshold_frame = ttk.Frame(main_frame)
        threshold_frame.grid(row=3, column=1, sticky="ew", padx=5)
        threshold_frame.columnconfigure(0, weight=1)
        
        self.threshold_slider = ttk.Scale(
            threshold_frame, from_=0.1, to=0.9,
            variable=self.threshold, orient="horizontal",
            command=self.update_threshold_label
        )
        self.threshold_slider.grid(row=0, column=0, sticky="ew")
        
        self.threshold_label = ttk.Label(threshold_frame, text="0.30", width=5)
        self.threshold_label.grid(row=0, column=1, padx=(5, 0))
        
        # DPI
        ttk.Label(main_frame, text="DPI:").grid(row=4, column=0, sticky="w", pady=5)
        
        dpi_frame = ttk.Frame(main_frame)
        dpi_frame.grid(row=4, column=1, sticky="ew", padx=5)
        
        self.dpi_combo = ttk.Combobox(dpi_frame, textvariable=self.dpi, values=[150, 200, 300], width=10, state="readonly")
        self.dpi_combo.current(2)  # Default 300
        self.dpi_combo.grid(row=0, column=0, sticky="w")
        
        ttk.Label(dpi_frame, text="(Higher = slower but more accurate)").grid(row=0, column=1, padx=(10, 0))
        
        # Run Button
        self.run_button = ttk.Button(main_frame, text="Split PDF", command=self.run_split)
        self.run_button.grid(row=5, column=0, columnspan=3, pady=15)
        
        # Progress
        ttk.Label(main_frame, text="Progress:").grid(row=6, column=0, sticky="w")
        
        self.progress = ttk.Progressbar(main_frame, mode="determinate")
        self.progress.grid(row=6, column=1, columnspan=2, sticky="ew", pady=5)
        
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=7, column=0, columnspan=3, sticky="w", pady=5)
        
        # Results
        ttk.Label(main_frame, text="Results:").grid(row=8, column=0, sticky="nw", pady=5)
        
        self.results_text = tk.Text(main_frame, height=8, width=60, state="disabled", wrap="word")
        self.results_text.grid(row=8, column=1, columnspan=2, sticky="nsew", pady=5)
        
        scrollbar = ttk.Scrollbar(main_frame, command=self.results_text.yview)
        scrollbar.grid(row=8, column=3, sticky="ns")
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        main_frame.rowconfigure(8, weight=1)
    
    def browse_pdf(self):
        filename = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.pdf_path.set(filename)
    
    def browse_template(self):
        filename = filedialog.askopenfilename(
            title="Select Template Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if filename:
            self.template_path.set(filename)
    
    def update_threshold_label(self, value):
        self.threshold_label.config(text=f"{float(value):.2f}")
    
    def log_result(self, message):
        self.results_text.config(state="normal")
        self.results_text.insert("end", message + "\n")
        self.results_text.see("end")
        self.results_text.config(state="disabled")
        self.root.update()
    
    def run_split(self):
        if self.running:
            return
        
        pdf_path = self.pdf_path.get()
        template_path = self.template_path.get()
        
        if not pdf_path:
            messagebox.showerror("Error", "Please select a PDF file.")
            return
        
        if not template_path:
            messagebox.showerror("Error", "Please select a template image.")
            return
        
        if not Path(pdf_path).exists():
            messagebox.showerror("Error", f"PDF file not found: {pdf_path}")
            return
        
        if not Path(template_path).exists():
            messagebox.showerror("Error", f"Template file not found: {template_path}")
            return
        
        self.running = True
        self.run_button.config(state="disabled")
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.config(state="disabled")
        
        thread = threading.Thread(target=self.split_pdf_thread, args=(pdf_path, template_path))
        thread.daemon = True
        thread.start()
    
    def split_pdf_thread(self, pdf_path, template_path):
        try:
            import shutil
            
            template = cv2.imread(template_path)
            if template is None:
                raise ValueError("Could not load template image")
            
            output_dir = Path(pdf_path).parent / f"{Path(pdf_path).stem}_split"
            work_dir = output_dir / "temp_pages"
            
            self.log_result(f"Template: {template.shape}")
            self.log_result(f"Output: {output_dir}\n")
            
            # Extract pages
            self.status_label.config(text="Extracting pages...")
            pages = self.extract_pages(pdf_path, work_dir, self.dpi.get())
            
            if not pages:
                raise ValueError("No pages extracted from PDF")
            
            self.progress.config(maximum=len(pages))
            
            # Detect stamps
            self.log_result("Detecting stamps...\n")
            stamp_pages = []
            
            for i, page_path in enumerate(pages):
                self.status_label.config(text=f"Processing page {i+1}/{len(pages)}...")
                self.progress.config(value=i+1)
                
                page_img = cv2.imread(str(page_path))
                found, confidence, scale = self.detect_stamp(page_img, template, self.threshold.get())
                
                status = "✓ STAMP" if found else "✗ None"
                self.log_result(f"  Page {i+1:3d}: {status} (conf={confidence:.2f}, scale={scale:.2f})")
                
                if found:
                    stamp_pages.append(i)
            
            if not stamp_pages:
                self.log_result("\nNo stamps detected.")
                self.status_label.config(text="No stamps found")
                return
            
            # Create split PDFs
            self.status_label.config(text="Creating split PDFs...")
            ranges = []
            for i, stamp_idx in enumerate(stamp_pages):
                start = stamp_idx
                end = stamp_pages[i + 1] if i + 1 < len(stamp_pages) else len(pages)
                ranges.append(list(range(start, end)))
            
            self.log_result(f"\nStamps found on pages: {[p+1 for p in stamp_pages]}")
            self.log_result(f"Creating {len(ranges)} PDF files...\n")
            
            reader = PdfReader(pdf_path)
            
            for i, page_range in enumerate(ranges):
                writer = PdfWriter()
                
                for page_idx in page_range:
                    writer.add_page(reader.pages[page_idx])
                
                stem = Path(pdf_path).stem
                output_path = output_dir / f"{stem}_part{i+1:03d}.pdf"
                
                with open(output_path, 'wb') as f:
                    writer.write(f)
                
                self.log_result(f"  Created: {output_path.name} ({len(page_range)} pages)")
            
            # Cleanup
            if work_dir.exists():
                shutil.rmtree(work_dir)
            
            self.status_label.config(text=f"Complete! Created {len(ranges)} PDF files")
            self.progress.config(value=len(pages))
            
            messagebox.showinfo("Success", f"Created {len(ranges)} PDF files in:\n{output_dir}")
            
        except Exception as e:
            self.log_result(f"\nERROR: {str(e)}")
            self.status_label.config(text="Error occurred")
            messagebox.showerror("Error", str(e))
        
        finally:
            self.running = False
            self.run_button.config(state="normal")
    
    def extract_pages(self, pdf_path, output_dir, dpi=300):
        """Convert PDF pages to images using pdftoppm."""
        output_dir = Path(output_dir)
        
        if output_dir.exists():
            for old_file in output_dir.glob("page-*"):
                old_file.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        pdftoppm = self.find_pdftoppm()
        if pdftoppm is None:
            raise ValueError("pdftoppm not found. Install Poppler.")
        
        result = subprocess.run([
            str(pdftoppm), "-png", "-r", str(dpi),
            str(pdf_path),
            str(output_dir / "page")
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise ValueError(f"pdftoppm failed: {result.stderr}")
        
        return sorted(output_dir.glob("page-*.png"))
    
    def find_pdftoppm(self):
        """Find pdftoppm executable."""
        paths = [
            "pdftoppm",
            r"C:\Program Files\poppler-24.07.0\Library\bin\pdftoppm.exe",
            r"C:\Program Files\poppler-25.07.0\Library\bin\pdftoppm.exe",
            r"C:\Users\aiden\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin\pdftoppm.exe",
            r"C:\Users\acollazo\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin\pdftoppm.exe",
        ]
        
        for p in paths:
            try:
                result = subprocess.run([p, "-v"], capture_output=True, text=True)
                if result.returncode == 0 or "pdftoppm" in result.stderr:
                    return p
            except FileNotFoundError:
                continue
        
        return None
    
    def detect_stamp(self, page_img, template, threshold=0.30):
        """Detect if stamp exists in page image using template matching."""
        # Quick pre-check for red ink
        small = cv2.resize(page_img, (page_img.shape[1]//4, page_img.shape[0]//4))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2
        
        if np.sum(red_mask > 0) < 10:
            return False, 0.0, 0.0
        
        page_gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        scales = np.arange(0.7, 1.35, 0.05)
        best_confidence = 0
        best_scale = 0
        
        for scale in scales:
            h, w = template_gray.shape
            resized = cv2.resize(template_gray, (int(w * scale), int(h * scale)))
            
            if resized.shape[0] > page_gray.shape[0] or resized.shape[1] > page_gray.shape[1]:
                continue
            
            result = cv2.matchTemplate(page_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_confidence:
                best_confidence = max_val
                best_scale = scale
        
        scale_penalty = abs(best_scale - 1.0)
        
        if scale_penalty < 0.15:
            adjusted_confidence = best_confidence * 1.2
        elif scale_penalty < 0.3:
            adjusted_confidence = best_confidence
        else:
            adjusted_confidence = best_confidence * 0.7
        
        found = adjusted_confidence >= threshold
        return found, adjusted_confidence, best_scale


def main():
    root = tk.Tk()
    app = PDFSplitterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
