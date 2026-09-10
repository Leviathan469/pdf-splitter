from PIL import Image
from pathlib import Path

def generate_box_file(image_path, gt_path, output_path):
    """
    Generate a Tesseract box file from an image and its ground truth text.
    
    Box format: char left bottom right top page (bottom-left origin)
    
    The stamp has 3 lines:
      Top 40%: RECEIVED
      Middle 35%: DATE (e.g., JAN 01 2026)
      Bottom 25%: BY:
    """
    img = Image.open(image_path)
    img_w, img_h = img.size
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    # Vertical zones for each line
    zones = [
        (0, int(img_h * 0.40)),           # RECEIVED: top 40%
        (int(img_h * 0.35), int(img_h * 0.70)),  # DATE: middle 35%
        (int(img_h * 0.65), img_h)         # BY: bottom 25%
    ]
    
    boxes = []
    
    for line_idx, (line, (zone_top, zone_bottom)) in enumerate(zip(lines, zones)):
        if not line:
            continue
        
        chars = list(line)
        char_width = img_w / len(chars)
        
        for char_idx, char in enumerate(chars):
            left = int(char_idx * char_width)
            right = int((char_idx + 1) * char_width)
            
            # Tesseract box format uses bottom-left origin
            # So we need to flip y coordinates
            boxes.append((char, left, img_h - zone_bottom, right, img_h - zone_top, 0))
    
    # Write box file
    with open(output_path, 'w', encoding='utf-8') as f:
        for char, left, bottom, right, top, page in boxes:
            f.write(f'{char} {left} {bottom} {right} {top} {page}\n')
    
    return boxes

# Generate box files for all stamps
training_dir = Path('.')
gt_files = sorted(training_dir.glob('*.gt.txt'))

for gt_file in gt_files:
    stem = gt_file.stem.replace('.gt', '')
    image_path = training_dir / f'{stem}.png'
    box_path = training_dir / f'{stem}.box'
    
    if image_path.exists():
        boxes = generate_box_file(str(image_path), str(gt_file), str(box_path))
        print(f'{stem}.box: {len(boxes)} chars')

print('All box files generated!')
