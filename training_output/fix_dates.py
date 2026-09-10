# Correct dates for all stamps (7 rows x 3 stamps = 21, plus 6 extra = 27 total)
# First 12 stamps: unique dates in order
# Remaining 15 stamps: SEP 10 2027

dates = [
    'JAN 01 2026', 'FEB 02 2026', 'MAR 13 2026',   # Row 1
    'APR 14 2026', 'MAY 25 2026', 'JUN 25 2026',   # Row 2
    'JUL 16 2026', 'AUG 31 2026', 'SEP 27 2026',   # Row 3
    'OCT 28 2026', 'NOV 29 2026', 'DEC 30 2026',   # Row 4
    'SEP 10 2027', 'SEP 10 2027', 'SEP 10 2027',   # Row 5
    'SEP 10 2027', 'SEP 10 2027', 'SEP 10 2027',   # Row 6
    'SEP 10 2027', 'SEP 10 2027', 'SEP 10 2027',   # Row 7
    'SEP 10 2027', 'SEP 10 2027', 'SEP 10 2027',   # Extra
    'SEP 10 2027', 'SEP 10 2027', 'SEP 10 2027',   # Extra
]

for i, date in enumerate(dates, 1):
    filename = f'stamp_{i:03d}.gt.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f'RECEIVED\n{date}\nBY:')
    print(f'{filename}: RECEIVED / {date} / BY:')

print('Done!')
