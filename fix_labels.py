import os

LABELS_DIR = r'C:\Users\jo063877\Downloads\labels_my-project-name_2026-03-16-10-03-32'

label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]
print(f"Encontrados {len(label_files)} arquivos de label")

fixed = 0
for fname in label_files:
    fpath = os.path.join(LABELS_DIR, fname)
    with open(fpath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls = parts[0]
            if cls == '0':
                parts[0] = '1'
            elif cls == '1':
                parts[0] = '0'
            new_lines.append(' '.join(parts) + '\n')
        else:
            new_lines.append(line)

    with open(fpath, 'w') as f:
        f.writelines(new_lines)
    fixed += 1

print(f"✓ {fixed} arquivos corrigidos (0=com_laranja, 1=sem_laranja)")
