import os
import shutil
from sklearn.model_selection import train_test_split

# ===========================================
# AJUSTE ESTES CAMINHOS CONFORME SEU SISTEMA
# ===========================================
SOURCE_IMAGES_LARANJA = r'C:\Users\jo063877\Pictures\Camera Roll'
SOURCE_LABELS = r'C:\Users\jo063877\Downloads\labels_my-project-name_2026-03-26-07-47-40'

# Criar estrutura de pastas
print("Criando estrutura de pastas...")
os.makedirs('dataset/images/train', exist_ok=True)
os.makedirs('dataset/images/val', exist_ok=True)
os.makedirs('dataset/labels/train', exist_ok=True)
os.makedirs('dataset/labels/val', exist_ok=True)
print("✓ Pastas criadas!")

# Verificar se os caminhos existem
if not os.path.exists(SOURCE_IMAGES_LARANJA):
    print(f"ERRO: Pasta não encontrada: {SOURCE_IMAGES_LARANJA}")
    exit(1)

if not os.path.exists(SOURCE_LABELS):
    print(f"ERRO: Pasta não encontrada: {SOURCE_LABELS}")
    exit(1)

# Listar imagens de com_laranja
image_files = [
    f for f in os.listdir(SOURCE_IMAGES_LARANJA)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

if len(image_files) == 0:
    print("ERRO: Nenhuma imagem encontrada na pasta!")
    exit(1)

print(f"\n✓ Encontradas {len(image_files)} imagens (com_laranja + sem_laranja)")

# Verificar labels disponíveis
label_files = [f for f in os.listdir(SOURCE_LABELS) if f.endswith('.txt')]
print(f"✓ Encontrados {len(label_files)} arquivos de label")

if len(label_files) < len(image_files):
    print(f"⚠ Atenção: {len(image_files)} imagens mas apenas {len(label_files)} labels!")

# Dividir em treino (80%) e validação (20%)
train_files, val_files = train_test_split(
    image_files,
    test_size=0.2,
    random_state=42
)

print(f"\nDivisão do dataset:")
print(f"  Treino    : {len(train_files)} imagens")
print(f"  Validação : {len(val_files)} imagens")


def safe_copy(src, dst, tipo="arquivo"):
    try:
        if os.path.exists(src):
            shutil.copy(src, dst)
            return True
        else:
            print(f"⚠ {tipo} não encontrado: {src}")
            return False
    except Exception as e:
        print(f"✗ Erro ao copiar {src}: {e}")
        return False


# Copiar arquivos de TREINO
print("\nCopiando arquivos de treino...")
train_imgs = 0
train_lbls = 0

for img_file in train_files:
    src_img = os.path.join(SOURCE_IMAGES_LARANJA, img_file)
    if safe_copy(src_img, os.path.join('dataset/images/train', img_file), "Imagem"):
        train_imgs += 1

    label_file = os.path.splitext(img_file)[0] + '.txt'
    if safe_copy(os.path.join(SOURCE_LABELS, label_file),
                 os.path.join('dataset/labels/train', label_file), "Label"):
        train_lbls += 1

print(f"✓ Treino: {train_imgs} imagens, {train_lbls} labels")

# Copiar arquivos de VALIDAÇÃO
print("\nCopiando arquivos de validação...")
val_imgs = 0
val_lbls = 0

for img_file in val_files:
    src_img = os.path.join(SOURCE_IMAGES_LARANJA, img_file)
    if safe_copy(src_img, os.path.join('dataset/images/val', img_file), "Imagem"):
        val_imgs += 1

    label_file = os.path.splitext(img_file)[0] + '.txt'
    if safe_copy(os.path.join(SOURCE_LABELS, label_file),
                 os.path.join('dataset/labels/val', label_file), "Label"):
        val_lbls += 1

print(f"✓ Validação: {val_imgs} imagens, {val_lbls} labels")

# Resumo final
print("\n" + "="*50)
print("RESUMO FINAL")
print("="*50)
print(f"  Total de imagens : {len(image_files)}")
print(f"  Treino           : {train_imgs} imagens, {train_lbls} labels")
print(f"  Validação        : {val_imgs} imagens, {val_lbls} labels")
print("="*50)

if train_lbls < train_imgs or val_lbls < val_imgs:
    print("\n⚠ Algumas imagens não têm label! Verifique as anotações.")

print("\n✓ Organização completa!")
print("Próximo passo: rodar trainv2.py")
