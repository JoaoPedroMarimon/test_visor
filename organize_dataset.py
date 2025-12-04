import os
import shutil
from sklearn.model_selection import train_test_split

# ===========================================
# AJUSTE ESTES CAMINHOS CONFORME SEU SISTEMA
# ===========================================
# Use r'' (raw string) ou barras duplas \\ ou barras normais / no Windows
SOURCE_IMAGES_COM = r'C:\Users\jo060842\Documents\porj_adesivo\novos_dados_ip\com_adesivo'
SOURCE_IMAGES_SEM = r'C:\Users\jo060842\Documents\porj_adesivo\novos_dados_ip\sem_adesivo'
SOURCE_LABELS = r'C:\Users\jo060842\Downloads\esse(1)'
# Criar estrutura de pastas
print("Criando estrutura de pastas...")
os.makedirs('dataset/images/train', exist_ok=True)
os.makedirs('dataset/images/val', exist_ok=True)
os.makedirs('dataset/labels/train', exist_ok=True)
os.makedirs('dataset/labels/val', exist_ok=True)
print("✓ Pastas criadas!")

# Verificar se os caminhos existem
if not os.path.exists(SOURCE_IMAGES_COM):
    print(f"ERRO: Pasta {SOURCE_IMAGES_COM} não encontrada!")
    print(f"Caminho absoluto procurado: {os.path.abspath(SOURCE_IMAGES_COM)}")
    exit(1)

if not os.path.exists(SOURCE_IMAGES_SEM):
    print(f"ERRO: Pasta {SOURCE_IMAGES_SEM} não encontrada!")
    print(f"Caminho absoluto procurado: {os.path.abspath(SOURCE_IMAGES_SEM)}")
    exit(1)

if not os.path.exists(SOURCE_LABELS):
    print(f"ERRO: Pasta {SOURCE_LABELS} não encontrada!")
    print(f"Caminho absoluto procurado: {os.path.abspath(SOURCE_LABELS)}")
    exit(1)

# Listar todas as imagens de ambas as pastas
image_files_com = [f for f in os.listdir(SOURCE_IMAGES_COM)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_files_sem = [f for f in os.listdir(SOURCE_IMAGES_SEM)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Combinar todas as imagens
image_files = []
for f in image_files_com:
    image_files.append(('com', f))
for f in image_files_sem:
    image_files.append(('sem', f))

if len(image_files) == 0:
    print(f"ERRO: Nenhuma imagem encontrada nas pastas!")
    exit(1)

print(f"\n✓ Encontradas {len(image_files)} imagens")
print(f"  - COM adesivo: {len(image_files_com)}")
print(f"  - SEM adesivo: {len(image_files_sem)}")

# Verificar quantos labels existem
label_files = [f for f in os.listdir(SOURCE_LABELS) if f.endswith('.txt')]
print(f"✓ Encontrados {len(label_files)} arquivos de label")

if len(label_files) < len(image_files):
    print(f"⚠ ATENÇÃO: Você tem {len(image_files)} imagens mas apenas {len(label_files)} labels!")
    print("Algumas imagens podem não ter anotações.")

# Dividir em treino (80%) e validação (20%)
train_files, val_files = train_test_split(
    image_files,
    test_size=0.2,
    random_state=42
)

print(f"\nDivisão do dataset:")
print(f"- Treino: {len(train_files)} imagens")
print(f"- Validação: {len(val_files)} imagens")

# Função para copiar arquivo com verificação
def safe_copy(src, dst, file_type="arquivo"):
    try:
        if os.path.exists(src):
            shutil.copy(src, dst)
            return True
        else:
            print(f"⚠ {file_type} não encontrado: {src}")
            return False
    except Exception as e:
        print(f"✗ Erro ao copiar {src}: {e}")
        return False

# Copiar arquivos de TREINO
print("\nCopiando arquivos de treino...")
train_copied = 0
train_labels_copied = 0

for folder, img_file in train_files:
    # Determinar pasta de origem
    if folder == 'com':
        src_img = os.path.join(SOURCE_IMAGES_COM, img_file)
    else:
        src_img = os.path.join(SOURCE_IMAGES_SEM, img_file)

    dst_img = os.path.join('dataset/images/train', img_file)
    if safe_copy(src_img, dst_img, "Imagem"):
        train_copied += 1

    # Copiar label correspondente
    label_file = os.path.splitext(img_file)[0] + '.txt'
    src_label = os.path.join(SOURCE_LABELS, label_file)
    dst_label = os.path.join('dataset/labels/train', label_file)
    if safe_copy(src_label, dst_label, "Label"):
        train_labels_copied += 1

print(f"✓ Treino: {train_copied} imagens e {train_labels_copied} labels copiados")

# Copiar arquivos de VALIDAÇÃO
print("\nCopiando arquivos de validação...")
val_copied = 0
val_labels_copied = 0

for folder, img_file in val_files:
    # Determinar pasta de origem
    if folder == 'com':
        src_img = os.path.join(SOURCE_IMAGES_COM, img_file)
    else:
        src_img = os.path.join(SOURCE_IMAGES_SEM, img_file)

    dst_img = os.path.join('dataset/images/val', img_file)
    if safe_copy(src_img, dst_img, "Imagem"):
        val_copied += 1

    # Copiar label correspondente
    label_file = os.path.splitext(img_file)[0] + '.txt'
    src_label = os.path.join(SOURCE_LABELS, label_file)
    dst_label = os.path.join('dataset/labels/val', label_file)
    if safe_copy(src_label, dst_label, "Label"):
        val_labels_copied += 1

print(f"✓ Validação: {val_copied} imagens e {val_labels_copied} labels copiados")

# Resumo final
print("\n" + "="*50)
print("RESUMO FINAL")
print("="*50)
print(f"Total de imagens processadas: {len(image_files)}")
print(f"Treino: {train_copied} imagens, {train_labels_copied} labels")
print(f"Validação: {val_copied} imagens, {val_labels_copied} labels")
print("="*50)

if train_labels_copied < train_copied or val_labels_copied < val_copied:
    print("\n⚠ ATENÇÃO: Algumas imagens não têm labels correspondentes!")
    print("Verifique se você anotou todas as imagens no Makesense.")

print("\n✓ Organização completa!")
print("Próximo passo: criar o arquivo data.yaml e rodar train.py")