"""
RETOMAR TREINAMENTO INTERROMPIDO
Use este script se o treinamento parou antes de terminar
"""

from ultralytics import YOLO
import torch
import os

print("="*60)
print("RETOMAR TREINAMENTO - YOLOV8")
print("="*60)

# Verificar GPU
print(f"\n✓ PyTorch versão: {torch.__version__}")
print(f"✓ CUDA disponível: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU detectada: {torch.cuda.get_device_name(0)}")
    device = 0
else:
    print("⚠ GPU não detectada. Treinamento será na CPU")
    device = 'cpu'

# Procurar checkpoint mais recente
print("\n" + "="*60)
print("PROCURANDO CHECKPOINTS...")
print("="*60)

# Verificar pastas de treinamento
training_folders = [
    'adesivo_detection/run1',
    'adesivo_detection/v2_dual_class',
    'adesivo_detection/run2',
]

found_checkpoints = []

for folder in training_folders:
    last_pt = f"{folder}/weights/last.pt"
    if os.path.exists(last_pt):
        found_checkpoints.append(last_pt)
        print(f"✓ Encontrado: {last_pt}")

if not found_checkpoints:
    print("❌ Nenhum checkpoint encontrado!")
    print("\nCheckpoints são salvos em:")
    print("  - adesivo_detection/run1/weights/last.pt")
    print("  - adesivo_detection/v2_dual_class/weights/last.pt")
    print("\nSe você não tem checkpoint, execute train.py ou trainv2.py")
    exit(1)

# Selecionar checkpoint
if len(found_checkpoints) == 1:
    checkpoint = found_checkpoints[0]
    print(f"\n✓ Usando checkpoint: {checkpoint}")
else:
    print(f"\n📋 Encontrados {len(found_checkpoints)} checkpoints:")
    for i, ckpt in enumerate(found_checkpoints):
        print(f"  {i+1}. {ckpt}")

    choice = input("\nEscolha o número do checkpoint para retomar: ").strip()
    try:
        checkpoint = found_checkpoints[int(choice) - 1]
    except:
        print("❌ Escolha inválida!")
        exit(1)

# Retomar treinamento
print("\n" + "="*60)
print("RETOMANDO TREINAMENTO...")
print("="*60)
print(f"📂 Checkpoint: {checkpoint}")
print("⏳ Isso pode levar várias horas...")
print("="*60 + "\n")

# Carregar modelo do checkpoint
model = YOLO(checkpoint)

# Retomar treinamento
# O YOLO automaticamente continua de onde parou
results = model.train(
    resume=True,  # IMPORTANTE: retoma do checkpoint
    device=device
)

print("\n" + "="*60)
print("✓ TREINAMENTO RETOMADO CONCLUÍDO!")
print("="*60)

# Validar
print("\n" + "="*60)
print("Validando modelo...")
print("="*60)

metrics = model.val()

print(f"\n📊 MÉTRICAS FINAIS:")
print(f"   mAP50: {metrics.box.map50:.3f}")
print(f"   mAP50-95: {metrics.box.map:.3f}")
print(f"   Precisão: {metrics.box.mp:.3f}")
print(f"   Recall: {metrics.box.mr:.3f}")

print("\n✓ Pronto!")
