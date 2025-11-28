from ultralytics import YOLO
import torch

print("="*60)
print("TREINAMENTO V2 - DETECÇÃO COM/SEM ADESIVO")
print("="*60)

# Verificar se CUDA está disponível
print(f"\n✓ PyTorch versão: {torch.__version__}")
print(f"✓ CUDA disponível: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU detectada: {torch.cuda.get_device_name(0)}")
    device = 0
else:
    print("⚠ GPU não detectada. Treinamento será na CPU (mais lento)")
    device = 'cpu'

print("\n" + "="*60)
print("Iniciando treinamento com 2 classes...")
print("Classes: com_adesivo | sem_adesivo")
print("="*60)

# Carregar modelo pré-treinado YOLOv8 nano (o mais leve)
model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',       # Já configurado com 2 classes
    epochs=100,
    imgsz=640,
    batch=4,
    patience=20,
    device=device,
    project='adesivo_detection',
    name='v2_dual_class',   # Nome diferente para não sobrescrever
    save=True,
    plots=True,
    workers=1,
    verbose=True
)


print("\n" + "="*60)
print("✓ TREINAMENTO CONCLUÍDO!")
print("="*60)
print(f"\n📊 Resultados salvos em: adesivo_detection/v2_dual_class/")
print(f"🏆 Melhor modelo: adesivo_detection/v2_dual_class/weights/best.pt")
print(f"📈 Gráficos: adesivo_detection/v2_dual_class/*.png")

# Validar o modelo
print("\n" + "="*60)
print("Validando modelo...")
print("="*60)

model = YOLO('adesivo_detection/v2_dual_class/weights/best.pt')
metrics = model.val()

print(f"\n📊 MÉTRICAS FINAIS (TODAS AS CLASSES):")
print(f"   mAP50: {metrics.box.map50:.3f}")
print(f"   mAP50-95: {metrics.box.map:.3f}")
print(f"   Precisão: {metrics.box.mp:.3f}")
print(f"   Recall: {metrics.box.mr:.3f}")

# Métricas por classe (se disponível)
if hasattr(metrics.box, 'maps'):
    print(f"\n📊 MÉTRICAS POR CLASSE:")
    class_names = ['com_adesivo', 'sem_adesivo']
    for i, name in enumerate(class_names):
        if i < len(metrics.box.maps):
            print(f"   {name}:")
            print(f"      mAP50-95: {metrics.box.maps[i]:.3f}")

print("\n✓ Pronto para usar! Execute testv2.py para testar em novas imagens.")
