from ultralytics import YOLO
import torch

print("="*60)
print("TREINAMENTO V3 - DETECÇÃO 2 CLASSES")
print("="*60)

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
print("Classes: com_laranja, sem_laranja")
print("="*60)

# Carregar modelo pré-treinado YOLOv8 nano (o mais leve)
model = YOLO('yolov8n.pt')

results = model.train(
    data='data_laranja.yaml',
    epochs=100,
    imgsz=640,
    batch=4,
    patience=20,
    device=device,
    project='adesivo_detection',
    name='v3_duas_classes',
    save=True,
    plots=True,
    workers=1,
    verbose=True
)

print("\n" + "="*60)
print("✓ TREINAMENTO CONCLUÍDO!")
print("="*60)
print(f"\n📊 Resultados salvos em: adesivo_detection/v3_duas_classes/")
print(f"🏆 Melhor modelo: adesivo_detection/v3_duas_classes/weights/best.pt")
print(f"📈 Gráficos: adesivo_detection/v3_duas_classes/*.png")

# Validar o modelo
print("\n" + "="*60)
print("Validando modelo...")
print("="*60)

model = YOLO('adesivo_detection/v3_duas_classes/weights/best.pt')
metrics = model.val()

print(f"\n📊 MÉTRICAS FINAIS:")
print(f"   mAP50:    {metrics.box.map50:.3f}")
print(f"   mAP50-95: {metrics.box.map:.3f}")
print(f"   Precisão: {metrics.box.mp:.3f}")
print(f"   Recall:   {metrics.box.mr:.3f}")

if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 0:
    print(f"\n📊 MÉTRICAS POR CLASSE:")
    classes = ['com_laranja', 'sem_laranja']
    for i, cls in enumerate(classes):
        if i < len(metrics.box.maps):
            print(f"   {cls}:")
            print(f"      mAP50-95: {metrics.box.maps[i]:.3f}")

print("\n✓ Pronto! Modelo salvo em: adesivo_detection/v3_duas_classes/weights/best.pt")
