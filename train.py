from ultralytics import YOLO
import torch

print("="*60)
print("TREINAMENTO - DETECÇÃO DE ADESIVO")
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
print("Iniciando treinamento...")
print("="*60)

# Carregar modelo pré-treinado YOLOv8 nano (o mais leve)
model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=4,            # antes 16
    patience=20,
    device=device,
    project='adesivo_detection',
    name='run2',
    save=True,
    plots=False,        # antes True
    workers=1,          # antes 4
    verbose=True
)


print("\n" + "="*60)
print("✓ TREINAMENTO CONCLUÍDO!")
print("="*60)
print(f"\n📊 Resultados salvos em: adesivo_detection/run1/")
print(f"🏆 Melhor modelo: adesivo_detection/run1/weights/best.pt")
print(f"📈 Gráficos: adesivo_detection/run1/*.png")

# Validar o modelo
print("\n" + "="*60)
print("Validando modelo...")
print("="*60)

model = YOLO('adesivo_detection/run1/weights/best.pt')
metrics = model.val()

print(f"\n📊 MÉTRICAS FINAIS:")
print(f"   mAP50: {metrics.box.map50:.3f}")
print(f"   mAP50-95: {metrics.box.map:.3f}")
print(f"   Precisão: {metrics.box.mp:.3f}")
print(f"   Recall: {metrics.box.mr:.3f}")

print("\n✓ Pronto para usar! Execute test.py para testar em novas imagens.")