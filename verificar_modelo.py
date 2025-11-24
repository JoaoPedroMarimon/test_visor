from ultralytics import YOLO
import cv2
import os

# Carregar modelo
print("Carregando modelo...")
model = YOLO('adesivo_detection/run1/weights/best.pt')
print("✓ Modelo carregado!")

# Mostrar informações do modelo
print("\n" + "="*60)
print("INFORMAÇÕES DO MODELO")
print("="*60)
print(f"Classes: {model.names}")
print(f"Número de classes: {len(model.names)}")
print("="*60)

# Testar em algumas imagens do dataset
print("\nTestando modelo em imagens do dataset...")

# Verificar se existem imagens
test_folders = [
    'novos_dados_ip/com_adesivo',
    'novos_dados_ip/sem_adesivo',
    'dataset/images/val'
]

found_images = []
for folder in test_folders:
    if os.path.exists(folder):
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            found_images.extend([os.path.join(folder, img) for img in images[:3]])  # Pegar 3 de cada

if not found_images:
    print("❌ Nenhuma imagem encontrada para testar!")
    print("Crie algumas imagens primeiro com coletar_dados.py")
    exit(1)

print(f"✓ Encontradas {len(found_images)} imagens para testar\n")

# Testar com diferentes thresholds
thresholds = [0.01, 0.05, 0.10, 0.25, 0.50]

for img_path in found_images[:5]:  # Testar apenas 5 imagens
    print(f"\n{'='*60}")
    print(f"Testando: {os.path.basename(img_path)}")
    print(f"{'='*60}")

    # Ler imagem
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠ Erro ao ler imagem: {img_path}")
        continue

    print(f"Resolução: {img.shape[1]}x{img.shape[0]}")

    # Testar com diferentes thresholds
    for threshold in thresholds:
        results = model.predict(source=img, conf=threshold, verbose=False)

        detections = 0
        for result in results:
            detections = len(result.boxes)
            if detections > 0:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    class_name = result.names[cls]
                    print(f"  Threshold {threshold:.2f}: {class_name} ({conf:.2%})")

        if detections == 0:
            print(f"  Threshold {threshold:.2f}: Nenhuma detecção")

    # Salvar resultado visual com threshold baixo
    results = model.predict(source=img, conf=0.01, verbose=False)

    for result in results:
        # Desenhar resultado
        annotated = result.plot()

        # Salvar
        output_path = f'teste_modelo_{os.path.basename(img_path)}'
        cv2.imwrite(output_path, annotated)
        print(f"\n✓ Resultado salvo em: {output_path}")

        # Mostrar na tela
        cv2.imshow('Teste do Modelo (Q para próxima)', annotated)
        key = cv2.waitKey(0)
        if key == ord('q') or key == ord('Q'):
            break

cv2.destroyAllWindows()

print("\n" + "="*60)
print("RESUMO DA VERIFICAÇÃO")
print("="*60)
print("Se não houve detecções:")
print("1. Verifique se o modelo foi treinado corretamente")
print("2. Verifique se as classes no data.yaml estão corretas")
print("3. Tente rodar: python train.py novamente")
print("="*60)
