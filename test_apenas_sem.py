from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
import threading

# ===========================================
# CONFIGURAÇÃO DA CÂMERA IP
# ===========================================
CAMERA_IP = "192.168.1.108"
CAMERA_USER = "admin"
CAMERA_PASS = "admin123"

# URL RTSP (Dahua)
RTSP_URL = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=0"

# Configurações
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
PROCESS_EVERY_N_FRAMES = 2  # Processar detecção a cada 2 frames

# Carregar modelo V2 (2 classes)
print("Carregando modelo V2 - APENAS SEM ADESIVO...")
model = YOLO('adesivo_detection/run3/weights/best.pt')
print("✓ Modelo V2 carregado!")

# Conectar via RTSP
print(f"\nConectando ao stream RTSP...")
print(f"  URL: {RTSP_URL.replace(CAMERA_PASS, '***')}")

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Testar conexão
print("Testando conexão...")
connected = False
for _ in range(10):
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"✓ Conectado! Resolução: {frame.shape[1]}x{frame.shape[0]}")
        connected = True
        break
    time.sleep(0.2)

if not connected:
    print("❌ Não foi possível conectar ao RTSP")
    exit(1)

print("\n" + "="*60)
print("INSPETOR - APENAS SEM ADESIVO")
print("="*60)
print("📹 Sistema iniciado!")
print("🟠 Mostra APENAS quando detectar SEM_ADESIVO")
print("⌨️  Q - Sair | S - Salvar | UP/DOWN - Ajustar sensibilidade")
print("="*60 + "\n")

# Configurações
confidence_threshold = 0.3  # Limiar mais baixo para sem_adesivo
detection_history = []
HISTORY_SIZE = 10
saved_count = 0
frame_count = 0

print("MODO: APENAS SEM_ADESIVO")
print("Mascara aparece somente quando detectar sem_adesivo\n")

# Thread de captura
current_frame = None
frame_lock = threading.Lock()
running = True

def capture_thread():
    """Thread dedicada para capturar frames"""
    global current_frame, running

    while running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame

capture_thread_obj = threading.Thread(target=capture_thread, daemon=True)
capture_thread_obj.start()

# Esperar primeiro frame
print("Aguardando frames...")
while current_frame is None:
    time.sleep(0.1)

print("✓ Recebendo frames!\n")

# Variáveis
last_frame_id = None
fps_counter = deque(maxlen=30)

# Contador de detecções
sem_adesivo_count_total = 0

cv2.namedWindow('Inspetor - Apenas SEM ADESIVO', cv2.WINDOW_NORMAL)

while running:
    # Pegar frame atual
    with frame_lock:
        if current_frame is None:
            cv2.waitKey(1)
            continue

        # Evitar processar o mesmo frame
        frame_id = id(current_frame)
        if frame_id == last_frame_id:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                running = False
                break
            continue

        last_frame_id = frame_id
        frame = current_frame.copy()

    frame_count += 1
    current_time = time.time()
    fps_counter.append(current_time)

    # Inicializar variável de detecções
    sem_adesivo_detections = []

    # Processar detecção apenas a cada N frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # Redimensionar para processamento mais rápido
        frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        results = model.predict(source=frame_resized, conf=confidence_threshold, verbose=False)

        # Processar resultado - APENAS SEM_ADESIVO
        display_frame = frame_resized.copy()

        for result in results:
            boxes = result.boxes

            if len(boxes) > 0:
                # Processar APENAS detecções de sem_adesivo
                for box in boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    class_name = result.names[cls]

                    # FILTRAR: Apenas sem_adesivo
                    if class_name == 'sem_adesivo':
                        coords = box.xyxy[0].cpu().numpy()
                        sem_adesivo_detections.append({
                            'conf': conf,
                            'coords': coords
                        })

        # Desenhar APENAS máscaras de sem_adesivo
        for detection in sem_adesivo_detections:
            conf = detection['conf']
            x1, y1, x2, y2 = map(int, detection['coords'])

            # Expandir a área de detecção em 15%
            width_box = x2 - x1
            height_box = y2 - y1
            expand = 0.15

            x1 = max(0, int(x1 - width_box * expand))
            y1 = max(0, int(y1 - height_box * expand))
            x2 = min(display_frame.shape[1], int(x2 + width_box * expand))
            y2 = min(display_frame.shape[0], int(y2 + height_box * expand))

            # Cor LARANJA para SEM ADESIVO
            if conf > 0.70:
                color = (0, 100, 255)  # Laranja forte
            elif conf > 0.50:
                color = (0, 140, 255)  # Laranja médio
            else:
                color = (0, 165, 200)  # Laranja fraco

            thickness = 3 if conf > 0.70 else 2

            # Desenhar máscara semi-transparente LARANJA
            mask = display_frame.copy()
            cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)  # Retângulo preenchido
            cv2.addWeighted(mask, 0.5, display_frame, 0.5, 0, display_frame)

            # Desenhar borda
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

            # Label
            label = f"SEM ADESIVO {conf:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Adicionar à história
        if sem_adesivo_detections:
            detection_history.append('sem_adesivo')
            sem_adesivo_count_total += 1
        else:
            detection_history.append(None)

        if len(detection_history) > HISTORY_SIZE:
            detection_history.pop(0)
    else:
        # Usar último display_frame se não processar
        if 'display_frame' not in locals():
            frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            display_frame = frame_resized.copy()

    # Contar detecções no histórico
    sem_adesivo_count = detection_history.count('sem_adesivo')

    # Decisão: precisa de 60% dos últimos frames
    threshold_count = int(HISTORY_SIZE * 0.6)

    if sem_adesivo_count >= threshold_count:
        stable_status = 'sem_adesivo'
        stability = "ESTÁVEL"
    else:
        stable_status = None
        stability = "INSTÁVEL"

    # Interface
    height, width = display_frame.shape[:2]

    # Fundo semi-transparente
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 220), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, height - 80), (width, height), (0, 0, 0), -1)
    display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)

    # Título
    cv2.putText(display_frame, "INSPETOR - APENAS SEM ADESIVO",
               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    # Linha separadora
    cv2.line(display_frame, (20, 65), (width - 20, 65), (100, 100, 100), 2)

    # Status
    y_pos = 130
    if stable_status == 'sem_adesivo':
        status_text = "SEM ADESIVO DETECTADO"
        status_color = (0, 100, 255)  # Laranja
    else:
        status_text = "AGUARDANDO..."
        status_color = (100, 100, 100)  # Cinza

    cv2.putText(display_frame, status_text,
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.8, status_color, 4)

    # Informações
    y_pos = 175
    info_text = f"Estabilidade: {stability} ({sem_adesivo_count}/{HISTORY_SIZE})"
    cv2.putText(display_frame, info_text,
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Confiança da melhor detecção
    if sem_adesivo_detections:
        best_det = max(sem_adesivo_detections, key=lambda x: x['conf'])
        y_pos += 30
        conf_text = f"Confianca: {best_det['conf']:.1%}"
        cv2.putText(display_frame, conf_text,
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # FPS
    if len(fps_counter) > 1:
        fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
        cv2.putText(display_frame, f"{fps:.0f} FPS",
                   (width - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Instruções rodapé
    cv2.putText(display_frame, "Q=Sair | S=Salvar | UP/DOWN=Sensibilidade",
               (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar
    cv2.imshow('Inspetor - Apenas SEM ADESIVO', display_frame)

    # Teclas
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("\n👋 Encerrando...")
        running = False
        break
    elif key == ord('s') or key == ord('S'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'captura_sem_adesivo_{timestamp}.jpg'
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_count += 1
        print(f"📸 Captura salva: {filename}")
    elif key == 82:  # Seta UP
        confidence_threshold = min(0.95, confidence_threshold + 0.05)
        print(f"Sensibilidade: {confidence_threshold:.0%}")
    elif key == 84:  # Seta DOWN
        confidence_threshold = max(0.10, confidence_threshold - 0.05)
        print(f"Sensibilidade: {confidence_threshold:.0%}")

# Limpar
running = False
time.sleep(0.2)
cap.release()
cv2.destroyAllWindows()

print("\n✓ Programa encerrado")
if saved_count > 0:
    print(f"✓ {saved_count} captura(s) salva(s)")

# Estatísticas finais
print(f"\n📊 ESTATÍSTICAS DA SESSÃO:")
print(f"   Total de frames processados: {frame_count}")
print(f"   Detecções SEM ADESIVO: {sem_adesivo_count_total}")
