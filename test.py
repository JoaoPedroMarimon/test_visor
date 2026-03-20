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

# Carregar modelo
print("Carregando modelo...")
#model = YOLO('adesivo_detection/adesivo_detection_results/weights/best.pt')
model = YOLO('adesivo_detection/v2_dual_class3/weights/best.pt')

print("✓ Modelo carregado!")

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
print("INSPETOR DE ADESIVO - VERSÃO RTSP")
print("="*60)
print("📹 Sistema iniciado!")
print("⌨️  Q - Sair | S - Salvar | UP/DOWN - Ajustar sensibilidade")
print("="*60 + "\n")

# Configurações
confidence_threshold = 0.95  # Começar com 5% (modelo tem recall 100%)
detection_history = []
HISTORY_SIZE = 10  # Histórico maior para estabilidade
saved_count = 0
frame_count = 0

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

cv2.namedWindow('Inspetor de Adesivo', cv2.WINDOW_NORMAL)

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

    # Processar detecção apenas a cada N frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # Redimensionar para processamento mais rápido
        frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        results = model.predict(source=frame_resized, conf=confidence_threshold, verbose=False)

        # Processar resultado
        current_detection = None
        best_confidence = 0
        display_frame = frame_resized.copy()

        for result in results:
            boxes = result.boxes

            if len(boxes) > 0:
                # Pegar detecção com maior confiança
                confidences = boxes.conf.cpu().numpy()
                best_idx = confidences.argmax()
                best_box = boxes[best_idx]

                cls = int(best_box.cls)
                conf = float(best_box.conf)
                class_name = result.names[cls]

                current_detection = class_name
                best_confidence = conf

                # Desenhar box com área expandida
                coords = best_box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)

                # Expandir a área de detecção em 15% em todas as direções
                width_box = x2 - x1
                height_box = y2 - y1
                expand = 0.15

                x1 = max(0, int(x1 - width_box * expand))
                y1 = max(0, int(y1 - height_box * expand))
                x2 = min(display_frame.shape[1], int(x2 + width_box * expand))
                y2 = min(display_frame.shape[0], int(y2 + height_box * expand))

                # Cor baseada na classe e confiança
                if class_name == 'com_adesivo':
                    if conf > 0.20:
                        color = (0, 255, 0)  # Verde forte
                    elif conf > 0.15:
                        color = (0, 200, 100)  # Verde médio
                    else:
                        color = (0, 150, 150)  # Verde fraco
                else:
                    color = (0, 165, 255)  # Laranja

                thickness = 3 if conf > 0.20 else 2

                # Desenhar máscara semi-transparente sobre o adesivo
                mask = display_frame.copy()
                cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)  # Retângulo preenchido
                cv2.addWeighted(mask, 0.3, display_frame, 0.7, 0, display_frame)

                # Desenhar borda
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

                # Label
                label = f"{class_name} {conf:.1%}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0] + 10, y1), color, -1)
                cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                display_frame = frame_resized.copy()
                current_detection = None

        # Adicionar à história
        detection_history.append(current_detection)
        if len(detection_history) > HISTORY_SIZE:
            detection_history.pop(0)
    else:
        # Usar último display_frame se não processar
        if 'display_frame' not in locals():
            frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            display_frame = frame_resized.copy()

    # Contar detecções no histórico
    com_adesivo_count = detection_history.count('com_adesivo')
    sem_adesivo_count = detection_history.count('sem_adesivo')

    # Decisão: precisa de 60% dos últimos frames
    threshold_count = int(HISTORY_SIZE * 0.6)

    if com_adesivo_count >= threshold_count:
        stable_status = 'com_adesivo'
        stability = "ESTÁVEL"
    elif sem_adesivo_count >= threshold_count:
        stable_status = 'sem_adesivo'
        stability = "ESTÁVEL"
    else:
        stable_status = None
        stability = "INSTÁVEL"

    # Interface
    height, width = display_frame.shape[:2]

    # Fundo semi-transparente - MAIOR
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 250), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, height - 80), (width, height), (0, 0, 0), -1)
    display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)

    # Título MAIOR
    cv2.putText(display_frame, "INSPETOR DE ADESIVO",
               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Linha separadora
    cv2.line(display_frame, (20, 65), (width - 20, 65), (100, 100, 100), 2)

    # Status MUITO GRANDE
    y_pos = 130
    if stable_status == 'com_adesivo':
        status_text = "COM ADESIVO"
        status_color = (0, 255, 0)
        icon = "✓"
    elif stable_status == 'sem_adesivo':
        status_text = "SEM ADESIVO"
        status_color = (0, 165, 255)
        icon = "✗"
    else:
        status_text = "AGUARDANDO..."
        status_color = (100, 100, 100)
        icon = "?"

    cv2.putText(display_frame, status_text,
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 2.0, status_color, 4)

    # Informações detalhadas MAIORES
    y_pos = 180
    info_text = f"Estabilidade: {stability} ({com_adesivo_count}/{HISTORY_SIZE})"
    cv2.putText(display_frame, info_text,
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if 'best_confidence' in locals() and best_confidence > 0:
        y_pos += 35
        conf_text = f"Confianca: {best_confidence:.1%}"
        cv2.putText(display_frame, conf_text,
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    y_pos += 35
    threshold_text = f"Sensibilidade: {confidence_threshold:.0%}"
    cv2.putText(display_frame, threshold_text,
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # FPS MAIOR
    if len(fps_counter) > 1:
        fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
        cv2.putText(display_frame, f"{fps:.0f} FPS",
                   (width - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Instruções rodapé MAIOR
    cv2.putText(display_frame, "Q=Sair | S=Salvar | UP/DOWN=Sensibilidade",
               (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar
    cv2.imshow('Inspetor de Adesivo', display_frame)

    # Teclas
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("\n👋 Encerrando...")
        running = False
        break
    elif key == ord('s') or key == ord('S'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'captura_{timestamp}.jpg'
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_count += 1
        print(f"📸 Captura salva: {filename}")
    elif key == 82:  # Seta UP
        confidence_threshold = min(0.5, confidence_threshold + 0.01)
        print(f"Sensibilidade: {confidence_threshold:.0%}")
    elif key == 84:  # Seta DOWN
        confidence_threshold = max(0.05, confidence_threshold - 0.01)
        print(f"Sensibilidade: {confidence_threshold:.0%}")

# Limpar
running = False
time.sleep(0.2)
cap.release()
cv2.destroyAllWindows()

print("\n✓ Programa encerrado")
if saved_count > 0:
    print(f"✓ {saved_count} captura(s) salva(s)")
