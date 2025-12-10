from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
import threading
import torch

# ===========================================
# CONFIGURAÇÃO DA CÂMERA IP
# ===========================================
CAMERA_IP = "192.168.1.108"
CAMERA_USER = "admin"
CAMERA_PASS = "admin123"

# URL RTSP (Dahua)
RTSP_URL = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=0"

# Configurações de Performance (ajustadas para CPU)
WINDOW_WIDTH = 800  # Reduzido de 1024
WINDOW_HEIGHT = 600  # Reduzido de 768
PROCESS_EVERY_N_FRAMES = 4  # Aumentado de 3 para 4
INFERENCE_SIZE = 320  # Reduzido de 416 para 320 (muito mais rápido)

# Detectar dispositivo disponível
if torch.cuda.is_available():
    device = 0
    device_name = torch.cuda.get_device_name(0)
    print(f"✓ GPU detectada: {device_name}")
else:
    device = 'cpu'
    print("⚠ GPU não detectada. Usando CPU (mais lento)")

# Carregar modelo V2 (2 classes) com otimizações
print("Carregando modelo V2 (com/sem adesivo)...")
model = YOLO('adesivo_detection/v2_dual_class3/weights/best.pt')
model.fuse()  # Fusão de camadas para maior velocidade
print("✓ Modelo V2 carregado e otimizado!")

# Conectar via RTSP
print(f"\nConectando ao stream RTSP...")
print(f"  URL: {RTSP_URL.replace(CAMERA_PASS, '***')}")

cap = cv2.VideoCapture(RTSP_URL)
# Otimizações de captura
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para reduzir latência
cap.set(cv2.CAP_PROP_FPS, 15)  # Limitar FPS da captura

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
print("INSPETOR DE ADESIVO V2 - DETECÇÃO DUAL CLASS")
print("="*60)
print("📹 Sistema iniciado!")
print("🎨 Classes: COM_ADESIVO (Verde) | SEM_ADESIVO (Laranja)")
print(f"⚡ Otimizações: Inferência {INFERENCE_SIZE}x{INFERENCE_SIZE} | Skip {PROCESS_EVERY_N_FRAMES} frames")
print("⌨️  Q - Sair | S - Salvar | 1/2 - COM | 3/4 - SEM")
print("="*60 + "\n")

# Configurações - THRESHOLDS SEPARADOS POR CLASSE
threshold_com_adesivo = 0.80    # Limiar para com_adesivo
threshold_sem_adesivo = 0.80    # Limiar para sem_adesivo (mais sensível)
confidence_threshold_geral = 0.7  # Limiar geral para o modelo detectar

detection_history = []
HISTORY_SIZE = 10
saved_count = 0
frame_count = 0

# LÓGICA DE PRIORIDADE COM THRESHOLDS SEPARADOS:
# - com_adesivo precisa >= threshold_com_adesivo (ex: 50%)
# - sem_adesivo precisa >= threshold_sem_adesivo (ex: 30%)
# - sem_adesivo tem PRIORIDADE se atingir seu threshold
# - Permite ajustar sensibilidade diferente para cada classe

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

# Contadores de detecções por classe
class_counters = {
    'com_adesivo': 0,
    'sem_adesivo': 0
}

cv2.namedWindow('Inspetor de Adesivo V2', cv2.WINDOW_NORMAL)

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
    all_detections = []

    # Processar detecção apenas a cada N frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # Redimensionar display
        frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        # Redimensionar para inferência (menor = mais rápido)
        frame_inference = cv2.resize(frame, (INFERENCE_SIZE, INFERENCE_SIZE))

        # Predição otimizada para CPU
        results = model.predict(
            source=frame_inference,
            conf=confidence_threshold_geral,
            verbose=False,
            imgsz=INFERENCE_SIZE,
            half=False,
            device=device,
            agnostic_nms=True,  # NMS mais rápido
            max_det=10  # Limita detecções (acelera pós-processamento)
        )

        # Processar resultado - TODAS AS DETECÇÕES
        display_frame = frame_resized.copy()

        # Calcular fator de escala (inferência -> display)
        scale_x = WINDOW_WIDTH / INFERENCE_SIZE
        scale_y = WINDOW_HEIGHT / INFERENCE_SIZE

        for result in results:
            boxes = result.boxes

            if len(boxes) > 0:
                # Processar TODAS as detecções (não apenas a melhor)
                for box in boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    class_name = result.names[cls]
                    coords = box.xyxy[0].cpu().numpy()

                    # Escalar coordenadas para tamanho do display
                    coords_scaled = [
                        coords[0] * scale_x,
                        coords[1] * scale_y,
                        coords[2] * scale_x,
                        coords[3] * scale_y
                    ]

                    all_detections.append({
                        'class': class_name,
                        'conf': conf,
                        'coords': coords_scaled
                    })

        # Desenhar todas as detecções com máscaras
        for detection in all_detections:
            class_name = detection['class']
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

            # Cores por classe
            if class_name == 'com_adesivo':
                # Verde para COM ADESIVO
                if conf > 0.70:
                    color = (0, 255, 0)  # Verde forte
                elif conf > 0.50:
                    color = (0, 200, 100)  # Verde médio
                else:
                    color = (0, 150, 150)  # Verde fraco
            else:  # sem_adesivo
                # Laranja/Vermelho para SEM ADESIVO
                if conf > 0.70:
                    color = (0, 100, 255)  # Laranja forte
                elif conf > 0.50:
                    color = (0, 140, 255)  # Laranja médio
                else:
                    color = (0, 165, 200)  # Laranja fraco

            thickness = 3 if conf > 0.70 else 2

            # Desenhar máscara semi-transparente
            mask = display_frame.copy()
            cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)  # Retângulo preenchido
            cv2.addWeighted(mask, 0.4, display_frame, 0.6, 0, display_frame)

            # Desenhar borda
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

            # Label com nome da classe
            label = f"{class_name.upper()} {conf:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Adicionar à história com THRESHOLDS SEPARADOS
        if all_detections:
            # Separar detecções por classe
            sem_adesivo_detections = [d for d in all_detections if d['class'] == 'sem_adesivo']
            com_adesivo_detections = [d for d in all_detections if d['class'] == 'com_adesivo']

            # Verificar se cada classe atinge seu threshold específico
            sem_adesivo_valido = False
            com_adesivo_valido = False

            if sem_adesivo_detections:
                best_sem = max(sem_adesivo_detections, key=lambda x: x['conf'])
                sem_adesivo_valido = best_sem['conf'] >= threshold_sem_adesivo

            if com_adesivo_detections:
                best_com = max(com_adesivo_detections, key=lambda x: x['conf'])
                com_adesivo_valido = best_com['conf'] >= threshold_com_adesivo

            # LÓGICA DE DECISÃO COM PRIORIDADE
            if sem_adesivo_valido:
                # sem_adesivo tem PRIORIDADE se atingir seu threshold
                detection_history.append('sem_adesivo')
            elif com_adesivo_valido:
                # com_adesivo só é usado se sem_adesivo não atingir threshold
                detection_history.append('com_adesivo')
            else:
                # Nenhuma classe atingiu threshold mínimo
                detection_history.append(None)
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

    # Fundo semi-transparente
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 280), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, height - 100), (width, height), (0, 0, 0), -1)
    display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)

    # Título
    cv2.putText(display_frame, "INSPETOR DE ADESIVO V2",
               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Linha separadora
    cv2.line(display_frame, (20, 65), (width - 20, 65), (100, 100, 100), 2)

    # Status MUITO GRANDE
    y_pos = 130
    if stable_status == 'com_adesivo':
        status_text = "COM ADESIVO"
        status_color = (0, 255, 0)  # Verde
    elif stable_status == 'sem_adesivo':
        status_text = "SEM ADESIVO"
        status_color = (0, 100, 255)  # Laranja
    else:
        status_text = "AGUARDANDO..."
        status_color = (100, 100, 100)  # Cinza

    cv2.putText(display_frame, status_text,
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 2.0, status_color, 4)

    # Informações detalhadas
    y_pos = 180
    info_text = f"Estabilidade: {stability}"
    cv2.putText(display_frame, info_text,
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Contadores por classe
    y_pos += 35
    counter_text = f"COM: {com_adesivo_count}/{HISTORY_SIZE}  |  SEM: {sem_adesivo_count}/{HISTORY_SIZE}"
    cv2.putText(display_frame, counter_text,
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Confiança da melhor detecção
    if all_detections:
        best_det = max(all_detections, key=lambda x: x['conf'])
        y_pos += 35
        conf_text = f"Melhor deteccao: {best_det['class'].upper()} ({best_det['conf']:.1%})"
        cv2.putText(display_frame, conf_text,
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    y_pos += 35
    threshold_text = f"Thresholds: COM>={threshold_com_adesivo:.0%} | SEM>={threshold_sem_adesivo:.0%}"
    cv2.putText(display_frame, threshold_text,
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # FPS
    if len(fps_counter) > 1:
        fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
        cv2.putText(display_frame, f"{fps:.0f} FPS",
                   (width - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Legenda de cores no rodapé
    y_footer = height - 60
    cv2.putText(display_frame, "Legenda:",
               (20, y_footer), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Quadrado verde
    cv2.rectangle(display_frame, (120, y_footer - 15), (140, y_footer - 5), (0, 255, 0), -1)
    cv2.putText(display_frame, "COM ADESIVO",
               (145, y_footer - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Quadrado laranja
    cv2.rectangle(display_frame, (320, y_footer - 15), (340, y_footer - 5), (0, 100, 255), -1)
    cv2.putText(display_frame, "SEM ADESIVO",
               (345, y_footer - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

    # Instruções
    cv2.putText(display_frame, "Q=Sair | S=Salvar | 1/2=COM | 3/4=SEM",
               (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar
    cv2.imshow('Inspetor de Adesivo V2', display_frame)

    # Teclas
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("\n👋 Encerrando...")
        running = False
        break
    elif key == ord('s') or key == ord('S'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'captura_v2_{timestamp}.jpg'
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_count += 1
        print(f"📸 Captura salva: {filename}")
    # Tecla 1: AUMENTAR threshold com_adesivo
    elif key == ord('1'):
        threshold_com_adesivo = min(0.95, threshold_com_adesivo + 0.05)
        print(f"Threshold COM_ADESIVO: {threshold_com_adesivo:.0%}")
    # Tecla 2: DIMINUIR threshold com_adesivo
    elif key == ord('2'):
        threshold_com_adesivo = max(0.10, threshold_com_adesivo - 0.05)
        print(f"Threshold COM_ADESIVO: {threshold_com_adesivo:.0%}")
    # Tecla 3: AUMENTAR threshold sem_adesivo
    elif key == ord('3'):
        threshold_sem_adesivo = min(0.95, threshold_sem_adesivo + 0.05)
        print(f"Threshold SEM_ADESIVO: {threshold_sem_adesivo:.0%}")
    # Tecla 4: DIMINUIR threshold sem_adesivo
    elif key == ord('4'):
        threshold_sem_adesivo = max(0.05, threshold_sem_adesivo - 0.05)
        print(f"Threshold SEM_ADESIVO: {threshold_sem_adesivo:.0%}")

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
print(f"   Detecções COM ADESIVO: {class_counters['com_adesivo']}")
print(f"   Detecções SEM ADESIVO: {class_counters['sem_adesivo']}")
