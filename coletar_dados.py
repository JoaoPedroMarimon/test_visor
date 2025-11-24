import cv2
import numpy as np
import time
import os
import threading
from collections import deque
import json

# ===========================================
# CONFIGURAÇÃO DA CÂMERA IP
# ===========================================
CAMERA_IP = "192.168.1.108"
CAMERA_USER = "admin"
CAMERA_PASS = "admin123"

# URL RTSP (Dahua)
RTSP_URL = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=0"

# Configurações
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
ULTRA_FAST_MODE = True  # Modo ultra-rápido: menos overlays, mais FPS

# Criar pastas
os.makedirs('novos_dados_ip/com_adesivo', exist_ok=True)
os.makedirs('novos_dados_ip/sem_adesivo', exist_ok=True)

# Arquivo para salvar contadores
COUNTER_FILE = 'coleta_contadores.json'

def carregar_contador():
    """Carrega contador global do arquivo"""
    if os.path.exists(COUNTER_FILE):
        try:
            with open(COUNTER_FILE, 'r') as f:
                data = json.load(f)
                return data.get('contador_global', 0)
        except:
            pass
    return 0

def salvar_contador(contador):
    """Salva contador global no arquivo"""
    with open(COUNTER_FILE, 'w') as f:
        json.dump({'contador_global': contador}, f)

print("="*60)
print("COLETAR DADOS - VERSÃO RTSP (RÁPIDA)")
print("="*60)

# Tentar conectar via RTSP
def test_rtsp_connection():
    """Testa conexão RTSP"""
    print("\nConectando ao stream RTSP...")
    print(f"  URL: {RTSP_URL.replace(CAMERA_PASS, '***')}")

    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para baixa latência

    # Tentar ler um frame
    for _ in range(10):  # Tentar até 10 vezes
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✓ Conectado! Resolução: {frame.shape[1]}x{frame.shape[0]}")
            return cap
        time.sleep(0.2)

    cap.release()
    print("❌ Não foi possível conectar ao RTSP")
    return None

cap = test_rtsp_connection()

if cap is None:
    print("\n⚠ DICA: Verifique se RTSP está habilitado na câmera")
    print("   Ou tente usar o OpenCV diretamente:")
    print(f"   cap = cv2.VideoCapture('http://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}/video')")
    exit(1)

# Otimizações do VideoCapture
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer de 1 frame (baixa latência)
cap.set(cv2.CAP_PROP_FPS, 30)  # Solicitar 30 FPS

print("\n" + "="*60)
print("COLETA INICIADA")
print("="*60)
print("C=COM | S=SEM | Q=SAIR")
print("="*60 + "\n")

# Carregar contador global
contador_global = carregar_contador()
print(f"📊 Continuando da imagem {contador_global + 1}\n")

# Variáveis
last_save_time = 0
fps_counter = deque(maxlen=30)
running = True

# Thread de captura (não bloqueia interface)
current_frame = None
frame_lock = threading.Lock()

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

# Configurar janela
cv2.namedWindow('Coleta RTSP', cv2.WINDOW_NORMAL)

# Pré-computar overlay
overlay_base = None
cached_width = None
cached_height = None
last_frame_id = None

# Loop principal
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

    # Redimensionar para display
    display = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST)
    height, width = display.shape[:2]

    current_time = time.time()

    if ULTRA_FAST_MODE:
        # Modo ultra-rápido: apenas texto essencial
        info = f"Total: {contador_global} imagens | Q=Sair"
        cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        # FPS real
        fps_counter.append(current_time)
        if len(fps_counter) > 1:
            fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
            cv2.putText(display, f"{fps:.0f}fps", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
    else:
        # Modo completo com overlays
        if overlay_base is None or cached_width != width or cached_height != height:
            overlay_base = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(overlay_base, (0, 0), (width, 80), (0, 0, 0), -1)
            cv2.rectangle(overlay_base, (0, height - 30), (width, height), (0, 0, 0), -1)
            cached_width = width
            cached_height = height

        cv2.add(display, overlay_base, display)

        info_text = f"Total: {contador_global} imagens coletadas"
        cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        fps_counter.append(current_time)
        if len(fps_counter) > 1:
            fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
            cv2.putText(display, f"FPS:{fps:.0f}", (10, 55), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

        cv2.putText(display, "C=COM S=SEM Q=SAIR",
                   (10, height - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        cv2.circle(display, (width - 15, 15), 5, (0, 255, 0), -1)

    # Mostrar
    cv2.imshow('Coleta RTSP', display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("\n👋 Encerrando...")
        running = False
        break

    elif key == ord('c') or key == ord('C'):
        if current_time - last_save_time > 0.15:
            contador_global += 1
            filename = f'novos_dados_ip/com_adesivo/imagem_{contador_global}.jpg'
            # Salvar frame ORIGINAL em alta resolução
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            salvar_contador(contador_global)  # Salvar após cada captura
            last_save_time = current_time
            print(f"✓ COM [imagem_{contador_global}]")

    elif key == ord('s') or key == ord('S'):
        if current_time - last_save_time > 0.15:
            contador_global += 1
            filename = f'novos_dados_ip/sem_adesivo/imagem_{contador_global}.jpg'
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            salvar_contador(contador_global)  # Salvar após cada captura
            last_save_time = current_time
            print(f"✓ SEM [imagem_{contador_global}]")

# Limpar
running = False
time.sleep(0.2)  # Dar tempo para thread terminar
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("✓ FINALIZADO!")
print(f"  Total de imagens coletadas: {contador_global}")
print("="*60)
