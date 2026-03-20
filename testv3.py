from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
import threading
import torch
import platform

# Importar bibliotecas para controle do relé USB HID
try:
    import usb.core
    import usb.util
    USB_AVAILABLE = True
except ImportError:
    USB_AVAILABLE = False
    print("⚠ PyUSB não instalado. Instale com: pip install pyusb")

# ===========================================
# CONFIGURAÇÃO DA CÂMERA IP
# ===========================================
CAMERA_IP = "192.168.1.108"
CAMERA_USER = "admin"
CAMERA_PASS = "admin123"

# URL RTSP (Dahua)
RTSP_URL = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=0"

# Configurações de Performance (ajustadas para CPU)
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
PROCESS_EVERY_N_FRAMES = 5  # Aumentado de 4 para 5 (menos inferências)
INFERENCE_SIZE = 320
FAST_DISPLAY = True  # Modo rápido: menos overlays, mais FPS

# ===========================================
# CONFIGURAÇÃO DO RELÉ USB HID (2 CANAIS)
# ===========================================
RELAY_CONFIG = {
    'enabled': True,           # Ativar/desativar funcionalidade do relé
    'canal_alarme': 1,         # Canal usado para "sem adesivo" (1 ou 2)
    'invert_logic': False,     # Inverter lógica ON/OFF (para relés NC)

    # IDs USB do seu relé (detectado automaticamente)
    'vendor_id': 0x16c0,       # www.dcttech.com
    'product_id': 0x5df,       # USBRelay2
}

# ===========================================
# CLASSE PARA CONTROLE DO RELÉ USB HID
# ===========================================
class RelayControllerHID:
    """Controla relé USB HID de 2 canais (Windows/Linux)"""

    def __init__(self, config):
        self.config = config
        self.device = None
        self.connected = False
        self.system = platform.system()
        self.endpoint = None

        if not USB_AVAILABLE:
            print("❌ PyUSB não está instalado!")
            print("   Windows: pip install pyusb")
            print("   Linux: pip install pyusb && sudo apt-get install libusb-1.0-0")
            return

        if not config['enabled']:
            print("⚠ Relé desabilitado na configuração")
            return

        # Tentar conectar
        self.connect()

    def find_relay_device(self):
        """Busca dispositivos USB que podem ser relés"""
        print("\n🔍 Buscando relés USB HID...")

        # Tentar IDs configurados primeiro
        if self.config.get('vendor_id') and self.config.get('product_id'):
            vid = self.config['vendor_id']
            pid = self.config['product_id']
            dev = usb.core.find(idVendor=vid, idProduct=pid)
            if dev is not None:
                print(f"✓ Encontrado (configurado): VID={hex(vid)}, PID={hex(pid)}")
                return dev, vid, pid

        # IDs comuns de relés USB de 2 canais
        known_relay_ids = [
            (0x16c0, 0x05df),  # USBRelay (mais comum) - SEU RELÉ!
            (0x1a86, 0x7523),  # CH340 HID
            (0x5131, 0x2007),  # Relay module 2 canais
            (0x0416, 0x5020),  # Relay board
        ]

        # Tentar IDs conhecidos
        for vid, pid in known_relay_ids:
            dev = usb.core.find(idVendor=vid, idProduct=pid)
            if dev is not None:
                print(f"✓ Encontrado: VID={hex(vid)}, PID={hex(pid)}")
                return dev, vid, pid

        # Se não encontrou, listar todos os dispositivos USB
        print("\n📋 Dispositivos USB encontrados:")
        devices = list(usb.core.find(find_all=True))

        for i, dev in enumerate(devices):
            try:
                vid = dev.idVendor
                pid = dev.idProduct
                manufacturer = usb.util.get_string(dev, dev.iManufacturer) if dev.iManufacturer else "Desconhecido"
                product = usb.util.get_string(dev, dev.iProduct) if dev.iProduct else "Desconhecido"

                print(f"\n   [{i+1}] VID={hex(vid)}, PID={hex(pid)}")
                print(f"       Fabricante: {manufacturer}")
                print(f"       Produto: {product}")

                # Verificar se parece ser um relé (por keywords)
                keywords = ['relay', 'usb', 'module', 'hid']
                product_lower = str(product).lower()
                manufacturer_lower = str(manufacturer).lower()

                if any(kw in product_lower or kw in manufacturer_lower for kw in keywords):
                    print(f"       ⭐ Possível relé detectado!")
                    return dev, vid, pid
            except:
                continue

        # Se não encontrou nada, retornar None
        return None, None, None

    def connect(self):
        """Conecta ao relé USB HID"""
        print(f"\n🔌 Procurando relé USB...")

        # Buscar dispositivo
        dev, vid, pid = self.find_relay_device()

        if dev is None:
            print("\n❌ Nenhum relé USB encontrado!")
            print("\n💡 DICAS:")
            print("   1. Conecte o relé USB")
            print("   2. Windows: Instale o driver libusb")
            print("      - Baixe Zadig: https://zadig.akeo.ie/")
            print("      - Execute Zadig > Options > List All Devices")
            print("      - Selecione seu relé > Instale driver WinUSB ou libusb-win32")
            print("   3. Linux: Execute com sudo ou configure permissões:")
            print("      sudo chmod 666 /dev/bus/usb/xxx/xxx")
            return

        self.device = dev

        try:
            # Tentar detach do kernel driver (Linux)
            if self.system == "Linux":
                try:
                    if dev.is_kernel_driver_active(0):
                        dev.detach_kernel_driver(0)
                        print("✓ Kernel driver removido")
                except:
                    pass

            # Configurar dispositivo
            try:
                dev.set_configuration()
            except usb.core.USBError:
                pass  # Já pode estar configurado

            # Obter configuração
            cfg = dev.get_active_configuration()
            intf = cfg[(0, 0)]

            # Encontrar endpoint OUT
            self.endpoint = usb.util.find_descriptor(
                intf,
                custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT
            )

            self.connected = True
            print(f"\n✅ Relé USB conectado!")
            print(f"   VID: {hex(vid)}")
            print(f"   PID: {hex(pid)}")

        except Exception as e:
            print(f"\n❌ Erro ao conectar: {e}")
            print("\n💡 SOLUÇÕES:")
            if self.system == "Windows":
                print("   Windows: Use Zadig para instalar driver libusb-win32 ou WinUSB")
                print("   Download: https://zadig.akeo.ie/")
            else:
                print("   Linux: Execute com sudo ou configure permissões USB:")
                print("   sudo chmod 666 /dev/bus/usb/*/* (temporário)")
                print("   ou crie regra udev (permanente)")

    def send_command(self, data):
        """Envia comando para o relé via USB"""
        if not self.connected or self.device is None:
            return False

        try:
            # Relés USB HID geralmente usam control transfer ou bulk transfer
            # Vamos tentar os métodos mais comuns

            # Método 1: Control Transfer (mais comum)
            try:
                self.device.ctrl_transfer(
                    0x21,  # bmRequestType (Host to Device, Class, Interface)
                    0x09,  # bRequest (SET_REPORT)
                    0x0300,  # wValue (Report Type: Feature)
                    0,     # wIndex (Interface 0)
                    data   # Data
                )
                return True
            except:
                pass

            # Método 2: Bulk Transfer
            if self.endpoint:
                try:
                    self.endpoint.write(data)
                    return True
                except:
                    pass

            # Método 3: Interrupt Transfer
            try:
                self.device.write(1, data, 1000)  # Endpoint 1, timeout 1000ms
                return True
            except:
                pass

            return False

        except Exception as e:
            print(f"❌ Erro ao enviar comando: {e}")
            return False

    def relay_on(self, canal):
        """Liga um canal do relé"""
        if not self.connected:
            return False

        # Comandos USBRelay2: 0xFF para ligar
        if canal == 1:
            data = bytes([0xFF, 0x01, 0x01])  # [255, 1, 1]
        elif canal == 2:
            data = bytes([0xFF, 0x02, 0x01])  # [255, 2, 1]
        else:
            return False

        # Inverter lógica se configurado (para relés NC)
        if self.config['invert_logic']:
            data = bytes([0xFF, canal, 0x00])

        try:
            if self.send_command(data):
                status = "DESLIGADO" if self.config['invert_logic'] else "LIGADO"
                print(f"🔴 RELÉ {canal} {status}")
                return True
        except Exception as e:
            print(f"❌ Erro ao ligar relé: {e}")
        return False

    def relay_off(self, canal):
        """Desliga um canal do relé"""
        if not self.connected:
            return False

        # Comando correto para USBRelay2: 0xFC para desligar
        if canal == 1:
            data = bytes([0xFC, 0x01, 0x00])  # [252, 1, 0]
        elif canal == 2:
            data = bytes([0xFC, 0x02, 0x00])  # [252, 2, 0]
        else:
            return False

        # Inverter lógica se configurado (para relés NC)
        if self.config['invert_logic']:
            data = bytes([0xFF, canal, 0x01])

        try:
            if self.send_command(data):
                status = "LIGADO" if self.config['invert_logic'] else "DESLIGADO"
                print(f"🟢 RELÉ {canal} {status}")
                return True
        except Exception as e:
            print(f"❌ Erro ao desligar relé: {e}")
        return False

    def all_off(self):
        """Desliga todos os canais"""
        self.relay_off(1)
        self.relay_off(2)

    def close(self):
        """Fecha conexão USB"""
        if self.device:
            self.all_off()
            try:
                usb.util.dispose_resources(self.device)
            except:
                pass
            print("✓ Relé desconectado")

# ===========================================
# INICIALIZAÇÃO
# ===========================================

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
model.fuse()
print("✓ Modelo V2 carregado e otimizado!")

# Inicializar controle do relé
relay = RelayControllerHID(RELAY_CONFIG)

# Conectar via RTSP
print(f"\nConectando ao stream RTSP...")
print(f"  URL: {RTSP_URL.replace(CAMERA_PASS, '***')}")

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
cap.set(cv2.CAP_PROP_FPS, 30)  # Solicitar mais FPS da câmera

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
print("INSPETOR DE ADESIVO V3 - COM CONTROLE DE RELÉ")
print("="*60)
print("📹 Sistema iniciado!")
print("🎨 Classes: COM_ADESIVO (Verde) | SEM_ADESIVO (Laranja)")
print(f"⚡ Otimizações: Inferência {INFERENCE_SIZE}x{INFERENCE_SIZE} | Skip {PROCESS_EVERY_N_FRAMES} frames")
if relay.connected:
    print(f"🔌 Relé: ATIVO (Canal {RELAY_CONFIG['canal_alarme']} para alarme)")
else:
    print("🔌 Relé: DESCONECTADO (modo visualização)")
print("⌨️  Q - Sair | S - Salvar | 1/2 - COM | 3/4 - SEM | R - Testar Relé")
print("="*60 + "\n")

# Configurações - THRESHOLDS SEPARADOS POR CLASSE
threshold_com_adesivo = 0.70
threshold_sem_adesivo = 0.80
confidence_threshold_geral = 0.7

detection_history = []
HISTORY_SIZE = 10
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

# Contadores de detecções por classe
class_counters = {
    'com_adesivo': 0,
    'sem_adesivo': 0
}

# Estado anterior do relé (para evitar comandos repetidos)
relay_state = None
relay_timer = None  # Timer para desligar após 0.1 segundos
relay_cooldown_timer = None  # Timer de cooldown - só pode ativar de novo após 3 segundos
RELAY_DELAY = 0.1   # Tempo em segundos para manter relé ligado
RELAY_COOLDOWN = 2.0  # Tempo de espera antes de poder ativar novamente

cv2.namedWindow('Inspetor de Adesivo V3', cv2.WINDOW_NORMAL)

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
        # Redimensionar display (INTER_NEAREST é 3x mais rápido)
        frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST)

        # Redimensionar para inferência (menor = mais rápido)
        frame_inference = cv2.resize(frame, (INFERENCE_SIZE, INFERENCE_SIZE), interpolation=cv2.INTER_NEAREST)

        # Predição otimizada para CPU
        results = model.predict(
            source=frame_inference,
            conf=confidence_threshold_geral,
            verbose=False,
            imgsz=INFERENCE_SIZE,
            half=False,
            device=device,
            agnostic_nms=True,
            max_det=10
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

        # Desenhar todas as detecções
        for detection in all_detections:
            class_name = detection['class']
            conf = detection['conf']
            x1, y1, x2, y2 = map(int, detection['coords'])

            # Cores por classe (simplificado)
            if class_name == 'com_adesivo':
                color = (0, 255, 0)  # Verde
            else:
                color = (0, 100, 255)  # Laranja

            thickness = 2

            # Apenas borda (sem máscara semi-transparente - muito mais rápido)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

            # Label simplificado
            label = f"{class_name[:3].upper()} {conf:.0%}"
            cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                      cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

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
                detection_history.append('sem_adesivo')
            elif com_adesivo_valido:
                detection_history.append('com_adesivo')
            else:
                detection_history.append(None)
        else:
            detection_history.append(None)

        if len(detection_history) > HISTORY_SIZE:
            detection_history.pop(0)
    else:
        # Usar último display_frame se não processar
        if 'display_frame' not in locals():
            frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST)
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

    # ===========================================
    # CONTROLE AUTOMÁTICO DO RELÉ COM TIMER E COOLDOWN
    # ===========================================
    if relay.connected:
        canal = RELAY_CONFIG['canal_alarme']
        current_time = time.time()

        # Verificar se está em cooldown
        em_cooldown = False
        if relay_cooldown_timer is not None:
            tempo_cooldown = current_time - relay_cooldown_timer
            if tempo_cooldown < RELAY_COOLDOWN:
                em_cooldown = True

        if stable_status == 'sem_adesivo':
            # SEM ADESIVO DETECTADO - LIGAR RELÉ (se não estiver em cooldown)
            if relay_state != 'ligado' and not em_cooldown:
                print("\n⚠️  SEM ADESIVO DETECTADO - ACIONANDO RELÉ!")
                if relay.relay_on(canal):
                    relay_state = 'ligado'
                    relay_timer = current_time  # Marcar quando ligou
                    print(f"⏱️  Relé permanecerá ligado por {RELAY_DELAY} segundos")

        # Verificar se precisa desligar após o timer
        if relay_state == 'ligado' and relay_timer is not None:
            tempo_decorrido = current_time - relay_timer

            # Desligar após RELAY_DELAY segundos
            if tempo_decorrido >= RELAY_DELAY:
                print(f"\n⏱️  {RELAY_DELAY} segundos decorridos - DESLIGANDO RELÉ")
                if relay.relay_off(canal):
                    relay_state = 'desligado'
                    relay_timer = None
                    relay_cooldown_timer = current_time  # Iniciar cooldown
                    print(f"🕐 Cooldown de {RELAY_COOLDOWN} segundos iniciado")

    # Interface
    height, width = display_frame.shape[:2]

    if not FAST_DISPLAY:
        # Modo detalhado (mais lento)
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 300), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, height - 100), (width, height), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)

        # Título
        cv2.putText(display_frame, "INSPETOR DE ADESIVO V3 + RELE",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    else:
        # Modo rápido: fundo sólido simples e menor
        cv2.rectangle(display_frame, (0, 0), (width, 100), (0, 0, 0), -1)

    # Status principal
    if stable_status == 'com_adesivo':
        status_text = "COM" if FAST_DISPLAY else "COM ADESIVO"
        status_color = (0, 255, 0)  # Verde
    elif stable_status == 'sem_adesivo':
        status_text = "SEM" if FAST_DISPLAY else "SEM ADESIVO"
        status_color = (0, 100, 255)  # Laranja
    else:
        status_text = "..." if FAST_DISPLAY else "AGUARDANDO..."
        status_color = (100, 100, 100)  # Cinza

    y_pos = 30 if FAST_DISPLAY else 130
    font = cv2.FONT_HERSHEY_PLAIN if FAST_DISPLAY else cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5 if FAST_DISPLAY else 2.0
    thickness = 2 if FAST_DISPLAY else 4
    cv2.putText(display_frame, status_text,
               (10, y_pos), font, font_scale, status_color, thickness)

    # Status do relé com timer e cooldown
    y_pos = 55 if FAST_DISPLAY else 180
    if relay.connected:
        if relay_state == 'ligado' and relay_timer is not None:
            tempo_restante = RELAY_DELAY - (time.time() - relay_timer)
            if tempo_restante > 0:
                relay_status_text = f"R:ON {tempo_restante:.1f}s" if FAST_DISPLAY else f"RELE: LIGADO ({tempo_restante:.1f}s)"
                relay_status_color = (0, 0, 255)  # Vermelho
            else:
                relay_status_text = "R:ON" if FAST_DISPLAY else "RELE: LIGADO"
                relay_status_color = (0, 0, 255)
        elif relay_cooldown_timer is not None:
            tempo_cooldown = time.time() - relay_cooldown_timer
            if tempo_cooldown < RELAY_COOLDOWN:
                tempo_restante_cooldown = RELAY_COOLDOWN - tempo_cooldown
                relay_status_text = f"R:CD {tempo_restante_cooldown:.1f}s" if FAST_DISPLAY else f"RELE: COOLDOWN ({tempo_restante_cooldown:.1f}s)"
                relay_status_color = (0, 165, 255)  # Laranja
            else:
                relay_status_text = "R:OFF" if FAST_DISPLAY else "RELE: DESLIGADO"
                relay_status_color = (0, 255, 0)  # Verde
        else:
            relay_status_text = "R:OFF" if FAST_DISPLAY else "RELE: DESLIGADO"
            relay_status_color = (0, 255, 0)  # Verde
    else:
        relay_status_text = "R:N/A" if FAST_DISPLAY else "RELE: DESCONECTADO"
        relay_status_color = (100, 100, 100)

    font = cv2.FONT_HERSHEY_PLAIN
    text_scale = 1.5 if FAST_DISPLAY else 0.8
    cv2.putText(display_frame, relay_status_text,
               (10, y_pos), font, text_scale, relay_status_color, 1 if FAST_DISPLAY else 2)

    if not FAST_DISPLAY:
        # Informações detalhadas (apenas modo lento)
        y_pos += 35
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
    else:
        # Modo rápido: apenas contador e FPS
        y_pos = 100
        counter_text = f"COM:{com_adesivo_count} SEM:{sem_adesivo_count}"
        cv2.putText(display_frame, counter_text,
                   (20, y_pos), cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)

    # FPS
    if len(fps_counter) > 1:
        fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
        fps_pos_x = width - 80 if FAST_DISPLAY else width - 120
        fps_pos_y = 25 if FAST_DISPLAY else 40
        fps_font = cv2.FONT_HERSHEY_PLAIN if FAST_DISPLAY else cv2.FONT_HERSHEY_SIMPLEX
        fps_scale = 1.5 if FAST_DISPLAY else 0.9
        cv2.putText(display_frame, f"{fps:.0f}fps",
                   (fps_pos_x, fps_pos_y), fps_font, fps_scale, (0, 255, 255), 2)

    if not FAST_DISPLAY:
        # Legenda de cores no rodapé (apenas modo lento)
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
        cv2.putText(display_frame, "Q=Sair | S=Salvar | 1/2=COM | 3/4=SEM | R=Testar Rele",
                   (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Mostrar
    cv2.imshow('Inspetor de Adesivo V3', display_frame)

    # Teclas
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("\n👋 Encerrando...")
        running = False
        break
    elif key == ord('s') or key == ord('S'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'captura_v3_{timestamp}.jpg'
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_count += 1
        print(f"📸 Captura salva: {filename}")
    # Tecla R: TESTAR RELÉ
    elif key == ord('r') or key == ord('R'):
        if relay.connected:
            print("\n🔧 TESTANDO RELÉ...")
            canal = RELAY_CONFIG['canal_alarme']
            print(f"   Ligando canal {canal}...")
            relay.relay_on(canal)
            time.sleep(1)
            print(f"   Desligando canal {canal}...")
            relay.relay_off(canal)
            print("✓ Teste concluído!")
        else:
            print("⚠ Relé não conectado!")
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

# Desligar relé ao encerrar
if relay.connected:
    relay.close()

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
