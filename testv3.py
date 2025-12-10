from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
import threading
import torch
import platform
import glob

# Importar serial com tratamento de erro
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠ PySerial não instalado. Instale com: pip install pyserial")

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
PROCESS_EVERY_N_FRAMES = 4
INFERENCE_SIZE = 320

# ===========================================
# CONFIGURAÇÃO DO RELÉ USB
# ===========================================
RELAY_CONFIG = {
    'enabled': True,           # Ativar/desativar funcionalidade do relé
    'port': 'COM3',            # Porta COM detectada (COM3 ou COM4 no seu PC)
    'baudrate': 9600,          # Taxa de comunicação padrão
    'timeout': 1,              # Timeout de comunicação
    'canal_alarme': 1,         # Canal usado para "sem adesivo" (1 ou 2)
    'invert_logic': False,     # Inverter lógica ON/OFF (para relés NC - Normalmente Fechado)

    # Comandos do relé CH340 - TESTE ESSAS 3 OPÇÕES:
    # OPÇÃO 1: Formato hexadecimal comum (já ativado)
    'commands': {
        'relay1_on': b'\xA0\x01\x01\xA2',
        'relay1_off': b'\xA0\x01\x00\xA1',
        'relay2_on': b'\xA0\x02\x01\xA3',
        'relay2_off': b'\xA0\x02\x00\xA2',
    }

    # OPÇÃO 2: Se não funcionar, descomente abaixo e comente o acima:
    # 'commands': {
    #     'relay1_on': b'\xFF\x01\x01',
    #     'relay1_off': b'\xFF\x01\x00',
    #     'relay2_on': b'\xFF\x02\x01',
    #     'relay2_off': b'\xFF\x02\x00',
    # }

    # OPÇÃO 3: Relés CH340 com comandos simples:
    # 'commands': {
    #     'relay1_on': b'\x51',
    #     'relay1_off': b'\x52',
    #     'relay2_on': b'\x53',
    #     'relay2_off': b'\x54',
    # }
}

# ============================================
# COMANDOS ALTERNATIVOS PARA OUTROS MODELOS:
# ============================================
# Se os comandos acima não funcionarem, teste estes:
#
# 1) Relés com comandos ASCII (texto):
# 'commands': {
#     'relay1_on': b'RELAY1_ON\n',
#     'relay1_off': b'RELAY1_OFF\n',
#     'relay2_on': b'RELAY2_ON\n',
#     'relay2_off': b'RELAY2_OFF\n',
# }
#
# 2) Relés SainSmart/similares:
# 'commands': {
#     'relay1_on': b'\x51',
#     'relay1_off': b'\x52',
#     'relay2_on': b'\x53',
#     'relay2_off': b'\x54',
# }
#
# 3) Relés Numato Lab:
# 'commands': {
#     'relay1_on': b'relay on 0\n\r',
#     'relay1_off': b'relay off 0\n\r',
#     'relay2_on': b'relay on 1\n\r',
#     'relay2_off': b'relay off 1\n\r',
# }
#
# 4) Relés LCUS-1:
# 'commands': {
#     'relay1_on': b'\xFF\x01\x01',
#     'relay1_off': b'\xFF\x01\x00',
#     'relay2_on': b'\xFF\x02\x01',
#     'relay2_off': b'\xFF\x02\x00',
# }

# ===========================================
# CLASSE PARA CONTROLE DO RELÉ
# ===========================================
class RelayController:
    """Controla relé USB de forma cross-platform (Windows/Linux)"""

    def __init__(self, config):
        self.config = config
        self.serial_port = None
        self.connected = False
        self.system = platform.system()

        if not SERIAL_AVAILABLE:
            print("❌ PySerial não está instalado!")
            print("   Instale com: pip install pyserial")
            return

        if not config['enabled']:
            print("⚠ Relé desabilitado na configuração")
            return

        # Tentar conectar
        self.connect()

    def list_available_ports(self):
        """Lista todas as portas seriais disponíveis"""
        ports = []

        if self.system == "Windows":
            # Windows: COMx
            for i in range(1, 21):  # Testar COM1 até COM20
                try:
                    port = f"COM{i}"
                    s = serial.Serial(port)
                    s.close()
                    ports.append(port)
                except:
                    pass
        else:
            # Linux/Mac: /dev/ttyUSBx ou /dev/ttyACMx
            ports.extend(glob.glob('/dev/ttyUSB*'))
            ports.extend(glob.glob('/dev/ttyACM*'))

        # Usar pyserial para listar também
        try:
            for port in serial.tools.list_ports.comports():
                if port.device not in ports:
                    ports.append(port.device)
        except:
            pass

        return ports

    def connect(self):
        """Conecta ao relé USB"""
        port = self.config['port']

        # Se porta não especificada, tentar autodetectar
        if port is None:
            print("\n🔍 Buscando portas seriais disponíveis...")
            available_ports = self.list_available_ports()

            if not available_ports:
                print("❌ Nenhuma porta serial encontrada!")
                print("\n💡 DICAS:")
                print("   1. Conecte o relé USB")
                if self.system == "Windows":
                    print("   2. Verifique no Gerenciador de Dispositivos (Portas COM)")
                    print("   3. Execute no PowerShell: Get-PnpDevice -Class Ports")
                else:
                    print("   2. Execute no terminal: ls /dev/ttyUSB* /dev/ttyACM*")
                    print("   3. Verifique permissões: sudo usermod -a -G dialout $USER")
                print("   4. Configure manualmente: RELAY_CONFIG['port'] = 'COM3'  (ou '/dev/ttyUSB0')")
                return

            print(f"✓ Portas encontradas: {', '.join(available_ports)}")

            # Tentar conectar em cada porta
            for test_port in available_ports:
                print(f"   Testando {test_port}...", end=" ")
                try:
                    self.serial_port = serial.Serial(
                        port=test_port,
                        baudrate=self.config['baudrate'],
                        timeout=self.config['timeout']
                    )
                    time.sleep(0.5)  # Aguardar estabilização
                    print("✓ CONECTADO!")
                    self.connected = True
                    print(f"\n✅ Relé conectado em: {test_port}")
                    return
                except Exception as e:
                    print(f"✗ ({str(e)[:30]})")
                    continue

            print("\n⚠ Não foi possível conectar em nenhuma porta")
            print("   Configure manualmente: RELAY_CONFIG['port'] = 'COMX'")

        else:
            # Porta especificada manualmente
            print(f"\n🔌 Conectando ao relé em {port}...")
            try:
                self.serial_port = serial.Serial(
                    port=port,
                    baudrate=self.config['baudrate'],
                    timeout=self.config['timeout']
                )
                time.sleep(0.5)
                self.connected = True
                print(f"✅ Relé conectado com sucesso em {port}!")
            except Exception as e:
                print(f"❌ Erro ao conectar: {e}")
                print("\n💡 VERIFIQUE:")
                print(f"   1. A porta {port} está correta?")
                if self.system == "Windows":
                    print("   2. Veja no Gerenciador de Dispositivos")
                else:
                    print("   2. Você tem permissão? sudo usermod -a -G dialout $USER")
                print("   3. O relé está conectado?")

    def send_command(self, command_name):
        """Envia comando para o relé"""
        if not self.connected or self.serial_port is None:
            return False

        try:
            command = self.config['commands'].get(command_name)
            if command:
                self.serial_port.write(command)
                self.serial_port.flush()
                return True
        except Exception as e:
            print(f"❌ Erro ao enviar comando {command_name}: {e}")
            return False

        return False

    def relay_on(self, canal):
        """Liga um canal do relé"""
        if not self.connected:
            return

        # Inverter lógica se configurado (para relés NC)
        if self.config['invert_logic']:
            command = f'relay{canal}_off'
        else:
            command = f'relay{canal}_on'

        if self.send_command(command):
            status = "DESLIGADO" if self.config['invert_logic'] else "LIGADO"
            print(f"🔴 RELÉ {canal} {status}")

    def relay_off(self, canal):
        """Desliga um canal do relé"""
        if not self.connected:
            return

        # Inverter lógica se configurado (para relés NC)
        if self.config['invert_logic']:
            command = f'relay{canal}_on'
        else:
            command = f'relay{canal}_off'

        if self.send_command(command):
            status = "LIGADO" if self.config['invert_logic'] else "DESLIGADO"
            print(f"🟢 RELÉ {canal} {status}")

    def all_off(self):
        """Desliga todos os canais"""
        self.relay_off(1)
        self.relay_off(2)

    def close(self):
        """Fecha conexão serial"""
        if self.serial_port and self.serial_port.is_open:
            self.all_off()  # Desligar todos antes de fechar
            self.serial_port.close()
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
relay = RelayController(RELAY_CONFIG)

# Conectar via RTSP
print(f"\nConectando ao stream RTSP...")
print(f"  URL: {RTSP_URL.replace(CAMERA_PASS, '***')}")

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 15)

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
threshold_com_adesivo = 0.80
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
                    color = (0, 255, 0)
                elif conf > 0.50:
                    color = (0, 200, 100)
                else:
                    color = (0, 150, 150)
            else:  # sem_adesivo
                # Laranja/Vermelho para SEM ADESIVO
                if conf > 0.70:
                    color = (0, 100, 255)
                elif conf > 0.50:
                    color = (0, 140, 255)
                else:
                    color = (0, 165, 200)

            thickness = 3 if conf > 0.70 else 2

            # Desenhar máscara semi-transparente
            mask = display_frame.copy()
            cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)
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

    # ===========================================
    # CONTROLE AUTOMÁTICO DO RELÉ
    # ===========================================
    if relay.connected:
        canal = RELAY_CONFIG['canal_alarme']

        if stable_status == 'sem_adesivo':
            # SEM ADESIVO DETECTADO - LIGAR RELÉ
            if relay_state != 'sem_adesivo':
                relay.relay_on(canal)
                relay_state = 'sem_adesivo'
        else:
            # COM ADESIVO OU INSTÁVEL - DESLIGAR RELÉ
            if relay_state != 'com_adesivo':
                relay.relay_off(canal)
                relay_state = 'com_adesivo'

    # Interface
    height, width = display_frame.shape[:2]

    # Fundo semi-transparente
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 300), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, height - 100), (width, height), (0, 0, 0), -1)
    display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)

    # Título
    cv2.putText(display_frame, "INSPETOR DE ADESIVO V3 + RELE",
               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

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

    # Status do relé
    y_pos = 180
    if relay.connected:
        relay_status_text = f"RELE: {'LIGADO' if relay_state == 'sem_adesivo' else 'DESLIGADO'}"
        relay_status_color = (0, 0, 255) if relay_state == 'sem_adesivo' else (0, 255, 0)
    else:
        relay_status_text = "RELE: DESCONECTADO"
        relay_status_color = (100, 100, 100)

    cv2.putText(display_frame, relay_status_text,
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, relay_status_color, 2)

    # Informações detalhadas
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
