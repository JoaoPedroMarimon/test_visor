from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
import threading
import torch
import platform
from datetime import datetime

# Áudio MP3 via pygame
import os
try:
    import pygame
    pygame.mixer.init()
    SOUND_AVAILABLE = True
except Exception:
    SOUND_AVAILABLE = False
    print("⚠ pygame não instalado. Instale com: pip install pygame")

ALARM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alarme.mp3")

# Controle do relé USB HID
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

RTSP_URL = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=0"

# Configurações de Performance
WINDOW_WIDTH  = 800
WINDOW_HEIGHT = 600
PANEL_WIDTH   = 320
TOTAL_WIDTH   = WINDOW_WIDTH + PANEL_WIDTH
PROCESS_EVERY_N_FRAMES = 5
INFERENCE_SIZE = 640

MAX_FAILURES_SHOWN = 8
FLASH_PERIOD_FRAMES = 12   # período de piscar borda FAIL

# ===========================================
# TEMA VERDE
# ===========================================
C_BG_PANEL    = (8,  40,  8)    # fundo do painel
C_BG_HEADER   = (12, 65, 12)    # cabeçalhos
C_DIVIDER     = (20, 80, 20)    # linhas divisórias
C_TEXT_TITLE  = (80, 210, 80)   # título das seções
C_TEXT_BODY   = (170, 230, 170) # texto genérico
C_TEXT_DIM    = (60, 120, 60)   # texto apagado
C_ACCENT      = (0,  255,  0)   # verde brilhante (destaque)
C_CAM_BAR     = (0,  22,   0)   # barra top da câmera

# Cores de estado (mantidas por semântica)
C_PASS  = (0, 175, 0)
C_FAIL  = (0,  0, 210)
C_WAIT  = (30, 80, 30)

# ===========================================
# CONFIGURAÇÃO DO RELÉ USB HID
# ===========================================
RELAY_CONFIG = {
    'enabled': True,
    'canal_alarme': 1,
    'invert_logic': False,
    'vendor_id': 0x16c0,
    'product_id': 0x5df,
}

# ===========================================
# CLASSE RELÉ USB HID
# ===========================================
class RelayControllerHID:
    def __init__(self, config):
        self.config = config
        self.device = None
        self.connected = False
        self.system = platform.system()
        self.endpoint = None

        if not USB_AVAILABLE:
            print("❌ PyUSB não instalado!")
            return
        if not config['enabled']:
            print("⚠ Relé desabilitado")
            return
        self.connect()

    def find_relay_device(self):
        print("\n🔍 Buscando relés USB HID...")
        if self.config.get('vendor_id') and self.config.get('product_id'):
            vid, pid = self.config['vendor_id'], self.config['product_id']
            dev = usb.core.find(idVendor=vid, idProduct=pid)
            if dev is not None:
                print(f"✓ Encontrado: VID={hex(vid)}, PID={hex(pid)}")
                return dev, vid, pid

        for vid, pid in [(0x16c0, 0x05df), (0x1a86, 0x7523),
                         (0x5131, 0x2007), (0x0416, 0x5020)]:
            dev = usb.core.find(idVendor=vid, idProduct=pid)
            if dev is not None:
                print(f"✓ Encontrado: VID={hex(vid)}, PID={hex(pid)}")
                return dev, vid, pid

        print("\n📋 Dispositivos USB encontrados:")
        for i, dev in enumerate(usb.core.find(find_all=True)):
            try:
                vid, pid = dev.idVendor, dev.idProduct
                mfr = usb.util.get_string(dev, dev.iManufacturer) if dev.iManufacturer else ""
                prd = usb.util.get_string(dev, dev.iProduct) if dev.iProduct else ""
                print(f"   [{i+1}] VID={hex(vid)} PID={hex(pid)}  {mfr} {prd}")
                if any(kw in (prd+mfr).lower() for kw in ['relay','usb','module','hid']):
                    print(f"       ⭐ Possível relé!")
                    return dev, vid, pid
            except:
                continue
        return None, None, None

    def connect(self):
        print("\n🔌 Procurando relé USB...")
        dev, vid, pid = self.find_relay_device()
        if dev is None:
            print("❌ Nenhum relé USB encontrado!")
            return
        self.device = dev
        try:
            if self.system == "Linux":
                try:
                    if dev.is_kernel_driver_active(0):
                        dev.detach_kernel_driver(0)
                except:
                    pass
            try:
                dev.set_configuration()
            except usb.core.USBError:
                pass
            cfg = dev.get_active_configuration()
            intf = cfg[(0, 0)]
            self.endpoint = usb.util.find_descriptor(
                intf,
                custom_match=lambda e: usb.util.endpoint_direction(
                    e.bEndpointAddress) == usb.util.ENDPOINT_OUT
            )
            self.connected = True
            print(f"✅ Relé USB conectado! VID:{hex(vid)} PID:{hex(pid)}")
        except Exception as e:
            print(f"❌ Erro ao conectar: {e}")

    def send_command(self, data):
        if not self.connected or self.device is None:
            return False
        try:
            try:
                self.device.ctrl_transfer(0x21, 0x09, 0x0300, 0, data)
                return True
            except:
                pass
            if self.endpoint:
                try:
                    self.endpoint.write(data)
                    return True
                except:
                    pass
            try:
                self.device.write(1, data, 1000)
                return True
            except:
                pass
        except Exception as e:
            print(f"❌ Erro ao enviar comando: {e}")
        return False

    def relay_on(self, canal):
        if not self.connected or canal not in (1, 2):
            return False
        data = bytes([0xFF, canal, 0x01])
        if self.config['invert_logic']:
            data = bytes([0xFF, canal, 0x00])
        if self.send_command(data):
            print(f"🔴 RELÉ {canal} LIGADO")
            return True
        return False

    def relay_off(self, canal):
        if not self.connected or canal not in (1, 2):
            return False
        data = bytes([0xFC, canal, 0x00])
        if self.config['invert_logic']:
            data = bytes([0xFF, canal, 0x01])
        if self.send_command(data):
            print(f"🟢 RELÉ {canal} DESLIGADO")
            return True
        return False

    def all_off(self):
        self.relay_off(1)
        self.relay_off(2)

    def close(self):
        if self.device:
            self.all_off()
            try:
                usb.util.dispose_resources(self.device)
            except:
                pass
            print("✓ Relé desconectado")


# ===========================================
# ALARME SONORO (alarme.mp3 em loop)
# ===========================================
alarm_active = False


def start_alarm():
    global alarm_active
    if alarm_active:
        return
    if SOUND_AVAILABLE and os.path.exists(ALARM_FILE):
        try:
            pygame.mixer.music.load(ALARM_FILE)
            pygame.mixer.music.play(loops=-1)   # -1 = loop infinito
        except Exception as e:
            print(f"⚠ Erro ao tocar alarme: {e}")
    alarm_active = True
    print("🔊 ALARME ATIVADO")


def stop_alarm():
    global alarm_active, relay_state, relay_timer, relay_cooldown_timer
    if not alarm_active:
        return
    if SOUND_AVAILABLE:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
    alarm_active = False
    # Desliga relé ao parar o alarme
    if relay.connected and relay_state == 'ligado':
        canal = RELAY_CONFIG['canal_alarme']
        relay.relay_off(canal)
        relay_state          = 'desligado'
        relay_timer          = None
        relay_cooldown_timer = None
    print("🔇 Alarme parado pelo operador")


# ===========================================
# FUNÇÕES DE DESENHO
# ===========================================

def put_text_bg(img, text, pos, font, scale, color, thickness=1,
                bg=(0, 0, 0), pad=3):
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img,
                  (x - pad,      y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  bg, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness)


def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash=10):
    x1, y1 = pt1
    x2, y2 = pt2
    for ax, ay, bx, by in [(x1,y1,x2,y1),(x2,y1,x2,y2),(x2,y2,x1,y2),(x1,y2,x1,y1)]:
        dist = int(((bx-ax)**2 + (by-ay)**2)**0.5)
        if dist == 0:
            continue
        for i in range(0, dist, dash * 2):
            t0, t1 = i/dist, min((i+dash)/dist, 1.0)
            p0 = (int(ax + t0*(bx-ax)), int(ay + t0*(by-ay)))
            p1 = (int(ax + t1*(bx-ax)), int(ay + t1*(by-ay)))
            cv2.line(img, p0, p1, color, thickness)


def draw_stop_button_on_panel(panel, flash_counter):
    """Botão STOP desenhado no painel lateral (fundo do painel)."""
    ph, pw = panel.shape[:2]
    bh = 90
    by = ph - bh - 6
    bx = 6
    bw = pw - 12

    pulse     = (flash_counter // 6) % 2 == 0
    bg_color  = (0, 0, 200) if pulse else (0, 0, 130)
    brd_color = (0, 0, 255) if pulse else (60, 60, 200)

    # Sombra
    cv2.rectangle(panel, (bx+3, by+3), (bx+bw+3, by+bh+3), (0, 0, 0), -1)
    # Fundo
    cv2.rectangle(panel, (bx, by), (bx+bw, by+bh), bg_color, -1)
    # Borda
    cv2.rectangle(panel, (bx, by), (bx+bw, by+bh), brd_color, 3)

    # Texto STOP
    (tw, _), _ = cv2.getTextSize("STOP", cv2.FONT_HERSHEY_DUPLEX, 2.6, 3)
    tx = bx + (bw - tw) // 2
    cv2.putText(panel, "STOP", (tx+2, by+57), cv2.FONT_HERSHEY_DUPLEX, 2.6, (0,0,0), 5)
    cv2.putText(panel, "STOP", (tx,   by+55), cv2.FONT_HERSHEY_DUPLEX, 2.6, (255,255,255), 3)

    # Instrução
    hint = "[ESPACO] parar alarme"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_PLAIN, 1.0, 1)
    hx = bx + (bw - hw) // 2
    cv2.putText(panel, hint, (hx, by+bh-8), cv2.FONT_HERSHEY_PLAIN, 1.0, (220, 220, 100), 1)


def draw_right_panel(stable_status, prod_stats, recent_failures, flash_counter, relay_text, relay_color, alarm_on=False):
    panel = np.zeros((WINDOW_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = C_BG_PANEL

    y = 0

    # --- Cabeçalho ---
    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 38), C_BG_HEADER, -1)
    cv2.putText(panel, "INSPETOR V4", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_ACCENT, 2)
    # Status relé à direita do header
    cv2.putText(panel, relay_text, (PANEL_WIDTH - 105, 26),
                cv2.FONT_HERSHEY_PLAIN, 1.05, relay_color, 1)
    y = 44

    # --- Indicador PASS / FAIL ---
    ind_h = 108
    if stable_status == 'com_laranja':
        ind_bg   = C_PASS
        ind_text = "PASS"
        ind_sub  = "ADESIVO PRESENTE"
        ind_fg   = (210, 255, 210)
    elif stable_status == 'sem_laranja':
        blink    = (flash_counter // FLASH_PERIOD_FRAMES) % 2 == 0
        ind_bg   = (0, 0, 200) if blink else (0, 0, 130)
        ind_text = "FAIL"
        ind_sub  = "ADESIVO AUSENTE"
        ind_fg   = (255, 210, 210)
    else:
        ind_bg   = C_WAIT
        ind_text = "---"
        ind_sub  = "AGUARDANDO"
        ind_fg   = (120, 180, 120)

    cv2.rectangle(panel, (4, y), (PANEL_WIDTH-4, y+ind_h), ind_bg, -1)
    cv2.rectangle(panel, (4, y), (PANEL_WIDTH-4, y+ind_h), C_DIVIDER, 1)

    (tw, th), _ = cv2.getTextSize(ind_text, cv2.FONT_HERSHEY_DUPLEX, 2.8, 3)
    tx = (PANEL_WIDTH - tw) // 2
    cv2.putText(panel, ind_text, (tx+2, y+72), cv2.FONT_HERSHEY_DUPLEX, 2.8, (0,0,0), 5)
    cv2.putText(panel, ind_text, (tx,   y+70), cv2.FONT_HERSHEY_DUPLEX, 2.8, ind_fg, 3)
    (sw, _), _ = cv2.getTextSize(ind_sub, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)
    sx = (PANEL_WIDTH - sw) // 2
    cv2.putText(panel, ind_sub, (sx, y+100), cv2.FONT_HERSHEY_PLAIN, 1.2, ind_fg, 1)

    y += ind_h + 10
    cv2.line(panel, (4, y), (PANEL_WIDTH-4, y), C_DIVIDER, 1)
    y += 8

    # --- Contadores de Produção ---
    cv2.putText(panel, "PRODUCAO ATUAL", (8, y+14),
                cv2.FONT_HERSHEY_PLAIN, 1.05, C_TEXT_TITLE, 1)
    y += 22

    total = prod_stats['total']
    ok    = prod_stats['pass']
    fail  = prod_stats['fail']
    rate  = (fail / total * 100) if total > 0 else 0.0

    for line, color in [
        (f"Total:  {total:5d}", C_TEXT_BODY),
        (f"OK:     {ok:5d}",    (0, 215, 0)),
        (f"FAIL:   {fail:5d}",  (80, 80, 230)),
        (f"Erro:   {rate:5.1f}%", (0,165,255) if rate > 2 else (0, 215, 0)),
    ]:
        cv2.putText(panel, line, (10, y+17), cv2.FONT_HERSHEY_PLAIN, 1.3, color, 1)
        y += 22

    y += 6
    cv2.line(panel, (4, y), (PANEL_WIDTH-4, y), C_DIVIDER, 1)
    y += 8

    # --- Histórico de Falhas ---
    cv2.putText(panel, "ULTIMAS FALHAS", (8, y+14),
                cv2.FONT_HERSHEY_PLAIN, 1.05, C_TEXT_TITLE, 1)
    y += 22

    if not recent_failures:
        cv2.putText(panel, "Sem falhas registradas", (10, y+14),
                    cv2.FONT_HERSHEY_PLAIN, 0.95, C_TEXT_DIM, 1)
    else:
        cv2.putText(panel, "Hora       #ID    Motivo", (6, y+12),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, C_TEXT_DIM, 1)
        y += 16
        cv2.line(panel, (4, y), (PANEL_WIDTH-4, y), C_DIVIDER, 1)
        y += 4

        for failure in reversed(list(recent_failures)):
            if y > WINDOW_HEIGHT - 22:
                break
            cv2.putText(panel,
                        f"{failure['time']}  #{failure['id']:04d}",
                        (6, y+13), cv2.FONT_HERSHEY_PLAIN, 0.9, (80, 130, 230), 1)
            y += 15
            if y > WINDOW_HEIGHT - 8:
                break
            cv2.putText(panel,
                        f"  {failure['reason']}",
                        (6, y+12), cv2.FONT_HERSHEY_PLAIN, 0.85, C_TEXT_DIM, 1)
            y += 16

    # Botão STOP no rodapé do painel (apenas quando alarme ativo)
    if alarm_on:
        draw_stop_button_on_panel(panel, flash_counter)

    return panel


# ===========================================
# INICIALIZAÇÃO
# ===========================================

if torch.cuda.is_available():
    device = 0
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("⚠ GPU não detectada — usando CPU")

print("Carregando modelo...")
model = YOLO(r'C:\Users\jo063877\Desktop\test_visor\runs\detect\adesivo_detection\v3_duas_classes10\weights\best.pt')
model.fuse()
print("✓ Modelo carregado!")

relay = RelayControllerHID(RELAY_CONFIG)

print(f"\nConectando RTSP: {RTSP_URL.replace(CAMERA_PASS, '***')}")
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

connected = False
for _ in range(10):
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"✓ Conectado! {frame.shape[1]}x{frame.shape[0]}")
        connected = True
        break
    time.sleep(0.2)

if not connected:
    print("❌ Falha na conexão RTSP")
    exit(1)

print("\n" + "="*65)
print("  INSPETOR DE LARANJA V4  (com_laranja)")
print("="*65)
print("  Q=Sair | S=Salvar | ESPACO=Parar alarme | C=Zerar stats")
print("  R=Testar relé | 1/2=thr com_laranja +/- | 3/4=thr sem_laranja +/-")
print("="*65 + "\n")

# Thresholds por classe
threshold_com_laranja = 0.40
threshold_sem_laranja  = 0.60

FAIL_FRAMES_NEEDED = 10   # frames consecutivos de sem_laranja para confirmar FAIL
fail_frame_counter = 0

saved_count = 0
frame_count = 0

# Thread de captura
current_frame = None
frame_lock = threading.Lock()
running = True


def capture_thread_fn():
    global current_frame, running
    while running:
        ret, f = cap.read()
        if ret:
            with frame_lock:
                current_frame = f


threading.Thread(target=capture_thread_fn, daemon=True).start()

print("Aguardando frames...")
while current_frame is None:
    time.sleep(0.1)
print("✓ Recebendo frames!\n")

last_frame_id  = None
fps_counter    = deque(maxlen=30)

# Relé
relay_state          = None
relay_timer          = None
relay_cooldown_timer = None
RELAY_DELAY          = 0.1
RELAY_COOLDOWN       = 2.0

# Produção
prod_stats      = {'total': 0, 'pass': 0, 'fail': 0}
piece_id        = 0
recent_failures = deque(maxlen=MAX_FAILURES_SHOWN)
last_stable_status = 'com_laranja'

# Visual
flash_counter  = 0
all_detections = []
display_frame  = None

ROI = None   # ex: (100, 80, 700, 520) para ativar

cv2.namedWindow('Inspetor de Adesivo V4', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Inspetor de Adesivo V4', TOTAL_WIDTH, WINDOW_HEIGHT)

# ===========================================
# LOOP PRINCIPAL
# ===========================================
while running:
    with frame_lock:
        if current_frame is None:
            cv2.waitKey(1)
            continue
        frame_id = id(current_frame)
        if frame_id == last_frame_id:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                running = False
                break
            if key == ord(' '):
                stop_alarm()
            continue
        last_frame_id = frame_id
        frame = current_frame.copy()

    frame_count += 1
    flash_counter += 1
    current_time = time.time()
    fps_counter.append(current_time)

    # -----------------------------------------------
    # DETECÇÃO (a cada N frames)
    # -----------------------------------------------
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        frame_resized  = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT),
                                    interpolation=cv2.INTER_LINEAR)   # melhor qualidade no display
        frame_inference = cv2.resize(frame, (INFERENCE_SIZE, INFERENCE_SIZE),
                                     interpolation=cv2.INTER_NEAREST)  # rápido, só para o modelo

        results = model.predict(
            source=frame_inference,
            conf=min(threshold_com_laranja, threshold_sem_laranja),
            verbose=False,
            imgsz=INFERENCE_SIZE,
            half=False,
            device=device,
            agnostic_nms=True,
            iou=0.3,
            max_det=30
        )

        display_frame  = frame_resized.copy()
        scale_x = WINDOW_WIDTH  / INFERENCE_SIZE
        scale_y = WINDOW_HEIGHT / INFERENCE_SIZE

        all_detections = []
        for result in results:
            for box in result.boxes:
                cls  = int(box.cls)
                conf = float(box.conf)
                name = result.names[cls]
                c    = box.xyxy[0].cpu().numpy()
                all_detections.append({
                    'class': name, 'conf': conf,
                    'coords': [c[0]*scale_x, c[1]*scale_y,
                               c[2]*scale_x, c[3]*scale_y]
                })

        # ROI
        if ROI:
            draw_dashed_rect(display_frame, (ROI[0], ROI[1]), (ROI[2], ROI[3]),
                             (0, 220, 100), 2)
            put_text_bg(display_frame, "ROI", (ROI[0]+4, ROI[1]+16),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 220, 100), 1, (0,0,0))

        # Bounding boxes — filtra pelo threshold de cada classe
        visible_dets = [
            d for d in all_detections
            if (d['class'] == 'com_laranja' and d['conf'] >= threshold_com_laranja)
            or (d['class'] == 'sem_laranja'  and d['conf'] >= threshold_sem_laranja)
        ]
        for i, det in enumerate(visible_dets):
            name = det['class']
            conf = det['conf']
            x1, y1, x2, y2 = map(int, det['coords'])

            if name == 'com_laranja':
                color    = (0, 200, 255)   # laranja
                status_s = "OK"
            else:
                color    = (0, 50, 230)
                status_s = "AUSENTE"

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            put_text_bg(display_frame,
                        f"{conf:.0%}",
                        (x1+2, y1-6),
                        cv2.FONT_HERSHEY_PLAIN, 1.1, color, 1, (0,0,0), 3)

        # Padrão PASS — FAIL só após frames consecutivos de sem_laranja
        sem_dets = [d for d in all_detections
                    if d['class'] == 'sem_laranja' and d['conf'] >= threshold_sem_laranja]
        if sem_dets:
            fail_frame_counter += 1
            best_conf = max(d['conf'] for d in sem_dets)
            print(f"[SEM_ADESIVO] frame {fail_frame_counter}/{FAIL_FRAMES_NEEDED} | conf={best_conf:.0%} | status={'FAIL!' if fail_frame_counter >= FAIL_FRAMES_NEEDED else 'aguardando'}")
            if fail_frame_counter >= FAIL_FRAMES_NEEDED:
                stable_status = 'sem_laranja'
            else:
                stable_status = 'com_laranja'
        else:
            if fail_frame_counter > 0:
                print(f"[SEM_ADESIVO] contador zerado (era {fail_frame_counter})")
            fail_frame_counter = 0
            stable_status = 'com_laranja'

    else:
        # frame pulado (não é frame de inferência) — mantém status anterior
        stable_status = last_stable_status if last_stable_status is not None else 'com_laranja'
        if display_frame is None:
            fr = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT),
                            interpolation=cv2.INTER_LINEAR)
            display_frame = fr.copy()

    # -----------------------------------------------
    # CONTAGEM DE PEÇAS + ALARME
    # -----------------------------------------------
    if stable_status == 'sem_laranja' and last_stable_status == 'com_laranja':
        piece_id += 1
        prod_stats['total'] += 1
        prod_stats['fail'] += 1
        recent_failures.append({
            'id':     piece_id,
            'time':   datetime.now().strftime('%H:%M:%S'),
            'reason': 'Adesivo ausente'
        })
        print(f"❌ Peça #{piece_id:04d} - FAIL")
        start_alarm()
    elif stable_status == 'com_laranja' and last_stable_status == 'sem_laranja':
        piece_id += 1
        prod_stats['total'] += 1
        prod_stats['pass'] += 1
        print(f"✅ Peça #{piece_id:04d} - PASS")

    last_stable_status = stable_status

    # -----------------------------------------------
    # CONTROLE DO RELÉ
    # -----------------------------------------------
    if relay.connected:
        canal      = RELAY_CONFIG['canal_alarme']
        em_cooldown = (relay_cooldown_timer is not None and
                       (current_time - relay_cooldown_timer) < RELAY_COOLDOWN)

        if stable_status == 'sem_laranja' and relay_state != 'ligado':
            if relay.relay_on(canal):
                relay_state = 'ligado'
                relay_timer = None   # sem auto-desligar por timer

    # -----------------------------------------------
    # OVERLAY NA CÂMERA
    # -----------------------------------------------
    h, w = display_frame.shape[:2]

    # Borda piscante FAIL
    if stable_status == 'sem_laranja':
        if (flash_counter // FLASH_PERIOD_FRAMES) % 2 == 0:
            cv2.rectangle(display_frame, (0, 0), (w-1, h-1), (0, 0, 255), 8)

    # Barra top verde escura
    cv2.rectangle(display_frame, (0, 0), (w, 36), C_CAM_BAR, -1)
    cv2.rectangle(display_frame, (0, 36), (w, 37), C_DIVIDER, -1)

    # FPS
    if len(fps_counter) > 1:
        fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
        put_text_bg(display_frame, f"{fps:.0f} fps",
                    (w-72, 25), cv2.FONT_HERSHEY_PLAIN, 1.3,
                    C_ACCENT, 1, C_CAM_BAR)

    # Status relé na barra top
    if relay.connected:
        if relay_state == 'ligado' and relay_timer:
            t = max(0.0, RELAY_DELAY - (current_time - relay_timer))
            rtxt, rcol = f"RELE ON {t:.1f}s", (0, 0, 255)
        elif relay_cooldown_timer and (current_time - relay_cooldown_timer) < RELAY_COOLDOWN:
            t = max(0.0, RELAY_COOLDOWN - (current_time - relay_cooldown_timer))
            rtxt, rcol = f"RELE CD {t:.1f}s", (0, 165, 255)
        else:
            rtxt, rcol = "RELE OFF", C_ACCENT
    else:
        rtxt, rcol = "RELE N/A", C_TEXT_DIM

    cv2.putText(display_frame, rtxt, (8, 25),
                cv2.FONT_HERSHEY_PLAIN, 1.3, rcol, 1)



    # Thresholds (rodapé)
    put_text_bg(display_frame,
                f"thr com:{threshold_com_laranja:.0%}  sem:{threshold_sem_laranja:.0%}",
                (6, h-8), cv2.FONT_HERSHEY_PLAIN, 0.9, C_TEXT_DIM, 1, C_CAM_BAR)

    # -----------------------------------------------
    # PAINEL LATERAL
    # -----------------------------------------------
    panel = draw_right_panel(
        stable_status, prod_stats, recent_failures,
        flash_counter, rtxt, rcol,
        alarm_on=alarm_active
    )

    sep = np.full((WINDOW_HEIGHT, 2, 3), 20, dtype=np.uint8)
    combined = np.hstack([display_frame, sep, panel])
    cv2.imshow('Inspetor de Adesivo V4', combined)

    # -----------------------------------------------
    # TECLAS
    # -----------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        running = False
        break
    elif key == ord(' '):
        stop_alarm()
    elif key == ord('s') or key == ord('S'):
        ts  = time.strftime("%Y%m%d_%H%M%S")
        fn  = f'captura_v4_{ts}.jpg'
        cv2.imwrite(fn, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_count += 1
        print(f"📸 Salvo: {fn}")
    elif key == ord('c') or key == ord('C'):
        prod_stats = {'total': 0, 'pass': 0, 'fail': 0}
        piece_id   = 0
        recent_failures.clear()
        fail_frame_counter = 0
        last_stable_status = 'com_laranja'
        print("🔄 Estatísticas zeradas!")
    elif key == ord('r') or key == ord('R'):
        if relay.connected:
            canal = RELAY_CONFIG['canal_alarme']
            relay.relay_on(canal)
            time.sleep(1)
            relay.relay_off(canal)
            print("✓ Teste relé concluído!")
        else:
            print("⚠ Relé não conectado!")
    elif key == ord('1'):
        threshold_com_laranja = min(0.95, threshold_com_laranja + 0.05)
        print(f"Threshold com_laranja: {threshold_com_laranja:.0%}")
    elif key == ord('2'):
        threshold_com_laranja = max(0.05, threshold_com_laranja - 0.05)
        print(f"Threshold com_laranja: {threshold_com_laranja:.0%}")
    elif key == ord('3'):
        threshold_sem_laranja = min(0.95, threshold_sem_laranja + 0.05)
        print(f"Threshold sem_laranja: {threshold_sem_laranja:.0%}")
    elif key == ord('4'):
        threshold_sem_laranja = max(0.05, threshold_sem_laranja - 0.05)
        print(f"Threshold sem_laranja: {threshold_sem_laranja:.0%}")

# ===========================================
# ENCERRAMENTO
# ===========================================
running = False
stop_alarm()
time.sleep(0.2)

if relay.connected:
    relay.close()

cap.release()
cv2.destroyAllWindows()

print("\n✓ Programa encerrado")
if saved_count > 0:
    print(f"✓ {saved_count} captura(s) salva(s)")

total = prod_stats['total']
ok    = prod_stats['pass']
fail  = prod_stats['fail']
rate  = (fail / total * 100) if total > 0 else 0.0

print(f"\n{'='*50}")
print("ESTATÍSTICAS DA SESSÃO")
print(f"{'='*50}")
print(f"  Frames : {frame_count}")
print(f"  Total  : {total}")
print(f"  PASS   : {ok}")
print(f"  FAIL   : {fail}")
print(f"  Erro   : {rate:.2f}%")
if recent_failures:
    print(f"\n  Últimas falhas:")
    for f in list(recent_failures)[-5:]:
        print(f"    [{f['time']}] #{f['id']:04d} - {f['reason']}")
print(f"{'='*50}")
