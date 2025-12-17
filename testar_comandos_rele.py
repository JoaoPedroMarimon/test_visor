"""
🧪 TESTADOR DE COMANDOS DO RELÉ USBRelay2
Testa diferentes comandos para descobrir qual funciona
"""

try:
    import usb.core
    import usb.util
    USB_AVAILABLE = True
except ImportError:
    USB_AVAILABLE = False
    print("❌ PyUSB não instalado!")
    print("   Instale com: pip install pyusb")
    exit(1)

import time

print("="*70)
print("🧪 TESTADOR DE COMANDOS DO RELÉ")
print("="*70)

# Conectar ao relé
print("\n🔌 Conectando ao relé USBRelay2...")
dev = usb.core.find(idVendor=0x16c0, idProduct=0x5df)

if dev is None:
    print("❌ Relé não encontrado!")
    print("   Verifique se está conectado")
    exit(1)

print("✅ Relé encontrado!")

# Configurar dispositivo
try:
    dev.set_configuration()
except:
    pass

print("\n" + "="*70)
print("🧪 TESTANDO COMANDOS PARA CANAL 1")
print("="*70)

# Lista de comandos para testar
comandos_testar = [
    ("0xFF 0x01 0x01", [0xFF, 0x01, 0x01], [0xFF, 0x01, 0x00]),
    ("0xFE 0x01 0x01", [0xFE, 0x01, 0x01], [0xFE, 0x01, 0x00]),
    ("0xA0 0x01 0x01", [0xA0, 0x01, 0x01, 0xA2], [0xA0, 0x01, 0x00, 0xA1]),
    ("0x51 / 0x52", [0x51], [0x52]),
    ("Estado atual", [0xFE], None),  # Ler estado
]

print("\n⚠️  FIQUE ATENTO AO CLIQUE DO RELÉ!\n")

for i, (nome, cmd_on, cmd_off) in enumerate(comandos_testar):
    print(f"\n{'─'*70}")
    print(f"Teste {i+1}: {nome}")
    print(f"{'─'*70}")

    # Comando ON
    print(f"\n📤 Enviando LIGAR: {cmd_on}")
    try:
        # Método 1: Control Transfer (mais comum)
        try:
            data = bytes(cmd_on)
            dev.ctrl_transfer(
                0x21,  # bmRequestType
                0x09,  # bRequest (SET_REPORT)
                0x0300,  # wValue
                0,     # wIndex
                data   # Data
            )
            print("   ✓ Enviado via control_transfer")
        except Exception as e:
            print(f"   ✗ Control transfer falhou: {e}")

        # Aguardar
        print("   🔴 Aguardando 2 segundos... (OUÇA O CLIQUE!)")
        time.sleep(2)

        # Comando OFF (se existir)
        if cmd_off:
            print(f"\n📤 Enviando DESLIGAR: {cmd_off}")
            try:
                data = bytes(cmd_off)
                dev.ctrl_transfer(
                    0x21,
                    0x09,
                    0x0300,
                    0,
                    data
                )
                print("   ✓ Enviado via control_transfer")
            except Exception as e:
                print(f"   ✗ Control transfer falhou: {e}")

            print("   🟢 Aguardando 2 segundos...")
            time.sleep(2)

        # Perguntar se funcionou
        resposta = input("\n❓ Você ouviu o CLIQUE? (s/n): ").strip().lower()

        if resposta == 's':
            print("\n" + "="*70)
            print("🎉 COMANDO ENCONTRADO!")
            print("="*70)
            print(f"\n✅ Comandos que funcionam: {nome}")
            print(f"\n   LIGAR:  {cmd_on}")
            if cmd_off:
                print(f"   DESLIGAR: {cmd_off}")

            print("\n📝 ATUALIZE O testv3.py:")
            print("\nNa função relay_on (linha ~237), substitua por:")
            print(f"   data = bytes({cmd_on})")
            print("\nNa função relay_off (linha ~264), substitua por:")
            if cmd_off:
                print(f"   data = bytes({cmd_off})")

            break

    except Exception as e:
        print(f"\n❌ Erro: {e}")

else:
    print("\n" + "="*70)
    print("⚠️  NENHUM COMANDO FUNCIONOU")
    print("="*70)
    print("\n💡 Vamos tentar ler o estado atual do relé...")

    # Tentar ler estado
    print("\n🔍 Tentando ler estado do relé...")
    try:
        # Control Transfer IN (ler dados)
        data = dev.ctrl_transfer(
            0xA1,  # bmRequestType (Device to Host)
            0x01,  # bRequest (GET_REPORT)
            0x0300,  # wValue
            0,     # wIndex
            8      # Tamanho do buffer
        )
        print(f"✓ Estado lido: {list(data)}")
        print(f"  Em hex: {[hex(b) for b in data]}")

        print("\n💡 Tente usar estes valores para controlar o relé!")

    except Exception as e:
        print(f"❌ Erro ao ler estado: {e}")

print("\n" + "="*70)
print("✓ Teste concluído!")
print("="*70)

# Limpar
try:
    usb.util.dispose_resources(dev)
except:
    pass
