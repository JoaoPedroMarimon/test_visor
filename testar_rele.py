"""
🔧 SCRIPT DE TESTE DE RELÉ USB
Testa diferentes comandos para descobrir qual funciona no seu relé CH340
"""

import serial
import time

# =====================================================
# CONFIGURAÇÃO
# =====================================================
PORTA = 'COM4'  # Testando COM4 (você tem COM3 e COM4 disponíveis)
BAUDRATE = 9600

# =====================================================
# COMANDOS PARA TESTAR
# =====================================================
TESTES = {
    "Teste 1 (Hexadecimal padrão)": {
        'on': b'\xA0\x01\x01\xA2',
        'off': b'\xA0\x01\x00\xA1'
    },
    "Teste 2 (LCUS-1 / CH340 comum)": {
        'on': b'\xFF\x01\x01',
        'off': b'\xFF\x01\x00'
    },
    "Teste 3 (SainSmart)": {
        'on': b'\x51',
        'off': b'\x52'
    },
    "Teste 4 (ASCII)": {
        'on': b'RELAY1_ON\n',
        'off': b'RELAY1_OFF\n'
    },
    "Teste 5 (Numato Lab)": {
        'on': b'relay on 0\n\r',
        'off': b'relay off 0\n\r'
    },
}

# =====================================================
# FUNÇÃO PRINCIPAL
# =====================================================
def testar_rele():
    print("="*60)
    print("🔧 TESTADOR DE RELÉ USB")
    print("="*60)
    print(f"\n📍 Porta: {PORTA}")
    print(f"⚡ Baudrate: {BAUDRATE}")

    # Tentar conectar
    print(f"\n🔌 Conectando ao relé em {PORTA}...")
    try:
        ser = serial.Serial(PORTA, BAUDRATE, timeout=1)
        time.sleep(0.5)  # Aguardar estabilização
        print("✅ Conectado com sucesso!\n")
    except Exception as e:
        print(f"❌ ERRO: Não foi possível conectar!")
        print(f"   {e}")
        print("\n💡 DICAS:")
        print("   1. Verifique se a porta está correta (COM3 ou COM4)")
        print("   2. Feche outros programas que usam o relé")
        print("   3. Reconecte o relé USB")
        return

    print("="*60)
    print("🧪 INICIANDO TESTES")
    print("="*60)
    print("\n⚠️  FIQUE ATENTO:")
    print("   - Ouça o CLIQUE do relé")
    print("   - Veja o LED do relé (se tiver)")
    print("   - Cada teste dura 4 segundos\n")

    input("Pressione ENTER para começar...")
    print()

    resultado_encontrado = False

    for nome_teste, comandos in TESTES.items():
        print(f"\n{'─'*60}")
        print(f"🔍 {nome_teste}")
        print(f"{'─'*60}")

        try:
            # Enviar comando LIGAR
            print(f"   📤 Enviando LIGAR: {comandos['on']}")
            ser.write(comandos['on'])
            ser.flush()
            print(f"   🔴 Aguardando 2 segundos... (OUÇA O CLIQUE!)")
            time.sleep(2)

            # Enviar comando DESLIGAR
            print(f"   📤 Enviando DESLIGAR: {comandos['off']}")
            ser.write(comandos['off'])
            ser.flush()
            print(f"   🟢 Aguardando 2 segundos...")
            time.sleep(2)

            # Perguntar se funcionou
            resposta = input("\n   ❓ Você ouviu o CLIQUE do relé? (s/n): ").strip().lower()

            if resposta == 's':
                print("\n   ✅ COMANDOS ENCONTRADOS!")
                print("\n" + "="*60)
                print("🎉 SUCESSO! Use estes comandos no testv3.py:")
                print("="*60)
                print("\n'commands': {")
                print(f"    'relay1_on': {repr(comandos['on'])},")
                print(f"    'relay1_off': {repr(comandos['off'])},")
                # Ajustar comando do canal 2
                if 'relay on 0' in str(comandos['on']):
                    print(f"    'relay2_on': b'relay on 1\\n\\r',")
                    print(f"    'relay2_off': b'relay off 1\\n\\r',")
                elif comandos['on'] == b'\xFF\x01\x01':
                    print(f"    'relay2_on': b'\\xFF\\x02\\x01',")
                    print(f"    'relay2_off': b'\\xFF\\x02\\x00',")
                elif comandos['on'] == b'\xA0\x01\x01\xA2':
                    print(f"    'relay2_on': b'\\xA0\\x02\\x01\\xA3',")
                    print(f"    'relay2_off': b'\\xA0\\x02\\x00\\xA2',")
                elif comandos['on'] == b'\x51':
                    print(f"    'relay2_on': b'\\x53',")
                    print(f"    'relay2_off': b'\\x54',")
                else:
                    print(f"    'relay2_on': b'RELAY2_ON\\n',")
                    print(f"    'relay2_off': b'RELAY2_OFF\\n',")
                print("}\n")
                resultado_encontrado = True
                break
            else:
                print("   ⏭️  Próximo teste...")

        except Exception as e:
            print(f"   ❌ Erro ao enviar comando: {e}")
            continue

    # Fechar porta serial
    ser.close()
    print("\n🔌 Desconectado do relé")

    if not resultado_encontrado:
        print("\n" + "="*60)
        print("⚠️  NENHUM COMANDO FUNCIONOU")
        print("="*60)
        print("\n💡 POSSÍVEIS CAUSAS:")
        print("   1. Baudrate incorreto (tente 4800, 19200 ou 115200)")
        print("   2. Relé com firmware customizado")
        print("   3. Relé com defeito")
        print("\n📝 PRÓXIMOS PASSOS:")
        print("   1. Consulte o manual do seu relé")
        print("   2. Procure o modelo exato no Google")
        print("   3. Tente outros baudrates modificando BAUDRATE no início do script")

    print("\n✓ Teste concluído!")

# =====================================================
# EXECUTAR
# =====================================================
if __name__ == "__main__":
    try:
        testar_rele()
    except KeyboardInterrupt:
        print("\n\n❌ Teste cancelado pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
