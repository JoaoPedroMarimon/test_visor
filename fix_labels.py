"""
Script para corrigir labels de imagens sem adesivo
Muda de classe 0 (com_adesivo) para classe 1 (sem_adesivo)
"""

import os
import glob

# ===========================================
# CONFIGURAÇÃO
# ===========================================
# Liste aqui os números das imagens SEM ADESIVO que estão marcadas erradas
IMAGES_SEM_ADESIVO = list(range(47, 60))  # 47 até 59

# Pastas de labels
LABEL_DIRS = [
    'dataset/labels/train',
    'dataset/labels/val'
]

print("="*60)
print("CORRECAO DE LABELS - SEM ADESIVO")
print("="*60)
print(f"\nImagens a corrigir: {IMAGES_SEM_ADESIVO}")
print(f"Mudanca: classe 0 -> classe 1 (sem_adesivo)\n")

# Contadores
total_corrigidos = 0
total_linhas = 0

for label_dir in LABEL_DIRS:
    if not os.path.exists(label_dir):
        print(f"Pasta nao encontrada: {label_dir}")
        continue

    print(f"\nProcessando: {label_dir}")

    for img_num in IMAGES_SEM_ADESIVO:
        label_file = os.path.join(label_dir, f'imagem_{img_num}.txt')

        if not os.path.exists(label_file):
            continue

        # Ler o arquivo
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Processar cada linha
        new_lines = []
        linhas_modificadas = 0

        for line in lines:
            if line.strip():  # Ignorar linhas vazias
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = parts[0]

                    if class_id == '0':
                        # Mudar de 0 para 1
                        parts[0] = '1'
                        new_lines.append(' '.join(parts) + '\n')
                        linhas_modificadas += 1
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)

        # Salvar se houve mudanças
        if linhas_modificadas > 0:
            with open(label_file, 'w') as f:
                f.writelines(new_lines)

            print(f"  OK {label_file}: {linhas_modificadas} linha(s) corrigida(s)")
            total_corrigidos += 1
            total_linhas += linhas_modificadas

print("\n" + "="*60)
print("RESUMO")
print("="*60)
print(f"Total de arquivos corrigidos: {total_corrigidos}")
print(f"Total de linhas modificadas: {total_linhas}")
print("\nProximos passos:")
print("  1. Verifique se os labels foram corrigidos")
print("  2. Execute trainv2.py para treinar com labels corretos")
print("="*60)
