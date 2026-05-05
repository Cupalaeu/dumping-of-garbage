import os
import cv2
import random

# ================= CONFIGURAÇÕES =================
PASTA_ENTRADA = "01. LIXO ORIGINAL"
PASTA_SAIDA = "02. LIXO RECORTADO"

# Ajuste esses valores em pixels (após medir no Paint/Photoshop)
CORTE_TOPO_PX = 75   # Quantos pixels remover de cima
CORTE_BASE_PX = 40   # Quantos pixels remover de baixo

# Novo controle de centralização do Jitter
LIMITE_JITTER_PX = 150  # O máximo de pixels que o quadrado pode "fugir" do centro absoluto
# =================================================

def processar_quadrados_centralizados():
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)

    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.lower().endswith('.png')]
    print(f"Iniciando o corte centralizado de {len(arquivos)} imagens...\n")

    for arquivo in arquivos:
        caminho_img = os.path.join(PASTA_ENTRADA, arquivo)
        img = cv2.imread(caminho_img)

        if img is None:
            print(f"⚠️ Erro ao ler a imagem: {arquivo}")
            continue

        altura_original, largura_original = img.shape[:2]

        random.seed(arquivo)

        # 1. Calcula a nova altura útil (tamanho do quadrado)
        altura_util = altura_original - CORTE_TOPO_PX - CORTE_BASE_PX
        tamanho_quadrado = altura_util

        if largura_original < tamanho_quadrado:
            print(f"⚠️ AVISO: A imagem {arquivo} é muito estreita. Pulando.")
            continue

        # 2. Nova Lógica do Jitter (Focado no Centro)
        # Total de espaço livre na horizontal
        folga_horizontal = largura_original - tamanho_quadrado
        
        # Descobre qual seria o ponto de partida 'X' para um quadrado PERFEITAMENTE no meio
        centro_perfeito_x = folga_horizontal // 2

        # Define os limites de variação (evitando que o valor fique menor que 0 ou maior que a folga)
        limite_minimo_x = max(0, centro_perfeito_x - LIMITE_JITTER_PX)
        limite_maximo_x = min(folga_horizontal, centro_perfeito_x + LIMITE_JITTER_PX)

        # Escolhe um ponto aleatório dentro dessa zona segura ao redor do centro
        ponto_inicio_x = random.randint(limite_minimo_x, limite_maximo_x)
        
        # Define os recortes
        y_inicial = CORTE_TOPO_PX
        y_final = altura_original - CORTE_BASE_PX
        x_inicial = ponto_inicio_x
        x_final = ponto_inicio_x + tamanho_quadrado

        # 3. Faz o recorte
        img_quadrada = img[y_inicial:y_final, x_inicial:x_final]

        # 4. Salva a imagem
        caminho_saida = os.path.join(PASTA_SAIDA, arquivo)
        cv2.imwrite(caminho_saida, img_quadrada)
        
        print(f"✅ {arquivo}: Quadrado ({tamanho_quadrado}x{tamanho_quadrado}) - Início X: {ponto_inicio_x} (Centro base era {centro_perfeito_x})")

    print("\n🚀 Processamento concluído! Verifique a pasta de saída.")

if __name__ == "__main__":
    processar_quadrados_centralizados()