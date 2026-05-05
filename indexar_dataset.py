import os
import json
import cv2

# ================= CONFIGURAÇÕES =================
PASTA_IMAGENS = "01. LIXO ORIGNAL" # <-- ALTERE AQUI para o caminho real
ARQUIVO_JSON = "dataset_index.json"
# =================================================

def indexar_imagens():
    # 1. Inicializa ou carrega o JSON existente
    if os.path.exists(ARQUIVO_JSON):
        with open(ARQUIVO_JSON, 'r', encoding='utf-8') as f:
            banco_dados = json.load(f)
    else:
        banco_dados = {}

    # Cria um 'set' (conjunto) com os nomes originais já cadastrados para busca rápida (performance)
    nomes_ja_processados = {info["nome_original"] for info in banco_dados.values()}

    # Descobre qual é o próximo índice disponível
    if banco_dados:
        # Pega a maior chave do JSON, converte pra int, e soma 1
        indice_atual = max([int(k) for k in banco_dados.keys()]) + 1
    else:
        indice_atual = 1 # Começamos do 0001

    arquivos_na_pasta = os.listdir(PASTA_IMAGENS)
    print(f"Iniciando varredura em {len(arquivos_na_pasta)} arquivos...\n")

    for arquivo in arquivos_na_pasta:
        # Ignora arquivos que não sejam PNG
        if not arquivo.lower().endswith('.png'):
            continue
            
        caminho_completo = os.path.join(PASTA_IMAGENS, arquivo)
        nome_sem_extensao = arquivo[:-4]
        
        # 2. Ignora as imagens que JÁ estão renomeadas (ex: '0001.png', '0002.png')
        # Verifica se o nome tem 4 caracteres e se é composto só de números
        if len(nome_sem_extensao) == 4 and nome_sem_extensao.isdigit():
            continue
            
        # 3. Verifica se a imagem é uma duplicata (já existe no JSON)
        if arquivo in nomes_ja_processados:
            print(f"🗑️ Duplicada detectada: {arquivo} -> Deletando arquivo...")
            os.remove(caminho_completo)
            continue
            
        # 4. Verifica o padrão do nome (se tem o '@' do Google Earth)
        if '@' not in arquivo:
            print(f"⚠️ AVISO: Arquivo ignorado (fora do padrão): {arquivo}")
            continue
            
        # 5. Extração de Latitude e Longitude
        try:
            # Corta a string no '@' e pega a segunda metade. Depois corta nas vírgulas.
            parte_coordenadas = arquivo.split('@')[1]
            coordenadas = parte_coordenadas.split(',')
            latitude = float(coordenadas[0])
            longitude = float(coordenadas[1])
        except Exception as e:
            print(f"⚠️ AVISO: Falha ao extrair coordenadas de {arquivo}. Erro: {e}. Ignorando.")
            continue
            
        # 6. Leitura da Imagem via OpenCV para extrair dimensões
        img = cv2.imread(caminho_completo)
        if img is None:
            print(f"⚠️ AVISO: Arquivo corrompido ou ilegível pelo OpenCV: {arquivo}. Ignorando.")
            continue
            
        # No OpenCV, o shape retorna (altura, largura, canais_de_cor)
        altura, largura = img.shape[:2] 
        
        # 7. Renomear e Cadastrar no JSON
        nova_chave = f"{indice_atual:04d}" # Formata com 4 zeros (ex: 1 vira 0001)
        novo_nome_arquivo = f"{nova_chave}.png"
        novo_caminho = os.path.join(PASTA_IMAGENS, novo_nome_arquivo)
        
        # Renomeia no sistema de arquivos
        os.rename(caminho_completo, novo_caminho)
        
        # Registra no nosso dicionário
        banco_dados[nova_chave] = {
            "nome_original": arquivo,
            "largura": largura,
            "altura": altura, # Usando "altura" conforme seu pedido
            "latitude": latitude,
            "longitude": longitude
        }
        
        nomes_ja_processados.add(arquivo)
        print(f"✅ Sucesso: {arquivo} -> Renomeado para {novo_nome_arquivo}")
        
        # Incrementa o contador para a próxima foto
        indice_atual += 1

    # 8. Salva o JSON atualizado na raiz do projeto
    with open(ARQUIVO_JSON, 'w', encoding='utf-8') as f:
        # indent=4 deixa o JSON formatado e legível, ensure_ascii=False permite acentos
        json.dump(banco_dados, f, indent=4, ensure_ascii=False)

    print("\n🚀 Processamento concluído! O arquivo 'dataset_index.json' foi atualizado.")

if __name__ == "__main__":
    indexar_imagens()