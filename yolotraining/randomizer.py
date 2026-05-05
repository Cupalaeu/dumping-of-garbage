import os
import random
import shutil

# ==========================================
# 1. CONFIGURAÇÕES DOS DIRETÓRIOS
# ==========================================
# Insira aqui o nome das pastas onde seus arquivos estão agora
PASTA_ORIGEM_IMAGENS = "minhas_imagens" 
PASTA_ORIGEM_LABELS = "meus_labels"     

# O nome da pasta raiz que será criada no padrão YOLO
PASTA_DESTINO_BASE = "dataset"          

# Proporções matemáticas do split
SPLIT_TREINO = 0.80
SPLIT_VAL = 0.10
# O teste será o restante (0.10)

# ==========================================
# 2. CRIAÇÃO DA ARQUITETURA DE PASTAS YOLO
# ==========================================
pastas_yolo = [
    f"{PASTA_DESTINO_BASE}/images/train",
    f"{PASTA_DESTINO_BASE}/images/val",
    f"{PASTA_DESTINO_BASE}/images/test",
    f"{PASTA_DESTINO_BASE}/labels/train",
    f"{PASTA_DESTINO_BASE}/labels/val",
    f"{PASTA_DESTINO_BASE}/labels/test"
]

print("Criando estrutura de diretórios do YOLO...")
for pasta in pastas_yolo:
    os.makedirs(pasta, exist_ok=True)

# ==========================================
# 3. VALIDAÇÃO DE PARES (IMAGEM + TXT)
# ==========================================
imagens = [f for f in os.listdir(PASTA_ORIGEM_IMAGENS) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
pares_validos = []

for img_nome in imagens:
    nome_base = os.path.splitext(img_nome)[0]
    txt_nome = f"{nome_base}.txt"
    
    # Trava de segurança: Só aceita se o .txt existir
    if os.path.exists(os.path.join(PASTA_ORIGEM_LABELS, txt_nome)):
        pares_validos.append(nome_base)

print(f"Total de pares (Imagem + Rótulo) válidos encontrados: {len(pares_validos)}")

# ==========================================
# 4. EMBARALHAMENTO E MATEMÁTICA DO SPLIT
# ==========================================
# A semente (seed) fixa garante que, se você rodar o script 2 vezes, 
# ele fará o mesmo sorteio, evitando misturar os dados acidentalmente depois.
random.seed(42)
random.shuffle(pares_validos)

total = len(pares_validos)
qtd_treino = int(total * SPLIT_TREINO)
qtd_val = int(total * SPLIT_VAL)

conjunto_treino = pares_validos[:qtd_treino]
conjunto_val = pares_validos[qtd_treino : qtd_treino + qtd_val]
conjunto_teste = pares_validos[qtd_treino + qtd_val:]

# ==========================================
# 5. EXECUÇÃO DA CÓPIA (O ROTEAMENTO)
# ==========================================
def copiar_arquivos(lista_nomes, categoria):
    for nome in lista_nomes:
        # Define os arquivos de origem
        img_origem = None
        # Procura a extensão correta da imagem
        for ext in ['.png', '.jpg', '.jpeg']:
            if os.path.exists(os.path.join(PASTA_ORIGEM_IMAGENS, nome + ext)):
                img_origem = os.path.join(PASTA_ORIGEM_IMAGENS, nome + ext)
                img_ext = ext
                break
                
        txt_origem = os.path.join(PASTA_ORIGEM_LABELS, nome + '.txt')
        
        # Define os destinos
        img_destino = os.path.join(PASTA_DESTINO_BASE, 'images', categoria, nome + img_ext)
        txt_destino = os.path.join(PASTA_DESTINO_BASE, 'labels', categoria, nome + '.txt')
        
        # Copia os arquivos
        shutil.copy(img_origem, img_destino)
        shutil.copy(txt_origem, txt_destino)

print("\nCopiando arquivos para as pastas...")
copiar_arquivos(conjunto_treino, 'train')
copiar_arquivos(conjunto_val, 'val')
copiar_arquivos(conjunto_teste, 'test')

# ==========================================
# 6. RELATÓRIO FINAL
# ==========================================
print("\n--- Separação Concluída com Sucesso! ---")
print(f"Arquivos em Treino (Train): {len(conjunto_treino)} imagens e labels")
print(f"Arquivos em Validação (Val): {len(conjunto_val)} imagens e labels")
print(f"Arquivos em Teste (Test): {len(conjunto_teste)} imagens e labels")
print(f"\nSua pasta raiz para colocar no data.yaml é: ./{PASTA_DESTINO_BASE}")