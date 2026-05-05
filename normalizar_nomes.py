import os
import re

# ================= CONFIGURAÇÕES =================
# Coloque aqui o caminho da pasta onde estão as fotos e os .txt
PASTA_ALVO = "03. RECORTADOS" 
# =================================================

def normalizar_nomes():
    if not os.path.exists(PASTA_ALVO):
        print(f"❌ Erro: A pasta '{PASTA_ALVO}' não foi encontrada.")
        return

    arquivos = os.listdir(PASTA_ALVO)
    print(f"🔎 Verificando {len(arquivos)} arquivos na pasta...\n")

    arquivos_renomeados = 0

    for arquivo in arquivos:
        # Separa o nome do arquivo da extensão (ex: "0001(1)(1)" e ".png")
        nome_original, extensao = os.path.splitext(arquivo)
        
        # Regex: Substitui tudo que estiver entre parênteses, incluindo os próprios parênteses, por nada ('')
        # O .strip() no final remove espaços em branco que possam sobrar, ex: "0001 (1).txt" -> "0001.txt"
        nome_limpo = re.sub(r'\([^)]*\)', '', nome_original).strip()
        
        novo_arquivo = f"{nome_limpo}{extensao}"
        
        # Se o nome limpo for diferente do nome atual, significa que ele tinha parênteses
        if arquivo != novo_arquivo:
            caminho_antigo = os.path.join(PASTA_ALVO, arquivo)
            caminho_novo = os.path.join(PASTA_ALVO, novo_arquivo)
            
            # Trava de segurança: Verifica se o arquivo limpo (ex: 0001.png) já existe na pasta
            if os.path.exists(caminho_novo):
                print(f"⚠️ CONFLITO: Não posso renomear '{arquivo}' para '{novo_arquivo}' porque o arquivo destino já existe!")
            else:
                os.rename(caminho_antigo, caminho_novo)
                print(f"✅ Renomeado: {arquivo} -> {novo_arquivo}")
                arquivos_renomeados += 1

    print(f"\n🚀 Limpeza concluída! {arquivos_renomeados} arquivos foram normalizados.")

if __name__ == "__main__":
    normalizar_nomes()