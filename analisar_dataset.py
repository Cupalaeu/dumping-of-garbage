import json
from collections import Counter

def analisar_dataset(caminho_arquivo):
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo {caminho_arquivo} não encontrado.")
        return
    except json.JSONDecodeError:
        print(f"Erro: Falha ao decodificar o JSON em {caminho_arquivo}.")
        return

    larguras = []
    alturas = []

    for key, info in data.items():
        if "largura" in info:
            larguras.append(info["largura"])
        if "altura" in info:
            alturas.append(info["altura"])

    contagem_largura = Counter(larguras)
    contagem_altura = Counter(alturas)

    print("=== Estatísticas de Largura ===")
    print(f"Total de registros com 'largura': {len(larguras)}")
    for val, count in sorted(contagem_largura.items()):
        print(f"Valor: {val} | Quantidade: {count}")

    print("\n=== Estatísticas de altura ===")
    print(f"Total de registros com 'altura': {len(alturas)}")
    for val, count in sorted(contagem_altura.items()):
        print(f"Valor: {val} | Quantidade: {count}")

if __name__ == "__main__":
    analisar_dataset('dataset_index.json')
