import os
from pathlib import Path
from ultralytics import RTDETR

def train_rtdetr(
    data_yaml_path: str = "data/dataset_final/data.yaml",
    epochs: int = 300,
    batch_size: int = -1, # Utilizando -1 para o AutoBatch encontrar o limite máximo seguro da GPU (48GB)
    project_dir: str = "runs/detect",
    name: str = "rtdetr_garbage_detection"
):
    """
    Função modular para treinar o modelo RT-DETR-X para detecção de lixo urbano.
    Aplica as premissas científicas definidas para lidar com objetos amorfos e
    com alta variância intra-classe, e preservação da orientação espacial (gravidade).
    """
    # Verifica se o arquivo de dados existe
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_yaml_path}. Certifique-se de executar os scripts de preparação de dados primeiro.")

    print(f"Iniciando treinamento do RT-DETR-X...")
    print(f"Dataset: {data_yaml_path}")
    print(f"Épocas: {epochs}")
    print(f"Batch Size: {batch_size}")
    
    # 1. Carregar o modelo RT-DETR-X (Extra Large)
    # A Ultralytics fará o download automático do modelo pré-treinado se não existir localmente.
    model = RTDETR("rtdetr-x.pt")

    # 2. Configurar Hiperparâmetros
    # Baseados nas restrições fornecidas e na avaliação técnica para objetos amorfos.
    train_args = {
        "data": data_yaml_path,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": 640, # Resolução de entrada fixa. Letterboxing nativo da Ultralytics cuidará do padding.
        "optimizer": "AdamW", # AdamW é mandatório para Transformers (sem viés indutivo)
        "patience": 100, # Paciência alta para aguardar a convergência global das camadas de atenção
        "amp": False, # Desativado (False) para evitar instabilidade numérica (Loss NaN) em Transformers
        "project": project_dir,
        "name": name,
        "exist_ok": True, # Permite sobrescrever a pasta se existir ou continuar treino
        
        # --- Data Augmentation Geométrica (Obrigatórias) ---
        "mosaic": 1.0, # Ativa Mosaic com 100% de probabilidade
        "mixup": 0.15, # Ativa MixUp (probabilidade conservadora para não descaracterizar totalmente o fundo)
        "copy_paste": 0.3, # Colar lixos em diferentes partes das imagens do batch
        "fliplr": 0.5, # Permite espelhamento horizontal
        "flipud": 0.0, # RESTRIÇÃO CRÍTICA: Lixo e pessoas respondem à gravidade. 0% espelhamento vertical.
        
        # --- Avaliação Técnica (Augmentações Extras Recomendadas) ---
        "scale": 0.9, # Alta variância de escala para simular lixo perto e longe (Zoom in/out de até 90%)
        "erasing": 0.4, # Random Erasing: Força o modelo a não decorar partes específicas do lixo (mitiga overfitting em 3k imagens)
        "cos_lr": True, # Cosine Annealing Learning Rate: Ajuda o Transformer a estabilizar nas últimas épocas
        
        # --- Prevenção de Memorização de Cores (Câmera Externa Tapo) ---
        "hsv_h": 0.015, # Variação de Hue
        "hsv_s": 0.7, # Alta variação de Saturação (simular dias muito claros e escuros)
        "hsv_v": 0.4, # Alta variação de Valor/Luminosidade (simular noite com infravermelho e meio-dia)
        
        # --- Outras Configurações Úteis ---
        "close_mosaic": 10, # Desativa o mosaic nas últimas 10 épocas para fine-tuning final na imagem "real"
        "save": True, # Salvar os melhores pesos
        "save_period": 10, # Fazer backup dos pesos a cada 10 épocas
    }

    # 3. Iniciar Treinamento
    results = model.train(**train_args)
    
    print("\nTreinamento finalizado com sucesso!")
    print(f"Os logs, gráficos e pesos do modelo foram salvos em: {project_dir}/{name}")
    return results

if __name__ == "__main__":
    # Garante que os caminhos sejam interpretados a partir da raiz do projeto, 
    # assumindo que o script é chamado da raiz com `python src/models/train_rtdetr.py`
    # Se o data.yaml estiver em outro lugar, ajuste aqui.
    data_path = os.path.join("data", "dataset_final", "data.yaml")
    
    # Executa a pipeline
    train_rtdetr(data_yaml_path=data_path)
