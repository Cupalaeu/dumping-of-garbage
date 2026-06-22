import os
import gc

# ---------------------------------------------------------
# MATRIZ DE EXPERIMENTOS (GRID SEARCH)
# Definido de forma global para fácil acesso e referência
# ---------------------------------------------------------
EXPERIMENTOS = [
    {'peso': 'yolov8m.pt', 'imgsz': 640,  'batch': 32, 'nome': 'yolov8-m-640'},
    {'peso': 'yolo11m.pt', 'imgsz': 640,  'batch': 32, 'nome': 'yolo11-m-640'},
    {'peso': 'yolov8l.pt', 'imgsz': 640,  'batch': 32, 'nome': 'yolov8-l-640'},
    {'peso': 'yolo11l.pt', 'imgsz': 640,  'batch': 32, 'nome': 'yolo11-l-640'},
    
    {'peso': 'yolov8m.pt', 'imgsz': 1280, 'batch': 8,  'nome': 'yolov8-m-1280'},
    {'peso': 'yolo11m.pt', 'imgsz': 1280, 'batch': 8,  'nome': 'yolo11-m-1280'},
    {'peso': 'yolov8l.pt', 'imgsz': 1280, 'batch': 8,  'nome': 'yolov8-l-1280'},
    {'peso': 'yolo11l.pt', 'imgsz': 1280, 'batch': 8,  'nome': 'yolo11-l-1280'}
]

def clean_vram():
    """Realiza limpeza cirúrgica de VRAM para evitar estouro de memória entre treinamentos."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    print("[INFO] VRAM limpa e liberada com sucesso.")

def train_yolo_experiment(exp: dict, data_yaml_path: str, project_dir: str, epochs: int = 100, device: str = None):
    """
    Treina um experimento YOLO específico da matriz.
    
    Args:
        exp (dict): Dicionário de configuração do experimento contendo 'peso', 'imgsz', 'batch' e 'nome'.
        data_yaml_path (str): Caminho absoluto ou relativo para o data.yaml.
        project_dir (str): Caminho onde os resultados do projeto de treinamento serão salvos.
        epochs (int): Número de épocas de treinamento.
        device (str, optional): GPU device string (ex: 'cuda:0') ou 'cpu'. Detectado automaticamente se None.
    """
    import torch
    from ultralytics import YOLO

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
    print(f"\n{'='*60}")
    print(f"[INICIANDO EXPERIMENTO]: {exp['nome']}")
    print(f"Modelo base: {exp['peso']} | Imgsz: {exp['imgsz']} | Batch: {exp['batch']} | Épocas: {epochs}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Instanciação dinâmica do modelo
    model = YOLO(exp['peso'])
    
    # Execução do Treinamento com as configurações e augmentações específicas do projeto
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=exp['imgsz'],
        amp=True,
        batch=exp['batch'],
        device=device,
        project=project_dir,
        name=exp['nome'],
        
        # --- Restrições críticas e hiperparâmetros de aumento ---
        # Data Augmentation: Cores e Iluminação
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Data Augmentation: Geometria da Câmera
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        perspective=0.0005,
        shear=2.0,
        
        # Restrição Crítica: Lixo responde à gravidade (sem espelhamento vertical)
        flipud=0.0, 
        fliplr=0.5, # Espelhamento horizontal permitido
        
        # Data Augmentation: Textura e Oclusão
        mosaic=1.0,
        erasing=0.4,
        mixup=0.1,
        
        workers=8
    )
    
    print(f"\n[SUCESSO] Treinamento do modelo {exp['nome']} finalizado.")
    
    # Liberação de memória
    del model
    clean_vram()
    
    return results

def train_yolo_grid(data_yaml_path: str = None, project_dir: str = None, epochs: int = 100):
    """
    Executa a grade completa de experimentos (Grid Search) sequencialmente.
    
    Args:
        data_yaml_path (str, optional): Caminho para o data.yaml. Resolvido para o padrão se None.
        project_dir (str, optional): Diretório do projeto para salvar os treinos.
        epochs (int): Número de épocas por experimento.
    """
    import torch
    # Resolução de caminhos padrão
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    if data_yaml_path is None:
        data_yaml_path = os.path.join(project_root, 'data', 'dataset_final', 'data.yaml')
    if project_dir is None:
        project_dir = os.path.join(project_root, 'runs', 'train')
        
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Arquivo data.yaml não encontrado em: {data_yaml_path}")
        
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Iniciando Grade Completa de Treinamento no dispositivo: {device}")
    print(f"Caminho do Dataset: {data_yaml_path}")
    print(f"Diretório de Destino: {project_dir}")
    
    total_experimentos = len(EXPERIMENTOS)
    
    for indice, exp in enumerate(EXPERIMENTOS, start=1):
        print(f"\nProgresso da Grade: {indice}/{total_experimentos} experimentos concluídos/em andamento.")
        train_yolo_experiment(
            exp=exp,
            data_yaml_path=data_yaml_path,
            project_dir=project_dir,
            epochs=epochs,
            device=device
        )
        
    print("\n[PROCESSO CONCLUÍDO] Todos os modelos da grade foram treinados com sucesso.")

if __name__ == '__main__':
    # Executa a grade por padrão quando chamado via terminal
    train_yolo_grid()
