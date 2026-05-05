from ultralytics import YOLO
import torch

def treinar_modelo_yolo():
    # 1. Validação de Hardware (Garantindo que a 4090 será usada)
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        dispositivo = '0' # O ID da sua RTX 4090
    else:
        print("Aviso: CUDA não detectado. O treinamento ocorrerá na CPU (Lento!).")
        dispositivo = 'cpu'

    # 2. Inicialização do Modelo Base
    # Carrega a arquitetura YOLOv8 nano (rápida) ou 'yolov8m.pt' (média) / 'yolov8x.pt' (pesada)
    # Se o arquivo não existir localmente, o script fará o download automático.
    print("Carregando arquitetura do modelo...")
    modelo = YOLO('yolov8n.pt') 

    # 3. Execução do Treinamento (Hyperparameters)
    print("\nIniciando o treinamento...")
    resultados = modelo.train(
        data='data.yaml',        # O caminho para o seu arquivo de mapa
        epochs=100,              # Quantas vezes o modelo verá o dataset completo (Épocas)
        imgsz=640,               # Resolução do Dataloader (mantém o padrão 640x640 na VRAM)
        batch=64,                # Quantas imagens processar por vez (Na 4090, você pode testar 32 ou 64)
        device=dispositivo,      # Força o uso da GPU mapeada
        patience=20,             # Early Stopping: Para se não melhorar após 20 épocas
        save=True,               # Salva o melhor modelo (.pt) no disco
        project='models',  # Nome da pasta onde os resultados serão salvos
        name='experimento_1',    # Nome da subpasta deste treino específico
        exist_ok=True            # Permite sobrescrever pastas de treinos anteriores
    )

    # 4. Avaliação Final (Validação da Prova Cega)
    print("\nTreinamento concluído. Executando métricas no conjunto de Validação...")
    metricas = modelo.val()
    print(f"Precisão (mAP50): {metricas.box.map50:.3f}")

if __name__ == '__main__':
    # Necessário no Windows para evitar erros de multiprocessamento
    treinar_modelo_yolo()