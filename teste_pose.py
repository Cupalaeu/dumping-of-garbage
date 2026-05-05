import cv2
from ultralytics import YOLO

# 1. Carrega o modelo pré-treinado de Pose Estimation
model = YOLO('yolov8n-pose.pt')

# 2. Inicia a captura de vídeo da webcam (o número 0 geralmente é a câmera padrão)
cap = cv2.VideoCapture(0)

print("Pressione 'q' para sair.")

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("Ignorando frame vazio da câmera.")
        break

    # 3. Roda o modelo no frame atual
    # O parâmetro conf=0.5 significa que ele só vai desenhar se tiver 50%+ de certeza
    results = model(frame, conf=0.5)

    # 4. Desenha as anotações (esqueleto) diretamente no frame original
    annotated_frame = results[0].plot()

    # 5. Mostra o resultado na tela
    cv2.imshow("YOLOv8 Pose - Teste de Webcam", annotated_frame)

    # Condição de parada (pressionar a tecla 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()