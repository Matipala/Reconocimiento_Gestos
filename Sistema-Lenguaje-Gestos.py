import tensorflow as tf
import numpy as np
import cv2

# Ruta a los archivos del modelo y etiquetas
model_path = '/Users/josue/Documents/ComunicacionGestios/Image Model Saved Model/model.savedmodel'
label_path = '/Users/josue/Documents/ComunicacionGestios/Image Model Saved Model/labels.txt'

# Cargar el modelo
model = tf.saved_model.load(model_path)

# Cargar etiquetas
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Verificar el nombre de salida del modelo
infer = model.signatures["serving_default"]
print("Salidas del modelo:", infer.structured_outputs)  # Imprime los nombres de salida

# Inicializar la cámara
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear la imagen horizontalmente para corregir la orientación
    frame = cv2.flip(frame, 1)  # 1 significa voltear horizontalmente

    # Preprocesamiento de la imagen
    img = cv2.resize(frame, (224, 224))  # Ajustar al tamaño esperado por el modelo
    img_array = np.expand_dims(img, axis=0)  # Añadir dimensión de batch
    img_array = img_array / 255.0  # Normalizar

    # Hacer la predicción
    predictions = infer(tf.convert_to_tensor(img_array, dtype=tf.float32))

    # Obtener la clave correcta de salida
    output_key = list(predictions.keys())[0]  # Tomar la primera clave
    predicted_class = np.argmax(predictions[output_key].numpy())  # Obtener la mejor predicción
    label = labels[predicted_class]

    # Definir colores y fuente
    text_color = (255, 255, 255)  # Blanco
    bg_color = (0, 0, 0)  # Negro (para el fondo)
    font_scale = 1.2
    thickness = 3
    margin = 20  # Espacio extra alrededor del texto

    # Obtener tamaño del texto
    (font_width, font_height), _ = cv2.getTextSize(f"GESTO: {label}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # Coordenadas del fondo más grande
    top_left = (30, 30)
    bottom_right = (30 + font_width + margin, 30 + font_height + margin)

    # Dibujar fondo del texto (rectángulo negro más grande)
    cv2.rectangle(frame, top_left, bottom_right, bg_color, -1)

    # Dibujar texto encima del fondo, alineado
    cv2.putText(frame, f"GESTO: {label}", (40, 50 + margin // 2 + font_height // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    cv2.imshow("Reconocimiento de Seña", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
