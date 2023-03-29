# Importamos las librerías necesarias.
import logging
import sys
from argparse import ArgumentParser

import cv2  # Usaremos el módulo dnn de OpenCV
import imutils
import numpy as np

# Configuremos el logger para que imprima en la terminal.
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Definimos los parámetros de entrada del script.
argument_parser = ArgumentParser()
argument_parser.add_argument('-i', '--image', required=True, help='Ruta a la imagen de entrada.')
argument_parser.add_argument('-p', '--prototxt', required=True, help='Ruta al archivo de despliegue del detector de'
                                                                     ' objetos en formato prototxt.')
argument_parser.add_argument('-m', '--model', required=True, help='Ruta al modelo en Caffe pre-entrenado.')
argument_parser.add_argument('-c', '--confidence', type=float, default=0.2, help='Probabilidad mínima de una detección'
                                                                                 'para no ser descartada.')
arguments = vars(argument_parser.parse_args())

# Definimos las clases reconocidas por el modelo, y creamos un color aleatorio para cada una.
CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Cargamos el modelo usando el archivo prototxt y los pesos.
logging.info('Cargando el modelo.')
model = cv2.dnn.readNetFromCaffe(arguments['prototxt'], arguments['model'])

# Cargamos la imagen de entrada y extraemos sus dimensiones.
image = cv2.imread(arguments['image'])
height, width = image.shape[:2]

# Redimensionamos la imagen.
resized_image = cv2.resize(image, (300, 300))

# Convertimos la imagen en un blob (un vector numérico), el cual normalizamos.
blob = cv2.dnn.blobFromImage(resized_image, 0.007843, (300, 300), 127.5)

# Pasamos el blob por el detector de objetos.
logging.info('Detectando objetos...')
model.setInput(blob)
detections = model.forward()

# Iteramos sobre cada detección.
for i in range(0, detections.shape[2]):
    # Extraemos la confianza o probabilidad de la detección actual.
    confidence = detections[0, 0, i, 2]

    # Sólo consideraremos esta detección si su probabilidad está por encima del parámetro --confidence.
    if confidence > arguments['confidence']:
        # Extraemos y reescalamos la caja correspondiente a la detección.
        index = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        x_start, y_start, x_end, y_end = box.astype('int')

        # Redactamos la etiqueta, la cual contiene la clase y la probablididad de la detección.
        label = f'{CLASSES[index]}: {confidence * 100:.2f}%'
        logging.info(label)

        # Pintamos la detección en la imagen original.
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), COLORS[index], 3)

        # Colocamos la etiqueta encima de la caja.
        y = y_start - 15 if y_start - 15 > 15 else y_start + 15
        cv2.putText(image, label, (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[index], 3)
# Mostramos el resultado y lo guardamos en disco.
cv2.imshow('Resultado', imutils.resize(image, width=1024))
cv2.imwrite(arguments['image'].rsplit('.', maxsplit=1)[0] + '_result.jpg', image)
cv2.waitKey(0)
