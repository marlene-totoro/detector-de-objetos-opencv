# Detector de objetos con OpenCV

Este es un proyecto de ejemplo para detectar objetos en una imagen usando OpenCV y un modelo pre-entrenado.

## Requerimientos

- numpy
- opencv-contrib-python
- imutils

## Instalación

```bash
git clone https://github.com/marlene-totoro/detector-de-objetos-opencv
```

no olvide que debe encontrarse en la ruta correcta dentro de detector-de-objetos-opencv
```bash
cd detector-de-objetos-opencv
```

- En Linux:
```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

- En windows:
```bash
py -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

## Uso

Ejecute el siguiente comando para detectar objetos en una imagen:
```bash
python detect.py -p resources/MobileNetSSD_deploy.prototxt.txt -m resources/MobileNetSSD_deploy.caffemodel -i image.jpg
```

Donde `-p` es la ruta al archivo de despliegue del detector de objetos en formato prototxt, `-m` es la ruta al modelo en Caffe pre-entrenado, e `-i` es la ruta a la imagen de entrada.

Los resultados se mostrarán en una ventana y se guardarán en un archivo con el sufijo `_result` agregado al nombre de la imagen de entrada en la misma carpeta. 