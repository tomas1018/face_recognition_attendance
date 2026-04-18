Sistema de Control de Asistencia Biométrico con Reconocimiento Facial FaceNet512

Descripción

Este proyecto implementa un sistema de reconocimiento facial orientado al control de asistencia laboral en tiempo real. Su objetivo es optimizar tanto la precisión en la identificación de personas como el rendimiento del sistema, mediante el uso de redes neuronales profundas y procesamiento eficiente de video.

El sistema permite registrar empleados a partir de imágenes, reconocerlos en vivo utilizando una cámara y almacenar automáticamente sus registros de asistencia en una base de datos local.

Tecnologías y Librerías

Lenguaje de programación
Python 3.10 o superior

Inteligencia Artificial
DeepFace utilizando el modelo FaceNet512

Procesamiento de imágenes
OpenCV con Haar Cascades para detección de rostros

Base de datos
SQLite3

Análisis de datos
Pandas

Características principales

Reconocimiento facial en tiempo real
Detección de rostros mediante OpenCV
Identificación basada en embeddings de 512 dimensiones
Comparación mediante similitud de coseno
Registro automático de asistencia en base de datos
Control de registros duplicados configurable por intervalo de tiempo
Sistema optimizado para bajo consumo de recursos

Funcionamiento del sistema

El programa se organiza en tres procesos principales accesibles desde un menú de consola:

Registro de personal
El sistema recorre la carpeta de imágenes, detecta los rostros y genera un vector de características para cada persona. Este vector se almacena en la base de datos junto con el nombre del empleado.

Control en vivo
Se accede a la cámara y se procesan los frames en tiempo real. El sistema detecta rostros utilizando OpenCV y luego genera embeddings con DeepFace. Estos se comparan contra la base de datos para identificar a cada persona.

Reporte de asistencia
Permite consultar los registros almacenados en la base de datos, aplicando filtros para evitar duplicaciones dentro de un intervalo de tiempo definido.

Optimizaciones implementadas

Reducción del tamaño de los frames para mejorar la velocidad de procesamiento
Separación de la detección y el reconocimiento para mayor eficiencia
Normalización de embeddings para mejorar la precisión
Comparación vectorizada para acelerar el reconocimiento
Control de frecuencia de registro para evitar redundancia de datos

Instrucciones de instalación

Clonar el repositorio en el equipo local

Instalar las dependencias necesarias ejecutando el comando
pip install -r requirements.txt

Crear una carpeta llamada Empleados en el directorio del proyecto

Agregar imágenes de los empleados dentro de la carpeta, utilizando como nombre de archivo el nombre de cada persona

Ejecutar el archivo principal face_recognition_attendance.py

Realizar el registro inicial de empleados antes de iniciar el reconocimiento en vivo

Consideraciones

El sistema requiere condiciones de iluminación adecuadas para un correcto funcionamiento
La precisión puede mejorar utilizando múltiples imágenes por persona
No incluye mecanismos de detección de suplantación de identidad
El rendimiento puede variar según el hardware disponible