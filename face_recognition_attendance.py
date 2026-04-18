import cv2
from deepface import DeepFace
import pandas as pd
import sqlite3
import os
import glob
import numpy as np
import time

# --- CONFIGURACIÓN ---
id_camara = 0
MODELO_IA = "Facenet512"
UMBRAL_SIMILITUD = 0.70
ESCALA_PROCESAMIENTO = 0.5

# Detector OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- BASE DE DATOS ---

def inicializar_db():

    conn = sqlite3.connect('biometria_empresa.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS empleados 
                     (id INTEGER PRIMARY KEY, nombre TEXT UNIQUE, embedding BLOB)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS asistencia 
                     (id INTEGER PRIMARY KEY, nombre TEXT, fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()


# --- REGISTRO DE IDENTIDADES ---
def registrar_nuevos_empleados(carpeta_path):
    conn = sqlite3.connect('biometria_empresa.db')
    archivos = glob.glob(os.path.join(carpeta_path, "*.*"))

    if not archivos:
        print(f"\n[!] No se encontraron fotos en la carpeta '{carpeta_path}'")
        return

    for archivo in archivos:
        nombre = os.path.splitext(os.path.basename(archivo))[0]
        try:
            # Procesamos la imagen para obtener la huella facial
            res = DeepFace.represent(img_path=archivo, model_name=MODELO_IA, enforce_detection=True)
            if len(res) > 0:
                embedding = np.array(res[0]["embedding"], dtype=np.float32)
                cursor = conn.cursor()
                cursor.execute("INSERT OR REPLACE INTO empleados (nombre, embedding) VALUES (?, ?)",
                               (nombre, embedding.tobytes()))
                print(f"[OK] Empleado registrado: {nombre}")
        except Exception as e:
            print(f"[Error] No se pudo procesar {archivo}: {e}")

    conn.commit()
    conn.close()
    print("--- Proceso de registro finalizado ---")


# --- VISUALIZACIÓN DE ASISTENCIAS ---
def ver_reporte_asistencias():
    conn = sqlite3.connect('biometria_empresa.db')
    df = pd.read_sql_query("SELECT * FROM asistencia ORDER BY fecha DESC", conn)
    conn.close()

    if df.empty:
        print("\n[!] Todavía no hay registros de asistencia en la base de datos.")
    else:
        print("\n" + "=" * 60)
        print("                HISTORIAL DE ASISTENCIAS")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60 + "\n")


# --- FUNCIONES AUXILIARES ---
def cargar_datos_db():
    conn = sqlite3.connect('biometria_empresa.db')
    df = pd.read_sql_query("SELECT nombre, embedding FROM empleados", conn)
    conn.close()
    if df.empty:
        return None, None
    nombres = df['nombre'].values
    embeddings = np.array([np.frombuffer(x, dtype=np.float32) for x in df['embedding']])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return nombres, embeddings


def registrar_marca_asistencia(nombre):
    conn = sqlite3.connect('biometria_empresa.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO asistencia (nombre) VALUES (?)", (nombre,))
    conn.commit()
    conn.close()


# --- SISTEMA DE CÁMARA ---
def sistema_asistencia_en_vivo():
    nombres_db, embeddings_db = cargar_datos_db()
    if nombres_db is None:
        print("\n[!] La base de datos está vacía. Registrá empleados primero (Opción 1).")
        return

    cap = cv2.VideoCapture(id_camara)
    ultima_deteccion = {}

    print("\n[INFO] Iniciando cámara... Presioná 'q' para volver al menú.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=ESCALA_PROCESAMIENTO, fy=ESCALA_PROCESAMIENTO)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        caras = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in caras:
            # Re-escalar para dibujo
            x_f, y_f, w_f, h_f = [int(v / ESCALA_PROCESAMIENTO) for v in [x, y, w, h]]
            rostro = frame[y_f:y_f + h_f, x_f:x_f + w_f]

            nombre_detectado = "Desconocido"
            similitud = 0
            try:
                res = DeepFace.represent(rostro, model_name=MODELO_IA, enforce_detection=False)
                if len(res) > 0:
                    emb_actual = np.array(res[0]["embedding"], dtype=np.float32)
                    emb_actual /= np.linalg.norm(emb_actual)

                    # Cálculo de similitud
                    sims = np.dot(embeddings_db, emb_actual)
                    idx = np.argmax(sims)
                    similitud = sims[idx]

                    if similitud > UMBRAL_SIMILITUD:
                        nombre_detectado = nombres_db[idx]
                        ahora = time.time()
                        # Evitar registros repetidos en la misma hora (3600 seg)
                        if nombre_detectado not in ultima_deteccion or ahora - ultima_deteccion[nombre_detectado] > 3600:
                            registrar_marca_asistencia(nombre_detectado)
                            ultima_deteccion[nombre_detectado] = ahora
                            print(f"[REGISTRO] Asistencia marcada: {nombre_detectado}")
            except:
                pass

            color = (0, 255, 0) if nombre_detectado != "Desconocido" else (0, 0, 255)
            cv2.rectangle(frame, (x_f, y_f), (x_f + w_f, y_f + h_f), color, 2)
            cv2.putText(frame, f"{nombre_detectado} ({similitud:.2f})", (x_f, y_f - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Control de Asistencia Biometrico", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- MENÚ DE INTERFAZ ---
if __name__ == "__main__":
    inicializar_db()

    while True:
        print("\n" + "---" * 10)
        print("  SISTEMA BIOMÉTRICO v1.0")
        print("---" * 10)
        print("1. Registrar Nuevos Empleados (desde carpeta)")
        print("2. Iniciar Control de Asistencia (Cámara)")
        print("3. Ver Historial de Asistencias")
        print("4. Salir")

        opcion = input("\nSeleccione una opción: ")

        if opcion == "1":
            registrar_nuevos_empleados("Empleados")
        elif opcion == "2":
            sistema_asistencia_en_vivo()
        elif opcion == "3":
            ver_reporte_asistencias()
        elif opcion == "4":
            print("Cerrando sistema. ¡Hasta luego!")
            break
        else:
            print("[!] Opción inválida, intente de nuevo.")
