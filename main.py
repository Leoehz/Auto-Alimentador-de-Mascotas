import RPi.GPIO as GPIO
import time
import subprocess
import os
import keras
import numpy as np

### Parametros ###
pin_servo = 12

# Ruta a la carpeta "Modelo" 
ruta_modelo = './Modelo' # si la ruta es ./Modelo debe estar en la misma carpeta desde donde se ejecuta este script

# Ruta temporal donde se guardara la nueva imagen
ruta_captura = '.'

# Limite a partir del cual se considera que una prediccion es un Plato Vacio
limite_vacio = 0.6

# Intervalo entre cada captura y analisis
INTERVALO_TIEMPO = 10

### Inicializar Pin del Servo ###
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin_servo,GPIO.OUT)

pwm = GPIO.PWM(pin_servo, 50)
pwm.start(0)

current_position = 0

### Funciones ###
def move_45_degrees() -> None:
    """Funcion para mover el servo 45°"""
    # Calcula el nuevo duty cycle para mover 45 grados
    # Nota: Ajusta el valor del step según tu servo
    step = 1  # Este valor puede necesitar ajuste
    
    new_position = current_position + step
    
    # Si se pasa de los límites, reinicia la posición
    if new_position > 12.5:
        new_position = 2.5
    
    # Mueve el servo a la nueva posición
    pwm.ChangeDutyCycle(new_position)
    time.sleep(0.5)  # Pausa para permitir que el servo se mueva
    
    current_position = new_position
    return
    
def captura(carpeta: str) -> str:
    """Toma una captura y la almacena en @carpeta"""
    image_out = "captura.jpg"
    ubicacion = os.path.join(carpeta, image_out)
    subprocess.run(['libcamera-still', '-o', ubicacion, '--width', '1024', '--height', '1024', '--timeout', '1000'])
    return ubicacion

def analiza_imagen(model, img_path) -> float:
    """Recibe un modelo y una imagen para realizar la prediccion."""
    image_size = (224, 224)
    img = keras.utils.load_img(img_path, target_size=image_size)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocesamos la imagen para que la consuma el modelo
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    #img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(keras.activations.sigmoid(predictions[0][0]))
    return score

def cargar_modelo(model_path: str):
    return keras.models.load_model(model_path)
    

def main():
    modelo = cargar_modelo(ruta_modelo)
    while True:
        print("Iniciando captura y análisis...")
        ruta_foto = captura(ruta_captura)
        score = analiza_imagen(modelo, ruta_foto)
        print(f"La imagen está {100 * (1 - score):.2f}% llena y {100 * score:.2f}% vacía.")
        
        if score > limite_vacio:  # Si está más vacío que lleno activamos el servo.
            move_45_degrees()
        
        time.sleep(INTERVALO_TIEMPO)
    
    # No debe llegar aca
    return
main()
