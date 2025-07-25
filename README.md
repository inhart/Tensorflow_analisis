---

### **Análisis comparativo de estrategias de compilación y ejecución en entrenamiento multisalida con TensorFlow**

#### **0. Objetivo**

Evaluar un mismo modelo de regresión multisalida sobre idénticos datos, comparando cinco modos de ejecución en TensorFlow:

1. **Sin estrategia – GPU0**
2. **`MirroredStrategy` default**
3. **`MirroredStrategy` + `NcclAllReduce`**
4. **`MirroredStrategy` + `ReductionToOneDevice`**
5. **Sin estrategia – CPU0**


#### **Configuración experimental**

* **Modelo**: Red secuencial Keras con BatchNorm, LeakyReLU, densas y Dropout.
* **Datos**: 671 muestras, 800 variables de entrada y 8 salidas; escaladas con `StandardScaler`.
* **Hardware**: 1 GPU NVIDIA y CPU single‑thread.
* **TensorFlow**: v2.10.0, CUDA 12.9.


#### **Resumen de resultados**

| Modo                             | Compilación (s) | Entrenamiento (s) | MAE    | MSE      | R²      |
| -------------------------------- | --------------- | ----------------- | ------ | -------- | ------- |
| **GPU0 (tf.device GPU)**         | 0.31            | 85.84             | 1.9882 | 39.1258  | 0.84914 |
| **MirroredStrategy (default)**   | 0.14            | 59.23             | 3.8356 | 132.9852 | 0.54358 |
| **Mirrored + NcclAllReduce**     | 0.14            | 71.77             | 3.5008 | 100.1732 | 0.63644 |
| **Mirrored + ReductionToOneDev** | 0.12            | 54.55             | 3.4485 | 109.7412 | 0.63827 |
| **Sin estrategia – CPU0**        | 9.20            | 713.30            | 3.4917 | 99.4479  | 0.60761 |



#### **Análisis de tiempos**

* **Compilación**:

  * Las estrategias `MirroredStrategy` añaden poca sobrecarga (<0.20 s) respecto a GPU0.
  * La ejecución en CPU0 incurre en un tiempo de compilación notablemente mayor (9.20 s), posiblemente por la inicialización de todos los hilos y operaciones de las capas en CPU.

* **Entrenamiento**:

  * **Más rápido**: `ReductionToOneDevice` (54.55 s) y `MirroredStrategy` default (59.23 s).
  * **Intermedio**: `NcclAllReduce` (71.77 s).
  * **GPU0 puro**: 85.84 s, al no distribuir cargas.
  * **CPU0**: 713.30 s, más de 8× más lento que GPU0, demostrando el coste de entrenar en CPU.

#### **Análisis de precisión**

* **GPU0 (sin estrategia)** logra el **mejor ajuste global**:

  * **R² = 0.8491**, MAE y MSE más bajos, aprovechando plenamente la GPU.

* **MirroredStrategy default** sacrifica precisión por velocidad:

  * **R² = 0.5436**, errores elevados (MAE 3.8356, MSE 132.99).
  * La reducción implícita en CPU para la sincronización parece degradar la convergencia.

* **NcclAllReduce** mejora la calidad frente al default:

  * **R² = 0.6364**, MAE 3.5008, MSE 100.17, gracias al uso de NCCL para comunicación GPU‑GPU.

* **ReductionToOneDevice** combina velocidad y precisión moderada:

  * **R² = 0.6383**, MAE 3.4485, MSE 109.74.

* **CPU0** (sin GPU) ofrece precisión intermedia:

  * **R² = 0.6076**, MAE 3.4917, MSE 99.45, similar a NcclAllReduce, pero a un coste de tiempo de entrenamiento muy elevado.


#### **Conclusiones**

1. **GPU0 puro** es la mejor opción si sólo se dispone de una GPU única: **maximiza R²** con tiempos razonables.
2. **`MirroredStrategy` default** no se recomienda: ultra‑rápido, pero con gran pérdida de calidad.
3. **`NcclAllReduce`** y **`ReductionToOneDevice`** son viables para múltiples GPUs:

   * `NcclAllReduce` recupera precisión y escala bien,
   * `ReductionToOneDevice` ofrece el entrenamiento más rápido entre distribuidos.
4. **Ejecutar en CPU** es impráctico salvo como línea base: muy lento y con precisión moderada.


#### **Recomendación**:
Para escenarios de producción con una sola GPU, entrena **sin estrategia distribuida**. Al escalar a múltiples GPUs, prioriza **NcclAllReduce** o **ReductionToOneDevice** y ajusta hiperparámetros para mejorar convergencia.

---
