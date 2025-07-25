### **Análisis comparativo de estrategias de compilación y ejecución en entrenamiento multisalida con TensorFlow**

#### **0. Objetivo**

Evaluar un mismo modelo de regresión multisalida sobre idénticos datos, comparando cuatro modos de ejecución en TensorFlow:

1. **Sin estrategia – GPU0**
2. **`MirroredStrategy` con GPUs lógicas (default)**
3. **`MirroredStrategy` + `NcclAllReduce`**
4. **`MirroredStrategy` + `ReductionToOneDevice`**

#### **Configuración experimental**

* **Modelo**: Red secuencial Keras con capas BatchNorm, LeakyReLU, densas y Dropout.
* **Datos**: 671 muestras, 800 entradas y 8 salidas; normalizadas con `StandardScaler`.
* **Hardware**: 1 GPU NVIDIA (detected) y CPU como fallback.
* **TensorFlow**: v2.10.0, CUDA 12.9.

#### **Resumen de resultados**

| Modo                             | Compilación (s) | Entrenamiento (s) | MAE    | MSE      | R²      |
| -------------------------------- | --------------- | ----------------- | ------ | -------- | ------- |
| **GPU0 (tf.device GPU)**         | 0.31            | 85.84             | 1.9882 | 39.1258  | 0.84914 |
| **MirroredStrategy (default)**   | 0.14            | 59.23             | 3.8356 | 132.9852 | 0.54358 |
| **Mirrored + NcclAllReduce**     | 0.14            | 71.77             | 3.5008 | 100.1732 | 0.63644 |
| **Mirrored + ReductionToOneDev** | 0.12            | 54.55             | 3.4485 | 109.7412 | 0.63827 |

#### **Análisis de tiempos**

* **Compilación**:

  * La inicialización de cualquier `MirroredStrategy` añade \~0.17 s de sobrecarga frente a la ejecución directa en GPU.
  * Entre las tres estrategias de espejo, las diferencias son mínimas (<0.02 s).

* **Entrenamiento**:

  * **Más rápido**: `ReductionToOneDevice` (54.55 s) y `MirroredStrategy` default (59.23 s).
  * **Intermedio**: `NcclAllReduce` (71.77 s).
  * **Más lento**: ejecución directa en GPU0 (85.84 s), al no aprovechar paralelismo en los kernels de distribución.

#### **Análisis de precisión**

* **Sin estrategia – GPU0** ofrece el **mejor ajuste**:

  * **R² = 0.8491**, MAE y MSE muy bajos, reflejando una buena adaptación a los datos.
* **Estrategias distribuidas**:

  * **Default Mirrored**: su peor R² (0.5436) y errores muy altos (MAE 3.8356, MSE 132.99) indican posible inestabilidad por la reducción implícita en CPU.
  * **NcclAllReduce** mejora notablemente R² a 0.6364 y reduce errores (MAE 3.5008, MSE 100.17), gracias al uso de NCCL para sincronización rápida.
  * **ReductionToOneDevice** consigue el **mejor compromiso** entre tiempo y precisión dentro de las estrategias distribuidas (R² 0.6383).

#### **Conclusiones**

1. **Ejecución directa en GPU** (sin `MirroredStrategy`) resulta la **mejor opción** cuando se dispone de una sola GPU, maximizando precisión a costa de un entrenamiento algo más lento.
2. **`MirroredStrategy` default** no es recomendable: rápido, pero sacrifica gran parte de la calidad del modelo.
3. **`NcclAllReduce`** recupera parte de la precisión perdida y puede escalar bien a múltiples GPUs, aunque no tan veloz como otras variantes.
4. **`ReductionToOneDevice`** ofrece el tiempo de entrenamiento más bajo y precisión aceptable entre las opciones distribuidas.

> **Recomendación**: Para un entorno monógrafo de GPU única, entrena **sin** `MirroredStrategy`. Si escalas a varias GPUs, emplea **NcclAllReduce** o **ReductionToOneDevice**, calibrando hiperparámetros para mejorar el ajuste.
