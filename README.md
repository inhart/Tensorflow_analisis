# Análisis comparativo de estrategias de reducción y ejecución simple en entrenamiento multisalida con TensorFlow
Introducción

El entrenamiento distribuido en TensorFlow mediante MirroredStrategy permite aprovechar múltiples GPUs sincronizando gradientes de forma eficiente. Sin embargo, también es importante contrastar estas estrategias con ejecuciones más simples:

    Ejecución en scope CPU: como línea base sin paralelismo ni aceleración por GPU.

    Ejecución en scope GPU (sin estrategia): uso de una única GPU sin distribución.

Este estudio compara los siguientes enfoques:

    NcclAllReduce

    ReductionToOneDevice

    Sin estrategia – Scope GPU

    Sin estrategia – Scope CPU

Configuración experimental

    Modelo: Red neuronal multisalida para regresión.

    Datos: Misma partición de entrenamiento y validación.

    Hardware: Dos GPUs idénticas para las estrategias distribuidas.

    Framework: TensorFlow 2.x con API de Keras.

Resultados

    Estrategia / Scope	Compilación (s)	Entrenamiento (s)	MAE	    MSE	    R²
    NcclAllReduce	        12.1    	218.7	            3.8356	132.98	0.5435
    ReductionToOneDevice	12.2	    221.1	            3.5008	100.17	0.6360
    Sin estrategia – GPU	8.6	        228.5	            1.9881	39.125	0.8491
    Sin estrategia – CPU	9.2	        713.3	            3.4916	99.44	0.6076

## Análisis de rendimiento

Compilación:

    Las ejecuciones sin estrategia compilan más rápido al no inicializar el entorno distribuido. En GPU sin estrategia se alcanza el menor tiempo de compilación (8.6 s).

    Las estrategias distribuidas (MirroredStrategy) añaden ~3.5 s de sobrecarga.

Entrenamiento:

    NcclAllReduce es el más rápido (218.7 s), seguido muy de cerca por ReductionToOneDevice.

    El modo sin estrategia en GPU es unos 10 s más lento, pero aún competitivo.

    El entrenamiento en CPU sin estrategia es claramente inviable para producción: más de 700 s.

Análisis de precisión

Errores (MAE, MSE):

    Sin estrategia en GPU logra el mejor rendimiento por amplio margen: MAE de 1.9881 y MSE de 39.125, 
    valores muy por debajo de los obtenidos con estrategias distribuidas.

    Las variantes distribuidas tienen errores altos, especialmente NcclAllReduce, con un MAE de 3.8356 y MSE de 132.98.

Coeficiente de determinación (R²):

    Sin estrategia – GPU obtiene el mejor ajuste del modelo (R² = 0.8491).

    ReductionToOneDevice y el modo CPU logran resultados aceptables (0.6360 y 0.6076).

    NcclAllReduce, pese a ser el más rápido, obtiene el peor R² (0.5435), indicando un ajuste muy pobre.

Conclusión

    Entrenar sin estrategia en una única GPU es la mejor opción en este caso: mejor ajuste, menor error y tiempos de entrenamiento razonables.

    ReductionToOneDevice es aceptable, pero no destaca ni en precisión ni en velocidad.

    NcclAllReduce, pese a ser el más rápido, degrada severamente la calidad del modelo, siendo inapropiado si la precisión es crítica.

    Entrenar en CPU no es recomendable: lento y con rendimiento mediocre.

Recomendación: 

    Para este tipo de modelos multisalida, si solo se dispone de una GPU, 
    entrenar sin estrategia de distribución 
    proporciona los mejores resultados. Si se usan estrategias distribuidas, se recomienda      
    validar cuidadosamente las métricas, ya que el paralelismo no siempre implica mejor ajuste.
