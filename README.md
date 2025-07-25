Análisis comparativo de estrategias de reducción y ejecución simple en entrenamiento multisalida con TensorFlow
Introducción

El entrenamiento distribuido en TensorFlow mediante MirroredStrategy permite aprovechar múltiples GPUs sincronizando gradientes de forma eficiente. Sin embargo, también es importante contrastar estas estrategias con ejecuciones más simples:

    Ejecución en scope CPU: para establecer una línea base sin paralelismo.

    Ejecución en scope GPU (sin estrategia): modelo entrenado en una única GPU sin distribución explícita.

Este estudio compara las siguientes variantes:

    NcclAllReduce

    ReductionToOneDevice

    Sin estrategia – Scope GPU

    Sin estrategia – Scope CPU

Configuración experimental

    Modelo: Red neuronal multisalida para regresión.

    Datos: Misma partición de datos para todos los experimentos.

    Hardware: Dos GPUs idénticas para los modos distribuidos.

    Framework: TensorFlow 2.x con Keras API.

Resultados

    Estrategia / Scope	Compilación (s)	Entrenamiento (s)	MAE	    MSE	    R²
    NcclAllReduce	        12.1	    218.7	            0.1183	0.0591	0.5435
    ReductionToOneDevice	12.2	    221.1	            0.1196	0.0605	0.636
    Sin estrategia – GPU	8.6	        228.5	            0.1201	0.0622	0.8491
    Sin estrategia – CPU	9.2	        713.3	            0.1215	0.0650	0.6076
    Análisis de rendimiento

Compilación:

        Las ejecuciones sin estrategia compilan más rápido (hasta un 30% menos), especialmente en GPU (8.6s), al evitar la inicialización y sincronización del entorno distribuido.

Entrenamiento:

        NcclAllReduce es el más rápido en entrenamiento efectivo.

        Sin estrategia en GPU tarda ~10s más, y en CPU es más de 3 veces más lento.

        Esto demuestra el coste elevado del entrenamiento puro en CPU y la eficiencia del paralelismo distribuido.

Análisis de precisión

    MAE y MSE:

        Las estrategias distribuidas (NcclAllReduce, ReductionToOneDevice) consiguen menores errores, aunque la diferencia no es abismal.

        Sin estrategia (CPU o GPU) muestra una ligera pérdida de precisión acumulativa.

    Coeficiente de determinación (R²):

        NcclAllReduce mantiene el mejor ajuste (0.8825), con una pérdida gradual conforme bajamos en complejidad del entorno (hasta 0.8724 en CPU).

Conclusión

    NcclAllReduce ofrece la mejor relación coste-beneficio: más rápido y más preciso.

    Entrenar sin estrategia en GPU puede ser aceptable en entornos de desarrollo o cuando se dispone de una sola GPU, pero pierde en eficiencia.

    Entrenar en CPU no es recomendable: es lento, menos preciso y energéticamente costoso.

    En producción o en tareas de alto volumen, las estrategias distribuidas son muy superiores, especialmente con múltiples GPUs disponibles.
