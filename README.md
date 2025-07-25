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
    NcclAllReduce	        12.1	            218.7	    0.1183	0.0591	0.5435
    ReductionToOneDevice	12.2	            221.1	    0.1196	0.0605	0.6360
    Sin estrategia – GPU	8.6	                228.5	    0.1201	0.0622	0.8491
    Sin estrategia – CPU	9.2	                713.3	    0.1215	0.0650	0.6076

Análisis de rendimiento

Compilación:

    Las ejecuciones sin estrategia compilan más rápido (hasta un 30% menos), especialmente en GPU (8.6s), al evitar la inicialización y sincronización del entorno distribuido.

Entrenamiento:

    NcclAllReduce es el más rápido en entrenamiento efectivo.

    Sin estrategia en GPU tarda ~10s más, y en CPU es más de 3 veces más lento.

    Esto demuestra el coste elevado del entrenamiento puro en CPU y la eficiencia del paralelismo distribuido.

Análisis de precisión

MAE y MSE:

    Las diferencias entre estrategias son pequeñas, pero las variantes distribuidas (NcclAllReduce, ReductionToOneDevice) obtienen ligeramente mejores errores absolutos.

Coeficiente de determinación (R²):

    El valor más alto lo consigue la ejecución sin estrategia en GPU (0.8491), indicando un ajuste más preciso en ese entorno.

    Le sigue ReductionToOneDevice (0.636), que supera a NcclAllReduce (0.5435), que obtiene el peor ajuste a pesar de entrenar más rápido.

    La ejecución en CPU (0.6076) también supera a NcclAllReduce, aunque es notablemente más lenta.

Conclusión

    A pesar de su rendimiento en tiempo, NcclAllReduce no logra el mejor ajuste: presenta el peor R² del conjunto, lo que sugiere posibles problemas de convergencia o sincronización agresiva.

    Entrenar sin estrategia en GPU consigue el mejor resultado en R², aunque a costa de un entrenamiento algo más lento (~10 s más).

    ReductionToOneDevice ofrece un equilibrio razonable, con buen tiempo y precisión decente.

    El entrenamiento en CPU sigue siendo desaconsejable por su lentitud, aunque su ajuste es mejor que el de NcclAllReduce.

Recomendación: en contextos donde la precisión del modelo es prioritaria frente al tiempo, evitar NcclAllReduce. Para desarrollo o producción con una sola GPU, la ejecución directa en GPU puede ser la más efectiva. Para entornos distribuidos, se recomienda probar otras configuraciones o ajustar hiperparámetros si se usa NcclAllReduce.
