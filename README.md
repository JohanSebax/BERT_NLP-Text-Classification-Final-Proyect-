
## El algoritmo que aprendió a sanar `BERT_Model.ipynb`

Este notebook implementa un modelo de clasificación de texto utilizando BERT (Bidirectional Encoder Representations from Transformers) sobre historias clínicas procesadas. El flujo principal del código es el siguiente:

1. **Importación de librerías**: Se utilizan librerías como `transformers` para BERT, `torch` para el manejo de tensores y redes neuronales, `pandas` para manipulación de datos y `sklearn` para la división del dataset.

2. **Configuración inicial**: Se definen parámetros como la semilla aleatoria, longitud máxima de texto, tamaño de batch, ruta del dataset y número de clases. Se configura el dispositivo para el procesamiento con (GPU) para el entrenamiento.

3. **Carga y preprocesamiento de datos**: Se carga el dataset desde un archivo Excel, se visualizan los datos y se realiza limpieza de texto (eliminación de corchetes, comillas y unión de palabras).

4. **Tokenización**: Se utiliza el tokenizer de BERT para convertir los textos en tokens y sus respectivos IDs, preparando los datos para el modelo.

5. **Creación del Dataset personalizado**: Se define una clase `CustomDataset` para adaptar los datos al formato requerido por PyTorch y BERT.

6. **DataLoader**: Se crean los DataLoaders para el conjunto de entrenamiento y prueba, facilitando el manejo de batches.

7. **Definición del modelo**: Se implementa la clase `BERTTextClassifier`, que utiliza la arquitectura de BERT y añade una capa lineal para la clasificación.

8. **Configuración de entrenamiento**: Se definen el optimizador, scheduler y función de pérdida (`CrossEntropyLoss`).

9. **Funciones de entrenamiento y evaluación**: Se implementan funciones para entrenar y evaluar el modelo, calculando métricas de precisión y pérdida.


Este notebook está orientado a la clasificación automática de textos médicos en diferentes grupos, utilizando el poder de BERT y PyTorch para el procesamiento de lenguaje natural.
