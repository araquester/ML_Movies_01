<p align=center><img width=450px height=250px src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> Proyecto Individual No 1 - `Machine Learning Operations (MLOps)` </h1>

¡Bienvenidos a mi primer proyecto individual de la etapa de labs! - lo que se pretende realizar es toda la etapa de ingenieria de datos en donde se destacan los siguientes procesos:

## - ETL:

proceso que se encargar de extraer los datos, verificarlos, tratarlos y dejarlos listos para la carga de lso mismos.

## - Creacion de funciones: 
elaborar varias funciones que permitan ver los datos y la informacion que contienen los datasets ya limpios.

## - Desarrollo de la API:
Crea una API que al usar las funciones nos permita ver la información que nos soliciten.

## - Virtualizacion o Deployment:
Subir todos los archivos hechos en los procesos anteriores acompañados de un archivo main.py el cual ejecuta la API y dejarlos en un sistema online que permite al usuario acceder a la información en cualquier momento.

## - Creacion de una EDA:
Se verifican las diversas relaciones que hay entre las variables de los datasets, ver si hay outliers o anomalías (que no tienen que ser errores necesariamente), y ver si hay algún patrón interesante que valga la pena explorar en un análisis posterior. Las nubes de palabras dan una buena idea de cuáles palabras son más frecuentes en los títulos.

## - Creación de un sistema de Machine Learning:
es el uso de una función en Python que de acuerdo a un sistema de aprendizaje basado en los mismos datos recolectados en los Datasets anteriores nos generan una recomendación de cinco peliculas semejantes a la que el usuario ingresa.



<p align="center">
<img src="https://binus.ac.id/wp-content/uploads/2021/04/49-2-Caragunacom-768x432.jpg"  height=250>
</p>

**************************************************************

## **Desarrollo de Cada Proceso**
<br>

[Proceso de ETL:](https://github.com/araquester/ML_Movies_01/tree/main/Proceso%20ETL):

Se encuentran dos archivos **`ETL_Movies.ipynb`** y **`ETL_Credits.ipynb`** escritos en el modo Notebook que permite ver cada uno de los comandos que llevaron a una limpieza correcta de al final los tres archivos o datasets con los que trabajara nuestra API y nuestro Modelo de Machine Learning

<br>

**`Creacion de funciones`**:

Se deben crear 6 funciones para los endpoints que se consumirán 
en la API.
  
(1) def **peliculas_mes( *`Mes`* )**:
    Se ingresa un mes en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en el mes consultado en la totalidad del dataset.

(2) def **peliculas_dia( *`Dia`* )**:
    Se ingresa un día en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en día consultado en la totalidad del dataset.

(3) def **score_titulo( *`titulo_de_la_filmación`* )**:
    Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score consultado en la totalidad del dataset.
    
(4) def **votos_titulo( *`titulo_de_la_filmación`* )**:
    Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. La misma variable deberá de contar con al menos 2000 valoraciones, caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.
    
(5) def **get_actor( *`nombre_actor`* )**:
    Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, la cantidad de películas que en las que ha participado y el promedio de retorno. 
    
(6) def **get_director( *`nombre_director`* )**:
    Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.

(7) def **recomendacion( *`recomendacion`*)**:
    Éste consiste en recomendar películas a los usuarios basándose en películas similares, por lo que se debe encontrar la similitud de puntuación entre esa película y el resto de películas, se ordenarán según el score de similaridad y devolverá una lista de Python con 5 valores


**`Virtualizacion o Deployment`**:

Utilizando el servicio [Render](https://render.com/docs/free#free-web-services) que permite tener el contenido de un Repositorio en Github y de allí tomar el archivo [main.py](https://github.com/araquester/ML_Movies_01/blob/main/main.py) que permite que la API pueda ser consumida desde la web.

**`Creacion de una EDA`**:

Hay un total de 45,466 registros (Peliculas) con 24 columnas (Caracteristicas). Muchas de estas caracteristicas tienen valores nulos como por ejemplo homepage y tagline. Intentaremos limpiar mientras al mismo tiempo se hace el analisis

El título original se refiere al título de la película en el idioma nativo en el que se filmó. Por lo tanto, se usara la columna title en vez de original title. Podremos deducir si la película es en otro idioma mirando la característica del idioma original, por lo que no se perderá información tangible al hacerlo.

- TITLE Y OVERVIEW NUBES DE PALABRAS
Con solo escuchar ciertas palabras podemos identificar ciertas peliculas, puede ser por los trailers o por las notas de prensa o por lo tan odiados spoilers miremos que nos dan esas nuebes de palabras por ejemplo en la columna Title

![Nube_palabras_titulo](https://github.com/araquester/ML_Movies_01/blob/main/Src/Nube%20de%20palabras_titulo.jpg)


La palabra "amor" es la palabra más comúnmente utilizada en los títulos de películas. "Chica", "día" y "hombre" también se encuentran entre las palabras más frecuentes. Creo que esto nos da una idea de la presencia del tema romance en las películas.

La palabra Life(vida) también se usa varias veces en los titulos, estas nubes de palabras nos dan una idea bastante clara que la vida y el romance son los mayores temas en las peliculas. 

![Nube palabras_overview](https://github.com/araquester/ML_Movies_01/blob/main/Src/nube%20de%20palabras%20overview.jpg)

Mirando la nube de palabras de la columna overview (Resumen) se puede ver nuevamente la palabra life (Vida) así como también love, lo que nos permite sacar una conclusion clara, por otro lado tambien se ve mucho las palabras Encontrar (find), Amigo (Friend), Pelicula (Film), así como muy poco Documentales (Documentary)

- COLUMNA ORIGINAL LENGUAGE

En esta sección, analicemos los idiomas de las películas en nuestro conjunto de datos. A partir de los países de producción, ya hemos deducido que la mayoría de las películas en el conjunto de datos son en inglés. Veamos cuáles son los otros idiomas principales representados.

![Histograma Idiomas](https://github.com/araquester/ML_Movies_01/blob/main/Src/Histograma%20de%20idiomas%20con%20ingles.jpg)

Hay más de 93 idiomas representados en nuestro conjunto de datos. Como esperábamos, las películas en inglés forman la abrumadora mayoría. Las películas francesas e italianas ocupan un lejano segundo y tercer lugar, respectivamente. Representemos los idiomas más populares (aparte del inglés) en forma de un gráfico de barras.

![Histograma Idiomas sin Ingles](https://github.com/araquester/ML_Movies_01/blob/main/Src/Histograma%20de%20idiomas%20sin%20ingles.jpg)

En esta sección, trabajaremos con las métricas proporcionadas por los usuarios de IMDB. Intentaremos obtener una comprensión más profunda de la popularidad, la calificación promedio y el recuento de votos, y trataremos de deducir cualquier relación entre ellos.

La puntuación de popularidad parece ser una cantidad extremadamente sesgada, con una media de solo 2.9, pero valores máximos que llegan tan alto como 547, lo que es casi un 1800% mayor que la media. Sin embargo, como se puede ver en el gráfico de distribución, casi todas las películas tienen una puntuación de popularidad inferior a 10 (el percentil 75 está en 3.678902).

![Histograma Popularidad](https://github.com/araquester/ML_Movies_01/blob/main/Src/Histograma%20popularidad.jpg)







**`Creación de un sistema de Machine Learning`**:

Una vez que todos los datos son accesibles para la interfaz de programación de aplicaciones (API), están listos para ser utilizados por los departamentos de análisis y aprendizaje automático. Mediante nuestro análisis exploratorio de datos (EDA), podemos comprender mejor los datos a los que tenemos acceso. Ahora es el momento de entrenar nuestro modelo de aprendizaje automático para crear un sistema de recomendación de películas.

Para lograr esto, debemos calcular la similitud de puntuación entre una película en particular y el resto de las películas. Luego, se ordenarán según su puntuación de similitud y se generará una lista en Python con los nombres de las cinco películas de mayor puntuación, en orden descendente.

en el sistema de recomendación su usaron las columnas popularity, vote_average, vote_count y como caracteristica adicional la columna de generos para que las recomendaciones tuvieran cierta relacion con la opcion ingresada, despues de organizar las columnas según sus valores utilizamos el algoritmo NearestNeighbors(n_neighbors=6), el cual puede ser traducido como los vecinos mas cercanos a la caracteristica principal que sería popularity y luego las columnas siguientes en las variables que esten mas cercanas al valor dado por la peicula ingresada 

Este sistema debe ser implementado como una función adicional en la API mencionada anteriormente y se llama: `recomendacion`.

**`Video`**: A continuacion se usa un video para mostrar las caracteristicas de la API y el contenido del Repositorio que permiten que el Render este en la web [VIDEO](https://www.youtube.com/shorts/OXGfNxWNFDU)

## **Material de apoyo**

- [links de ayuda](https://github.com/HX-PRomero/PI_ML_OPS/raw/main/Material%20de%20apoyo.md). 
