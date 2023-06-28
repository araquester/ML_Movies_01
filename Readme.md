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

There are a total of 45,466 movies with 24 features. Most of the features have very few NaN values (apart from homepage and tagline). We will attempt at cleaning this dataset to a form suitable for analysis in the next section.

The original title refers to the title of the movie in the native language in which the movie was shot. As such, I will prefer using the translated, Anglicized name in this analysis and hence, will drop the original titles altogether. We will be able to deduce if the movie is a foreign language film by looking at the original_language feature so no tangible information is lost in doing so.

Title and Overview Wordclouds
Are there certain words that figure more often in Movie Titles and Movie Blurbs? I suspect there are some words which are considered more potent and considered more worthy of a title. Let us find out!

![Texto alternativo](src/nube de palabras overview.jpg)

The word Love is the most commonly used word in movie titles. Girl, Day and Man are also among the most commonly occuring words. I think this encapsulates the idea of the ubiquitious presence of romance in movies pretty well.

Life is the most commonly used word in Movie titles. One and Find are also popular in Movie Blurbs. Together with Love, Man and Girl, these wordclouds give us a pretty good idea of the most popular themes present in movies.

Original Language
In this section, let us look at the languages of the movies in our dataset. From the production countries, we have already deduced that the majority of the movies in the dataset are English. Let us see what the other major languages represented are.

There are over 93 languages represented in our dataset. As we had expected, English language films form the overwhelmingly majority. French and Italian movies come at a very distant second and third respectively. Let us represent the most popular languages (apart from English) in the form of a bar plot.

In this section, we will work with metrics provided to us by TMDB users. We will try to gain a deeper understanding of the popularity, vote average and vote count features and try and deduce any relationships between them as well as other numeric features such as budget and revenue.

The Popularity score seems to be an extremely skewed quentity with a mean of only 2.9 but maximum values reaching as high as 547, which is almost 1800% greater than the mean. However, as can be seen from the distribution plot, almost all movies have a popularity score less than 10 (the 75th percentile is at 3.678902).


**`Creación de un sistema de Machine Learning`**:

Una vez que todos los datos son accesibles para la interfaz de programación de aplicaciones (API), están listos para ser utilizados por los departamentos de análisis y aprendizaje automático. Mediante nuestro análisis exploratorio de datos (EDA), podemos comprender mejor los datos a los que tenemos acceso. Ahora es el momento de entrenar nuestro modelo de aprendizaje automático para crear un sistema de recomendación de películas.

Para lograr esto, debemos calcular la similitud de puntuación entre una película en particular y el resto de las películas. Luego, se ordenarán según su puntuación de similitud y se generará una lista en Python con los nombres de las cinco películas de mayor puntuación, en orden descendente.

en el sistema de recomendación su usaron las columnas popularity, vote_average, vote_count y como caracteristica adicional la columna de generos para que las recomendaciones tuvieran cierta relacion con la opcion ingresada, despues de organizar las columnas según sus valores utilizamos el algoritmo NearestNeighbors(n_neighbors=6), el cual puede ser traducido como los vecinos mas cercanos a la caracteristica principal que sería popularity y luego las columnas siguientes en las variables que esten mas cercanas al valor dado por la peicula ingresada 

Este sistema debe ser implementado como una función adicional en la API mencionada anteriormente y se llama: `recomendacion`.

**`Video`**: A continuacion se usa un video para mostrar las caracteristicas de la API y el contenido del Repositorio que permiten que el Render este en la web [VIDEO](https://www.youtube.com/shorts/OXGfNxWNFDU)

## **Material de apoyo**

- [links de ayuda](https://github.com/HX-PRomero/PI_ML_OPS/raw/main/Material%20de%20apoyo.md). 
