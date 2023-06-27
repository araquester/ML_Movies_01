import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from scipy.stats import skew
from math import sqrt
from numpy import mean, var
import copy
from sklearn import preprocessing
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors



from fastapi import FastAPI, Query
app = FastAPI(title='Proyecto Individual 1 - Recomndacion Peliculas',
            description='PI_ML_OPS_01',
            version='1.0.5')

df = pd.read_csv('movies_dataset_Limpio.csv',index_col=0)
df_merged = pd.read_csv('dataset_merged.csv')
df_director = pd.read_csv('df_director.csv')
dfML = pd.read_csv('movies_dataset_Limpio.csv',index_col=0) 



@app.get('/peliculas_mes/{mes}')
async def peliculas_mes(mes:str):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que
    se estrenaron ese mes historicamente
    '''
    
    # volvemos al mes ingresado a minusculas
    mes = mes.lower()
    # Creamos un diccionario de meses
    meses = {
    'enero': 1,
    'febrero': 2,
    'marzo': 3,
    'abril': 4,
    'mayo': 5,
    'junio': 6,
    'julio': 7,
    'agosto': 8,
    'septiembre': 9,
    'octubre': 10,
    'noviembre': 11,
    'diciembre': 12}
    # Extraemos el mes solicitado
    mes_numero = meses[mes]

    # Tratamos el comportamiento cuando no encuentra nada (Excepcion)
    try:
        month_filtered = df[df['release_month'] == mes_numero]
    except (ValueError, KeyError, TypeError):
        return None
    
    # calculamos la cantidad de peliculas pero sin duplicarse
    respuesta = month_filtered.shape[0]

    return {'mes':mes, 'cantidad de peliculas':respuesta}

@app.get('/peliculas_dia{dia}')
async def peliculas_dia(dia:str):
    '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se 
    estrenaron ese dia historicamente (ejemplo: Se ingresa el dia de la semana, 
    en formato cadena, ejemplo 'lunes')'''
    
    # Creamos del diccionario para normalizar los días
    days = {
    'lunes' 	: 0,
    'martes'	: 1,
    'miercoles'	: 2,
    'jueves'	: 3,
    'viernes'	: 4,
    'sabado'	: 5,
    'domingo'	: 6}

    day = days[dia.lower()]
    lista_peliculas_dia = df[df['release_day'] == day]
    respuesta = lista_peliculas_dia.shape[0]
    return {'día':dia, 'cantidad de peliculas':respuesta}

@app.get('/score_titulo/{titulo}')
async def score_titulo(titulo:str):
    '''Se ingresa el título de una filmación esperando como respuesta el 
    título, el año de estreno y el score'''

    # Convertimos la entrada del usuario en minusculas
    pelicula = titulo.lower()

    # Filtramos el Dataframe con la película solicitada volviendo el contenido a minúsculas
    info_pelicula = df[df['title'].str.lower() == pelicula].drop_duplicates(subset='title')

    '''
    A partir de la informacion de la pelicula cogemos los valores en las columnas y los
    asignamos a las variables
    '''
    nombre_pelicula = info_pelicula['title'].iloc[0]
    year_pelicula = str(info_pelicula['release_year'].iloc[0])
    score_pelicula =  str(info_pelicula['popularity'].iloc[0])

    return {'titulo':nombre_pelicula, 'año':year_pelicula, 'popularidad':score_pelicula}

@app.get('/votos_titulo/{titulo}')
async def votos_titulo(titulo:str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos
    y el valor promedio de las votaciones.
    
    La misma variable deberá de contar con al menos 2000 valoraciones, caso contrario,
    debemos contar con un mensaje avisando que no cumple esta condición y que por ende,
    no se devuelve ningun valor.'''

# Convertimos la entrada del usuario en minusculas
    pelicula = titulo.lower()

# Filtramos el Dataframe con la película solicitada volviendo el contenido a minúsculas
    info_pelicula = df[df['title'].str.lower() == pelicula].drop_duplicates(subset='title')

# A partir de la informacion de la pelicula asignamos las variables que necesitamos
    votos_pelicula = str(info_pelicula['vote_count'].iloc[0])
    year_pelicula = str(info_pelicula['release_year'].iloc[0])
    promedio_pelicula =  str(info_pelicula['vote_average'].iloc[0])

    if (float(votos_pelicula)>2000):
        return {
        'Titulo': titulo, 'año':year_pelicula, 'total votos':votos_pelicula, 'valoracion promedio':promedio_pelicula}
    else:
        # return {'La Pelicula no tiene mas de 2000 votos'}
        return None
    
@app.get('/get_actor/{actor}')
async def get_actor(nombre_actor):
    actor_films = df_merged[df_merged['cast'].str.contains(nombre_actor, case=False, na=False)]
    cantidad_peliculas = len(actor_films)
    
    # Inicializar el éxito del actor y el total de 'return'
    exito_actor = 0
    total_return = 0
    
    # Calcular el éxito del actor y el total de 'return'
    for _, row in actor_films.iterrows():
        budget = row['budget']
        revenue = row['revenue']
        if revenue != 0:
            return_value = budget / revenue
            exito_actor += return_value
            total_return += 1
    
    # Calcular el promedio de 'return'
    promedio_return = exito_actor / total_return if total_return != 0 else 0

    return {'actor': nombre_actor, 'peliculas': int(cantidad_peliculas),'retorno promedio':float(round(promedio_return,2)),'exito':float(round(exito_actor,2))}

@app.get("/get_director/{nombre_director}")
async def get_director( nombre_director: str ):
    '''
    Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver
    el éxito del mismo medido a través del retorno. Además, deberá devolver el nombre de cada película
    con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.
    '''
    # convierto la variable a minúsculas para independizarme de la forma en que el usuario escribe el nombre
    director = nombre_director.lower()

    #Filtro el DF por el nombre que el usuario puso
    df_director_filtrado = df_director[df_director['crew'].str.lower() == director]
    
    #Sumo el total_return
    dir_total_return = df_director_filtrado['return'].mean()
    
    #asignamos las variables
    titulos = df_director_filtrado['title'].to_list()
    fechas_estreno = df_director_filtrado["release_date"].to_list()
    presupuesto = df_director_filtrado['budget'].to_list()
    ganancia = df_director_filtrado['revenue'].to_list()
    pelis_json = [ {'titulo': e1,'fecha estreno':e2,'costo':e3,'ganancia':e4} for e1,e2,e3,e4 in zip (titulos,fechas_estreno,presupuesto,ganancia) ]
    return {'director': nombre_director,
            'exito': float(dir_total_return),
            'peliculas':pelis_json}

@app.get("/recomendacion/{titulo}")
async def recomendacion(titulo):
    
    
     # Obtener el índice de la película que coincide con el título ingresado (case insensitive)
    indices = pd.Series(dfML.index, index=dfML['title'].str.lower()).drop_duplicates()
    titulo_lower = titulo.lower()
    
    if titulo_lower not in indices:
        return "Título no encontrado"
    
    idx = indices[titulo_lower]
    
    # Seleccionar las características relevantes para el algoritmo KNN
    features = ['popularity', 'vote_average', 'vote_count', 'genres']
    X = dfML[features]
    
    # Convertir las listas en columnas binarias utilizando MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    encoded_genres = pd.DataFrame(mlb.fit_transform(X['genres']), columns=mlb.classes_, index=X.index)
    
    # Convertir los nombres de características numéricas a cadenas
    numeric_features = X.drop('genres', axis=1)
    numeric_features.columns = numeric_features.columns.astype(str)
    
    # Aplicar PCA para reducir la dimensionalidad
    pca = PCA(n_components=0.9)  # Mantenemos el 90% de la varianza explicada
    encoded_genres_pca = pca.fit_transform(encoded_genres)
    
    # Concatenar las características numéricas y las características transformadas por PCA
    X = pd.concat([numeric_features, pd.DataFrame(encoded_genres_pca)], axis=1)
    
    # Normalizar las características
    X = (X - X.mean()) / X.std()
    
    # Crear una instancia del algoritmo KNN
    knn = NearestNeighbors(n_neighbors=6)  # Consideramos 6 vecinos, incluyendo la película seleccionada
    
    # Resolvemos el siguiente error con la recomendacion de la propia libreria:  
    # Feature names are only supported if all input features have string names, but your input has ['int', 'str'] 
    # as feature name / column name types. If you want feature names to be stored and validated, 
    # you must convert them all to strings, by using: =>>>>> X.columns = X.columns.astype(str)
    X.columns = X.columns.astype(str)
    
    # Entrenar el modelo KNN
    knn.fit(X)
    
    # Obtener la distancia y los índices de los vecinos más cercanos
    distances, indices = knn.kneighbors(X.iloc[idx].values.reshape(1, -1))
    
    # Obtener los títulos de las películas más similares (excluyendo la película seleccionada)
    similar_movies = dfML['title'].iloc[indices[0][1:6]]
    
    return similar_movies
	