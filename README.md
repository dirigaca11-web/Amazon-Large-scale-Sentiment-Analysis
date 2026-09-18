
# Amazon-Large-scale-Sentiment-Analysis

Final project for the Building AI course


## Summary

Development of an end-to-end sentiment classification system capable of processing 35 million of records through a LightGBM model with a F1 metric score of 0.86 .The project encompasses everything from SQL queries and Big Data management with high-performance tools to the deployment of an interactive web application in the cloud. Building AI course project


## Background

I downloaded the dataset “Amazon Review Polarity Dataset” from Kaggle, which contains reviews from 6,643,669 users on 2,441,053 products, which contains reviews from 6,643,669 users on 2,441,053 products, along with a polarity variable (1 for negative and 2 for positive) that we want to predict

Analyzing customer feedback on massive platforms like Amazon presents two critical challenges, the first one is the data volume, review datasets often exceed the capacity of conventional RAM, causing standard tools like Pandas to fail.

The second one is the response speed, companies need to classify thousands of reviews in real time to react to product issues or improve the user experience.

You can read more about this dataset in this link: amazon reviews

## How to use it

You can try it yourself by entering this link: https://data-science-portfolio-b7bsgyqcct24vvzxphfgsh.streamlit.app/


## Approach

In order to complete the project the next steps were performed:

1. Convert the massive dataset into a Data Base File.
2. Use SQL for data cleaning and ED, directly in the database engine to optimize performance.
3. Process the results, create a NLP pipeline and develop the model.
4. Develop a web-based application using Streamlit that allows users to write reviews and receive an analysis of them.





## Data sources and AI methods
Data downladed form kaggle
Convertion to db:
```
import polars as pl
import sqlite3

#creamos funcion para convertir csv a db
def csv_to_sqlite(csv_file, table_name):
    print(f"Procesando {csv_file}...")
    # leemos con polars
    df = pl.read_csv(csv_file, has_header=False, new_columns=["polarity", "title", "text"])
    
    # conectamos y guardamos
    conn = sqlite3.connect("amazon_reviews.db")
    df.to_pandas().to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Tabla '{table_name}' creada con éxito")

# ejecutamos para ambos conjuntos entrenamiento y prueba
csv_to_sqlite("train.csv", "train")
csv_to_sqlite("test.csv", "test")
```


Main code:
```
import polars as pl
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import f1_score
import joblib

# Usamos Scan (Modo Lazy) - No carga nada en RAM aún
print("Iniciando motor Polars...")
query = pl.scan_csv("train.csv", has_header=False, new_columns=["polarity", "title", "text"])

# Pipeline de limpieza (Se define el plan, no se ejecuta todavía)
procesamiento = (
    query
    .drop_nulls() 
    .with_columns([
        #ponemos en minusculas y eliminamos caracteres raros
        # Limpiamos y renombramos de una vez para evitar el error de duplicados
        pl.col("text").str.to_lowercase().str.replace_all(r"[^a-zA-Z\s]", "").alias("text_clean"),
        pl.col("title").str.to_lowercase().str.replace_all(r"[^a-zA-Z\s]", "").alias("title_clean")
    ])
    .with_columns([
        # Combinamos las versiones YA limpias
        (pl.col("title_clean") + ", "+ pl.col("text_clean")).alias("full_content")
    ])
    .select(["polarity", "full_content"])
)

# ejecutamos el procesamiento
print("Limpiando 3.6 millones de filas... observa la velocidad.")
df_final = procesamiento.collect() 

print(f"Dataset listo con {len(df_final)} filas.")
print(df_final.head())



print("Cargando muestra de datos...")
df_sample = df_final.sample(n=500000, seed=42)
# Configuramos el HashingVectorizer con features (262,144 columnas virtuales)
vectorizador = HashingVectorizer(n_features=2**18, alternate_sign=False)

print("Convirtiendo texto a matriz numérica")
features = vectorizador.transform(df_sample["full_content"])
target = df_sample["polarity"]
#Dividir en Entrenamiento y Validación (80/20)
features_train, features_val, target_train, target_val = train_test_split(features, target, test_size=0.2, random_state=42)
print(f"Matriz de características creada con forma: {features_train.shape}")


model_lgbm = lgb.LGBMClassifier( objective='binary', random_state=12345)
model_lgbm.fit(features_train, target_train )
#evaluamos el modelo lgbm
predictions_lgbm = model_lgbm.predict(features_val)
score_lgbm=f1_score(target_val, predictions_lgbm)
print(f"Valor F1 para modelo LightGBM: {score_lgbm} ")

#guardamos el modelo y el vectorizador
joblib.dump(model_lgbm, 'modelo_amazon.pkl')
joblib.dump(vectorizador, 'vectorizador.pkl')
print("Modelo y Vectorizador guardados con éxito.")


```



## Conclusions


The development of this sentiment analysis model successfully demonstrated the ability to process massive datasets while maintaining high predictive accuracy and computational efficiency. By leveraging a modern data stack consisting of Polars and LightGBM, the project achieved a robust F1-Score of 0.86 on the final test set (indicating a perfect balance in learning).

The model shows high symmetry in its ability to classify both positive and negative reviews, utilizing HashingVectorizer was the key to optimize memory usage without sacrificing performance.

This project forms a good machine learning implementation, showcasing not only strong technical metrics but also a professional approach to memory management, feature engineering, app creating and model validation in a Big Data environment.

