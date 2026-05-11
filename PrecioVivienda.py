import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. Cargar y Preprocesar los Datos
# ---------------------------------

# Cargar los datos
datos = pd.read_csv('housing.csv')

# Eliminar los datos faltantes
datos.dropna(inplace=True)

# Convertir la variable categórica a numérica
def convert_ocean_proximity_to_numeric(data):
    """
    Converts the 'ocean_proximity' column in the given DataFrame to numeric dummy variables.

    This function uses pandas' get_dummies method to convert the categorical 'ocean_proximity'
    column into multiple binary columns, each representing a category. The original 'ocean_proximity'
    column is then dropped from the DataFrame, and the new dummy variable columns are joined to it.

    Parameters:
    data (pandas.DataFrame): The input DataFrame containing the 'ocean_proximity' column.

    Returns:
    pandas.DataFrame: The DataFrame with the 'ocean_proximity' column replaced by its numeric dummy variables.
    """
    ocean_proximity_numeric = pd.get_dummies(data.ocean_proximity, dtype=int)
    data.drop('ocean_proximity', axis=1, inplace=True)
    return data.join(ocean_proximity_numeric)

# Convertir la variable categórica a numérica
datos = convert_ocean_proximity_to_numeric(datos)

# Crear nuevas características
def add_features(data):
    """
    Adds new features to the housing data and removes some existing ones.

    This function calculates and adds the following features to the input DataFrame:
    - 'rooms_per_household': The ratio of total rooms to households.
    - 'bedrooms_per_room': The ratio of total bedrooms to total rooms.

    It also removes the following columns from the DataFrame:
    - 'total_rooms'
    - 'total_bedrooms'
    - 'households'

    Parameters:
    data (pandas.DataFrame): The input DataFrame containing housing data with columns 
                             'total_rooms', 'total_bedrooms', and 'households'.

    Returns:
    pandas.DataFrame: The modified DataFrame with the new features added and specified columns removed.
    """
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data.drop(['total_rooms', 'total_bedrooms', 'households'], axis=1, inplace=True)
    return data

# Añadir nuevas características a los datos
datos = add_features(datos)

# 2. Dividir los Datos en Entrenamiento y Prueba
# ----------------------------------------------

# Dividir los datos en variables dependientes e independientes
X = datos.drop('median_house_value', axis=1)
y = datos['median_house_value']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar las primeras filas de los datos de entrenamiento
print(X_train.head())
print(y_train.head())

# 3. Escalar los Datos
# --------------------

# Escalar los datos
def scale_data(data):
    """
    Scales the input data using StandardScaler.

    Parameters:
    data (array-like): The data to be scaled.

    Returns:
    array-like: The scaled data.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Escalar los datos de entrenamiento y prueba
X_train = scale_data(X_train)
X_test = scale_data(X_test)

# 4. Crear y Entrenar el Modelo
# -----------------------------

# Crear y entrenar el modelo
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)

# 5. Predecir y Evaluar el Modelo
# -------------------------------

# Predecir y evaluar el modelo
y_pred = linear_regression_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

y_pred_rf = random_forest_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

# Imprimir el puntaje R^2
print(f"R^2 Score (Linear Regresion): {r2}")
print(f"R^2 Score (Random Forest): {r2_rf}")

