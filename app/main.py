import os
import pandas as pd
from fastapi import FastAPI

# Definir la ruta relativa para el archivo CSV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "df_co2_countrys.csv")

# Intentar cargar el archivo CSV
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    df = pd.DataFrame()
    print(f"Error: Archivo no encontrado en la ruta {file_path}")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de emisiones de CO₂"}

@app.get("/data")
def get_all_data():
    """
    Retorna todos los datos del CSV
    """
    if df.empty:
        return {"error": "El archivo CSV no se cargó correctamente"}
    return df.to_dict(orient="records")

@app.get("/data/filter")
def filter_data(
    year: int = None,
    parent_entity: str = None,
    parent_type: str = None,
    commodity: str = None,
    country: str = None
):
    """
    Filtrar los datos por cualquier columna disponible
    """
    if df.empty:
        return {"error": "El archivo CSV no se cargó correctamente"}
    
    filtered_df = df
    
    filters = {
        'year': year,
        'parent_entity': parent_entity,
        'parent_type': parent_type,
        'commodity': commodity,
        'country': country
    }
    
    for column, value in filters.items():
        if value is not None:
            filtered_df = filtered_df[filtered_df[column] == value]
    
    return filtered_df.to_dict(orient="records")

@app.get("/stats/by_parent_entity")
def stats_by_parent_entity():
    """
    Estadísticas agrupadas por parent_entity
    """
    if df.empty:
        return {"error": "El archivo CSV no se cargó correctamente"}
    
    stats = df.groupby("parent_entity").agg({
        'production_value': 'sum',
        'total_emissions_MtCO2e': 'sum',
        'commodity': 'count'
    }).reset_index()
    
    return stats.to_dict(orient="records")

@app.get("/stats/by_parent_type")
def stats_by_parent_type():
    """
    Estadísticas agrupadas por parent_type
    """
    if df.empty:
        return {"error": "El archivo CSV no se cargó correctamente"}
    
    stats = df.groupby("parent_type").agg({
        'production_value': 'sum',
        'total_emissions_MtCO2e': 'sum',
        'commodity': 'count'
    }).reset_index()
    
    return stats.to_dict(orient="records")

@app.get("/stats/annual")
def annual_stats():
    """
    Estadísticas anuales de emisiones y producción
    """
    if df.empty:
        return {"error": "El archivo CSV no se cargó correctamente"}
    
    annual = df.groupby("year").agg({
        'production_value': 'sum',
        'total_emissions_MtCO2e': 'sum',
        'log_total_emissions_MtCO2e': 'mean'
    }).reset_index()
    
    return annual.to_dict(orient="records")

@app.get("/stats/emissions_by_commodity")
def emissions_by_commodity():
    """
    Análisis detallado de emisiones por commodity
    """
    if df.empty:
        return {"error": "El archivo CSV no se cargó correctamente"}
    
    emissions = df.groupby("commodity").agg({
        'total_emissions_MtCO2e': 'sum',
        'production_value': 'sum',
        'production_unit': 'first'
    }).reset_index()
    
    emissions['emissions_per_unit'] = emissions['total_emissions_MtCO2e'] / emissions['production_value']
    
    return emissions.to_dict(orient="records")

@app.get("/stats/country_time_series")
def country_time_series(country: str = None):
    """
    Series temporal de emisiones por país
    """
    if df.empty:
        return {"error": "El archivo CSV no se cargó correctamente"}
    
    if country:
        country_df = df[df['country'] == country]
    else:
        country_df = df
    
    time_series = country_df.groupby(['country', 'year']).agg({
        'total_emissions_MtCO2e': 'sum',
        'production_value': 'sum',
        'commodity': 'count'
    }).reset_index()
    
    return time_series.to_dict(orient="records")

@app.get("/stats/total_emissions_by_year")
def total_emissions_by_year():
    """
    Emisiones totales por año para todos los países
    """
    if df.empty:
        return {"error": "El archivo CSV no se cargó correctamente"}
    
    yearly_emissions = df.groupby('year').agg({
        'total_emissions_MtCO2e': 'sum',
        'log_total_emissions_MtCO2e': 'mean'
    }).reset_index()
    
    return yearly_emissions.to_dict(orient="records")