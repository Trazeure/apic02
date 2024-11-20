from fastapi import FastAPI, Query
import pandas as pd

# Cargar el archivo CSV
file_path = r"C:\Users\Desti\OneDrive\Escritorio\api_c02\data\df_co2_countrys.csv"
df = pd.read_csv(file_path)

# Crear la aplicación FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de emisiones de CO₂"}

@app.get("/data")
def get_all_data():
    """
    Retorna todos los datos del CSV
    """
    return df.to_dict(orient="records")

@app.get("/data/filter")
def filter_data(
    country: str = Query(None, description="Filtrar por país"),
    year: int = Query(None, description="Filtrar por año"),
    commodity: str = Query(None, description="Filtrar por commodity (ejemplo: Oil & NGL, Natural Gas)")
):
    """
    Filtrar los datos por país, año y/o tipo de commodity
    """
    filtered_df = df
    if country:
        filtered_df = filtered_df[filtered_df["country"] == country]
    if year:
        filtered_df = filtered_df[filtered_df["year"] == year]
    if commodity:
        filtered_df = filtered_df[filtered_df["commodity"] == commodity]
    
    return filtered_df.to_dict(orient="records")

@app.get("/emissions/total")
def total_emissions_by_country():
    """
    Obtener las emisiones totales por país
    """
    total_emissions = df.groupby("country")["total_emissions_MtCO2e"].sum().reset_index()
    return total_emissions.to_dict(orient="records")

@app.get("/emissions/commodity")
def emissions_by_commodity():
    """
    Obtener las emisiones totales por tipo de commodity
    """
    emissions_by_commodity = df.groupby("commodity")["total_emissions_MtCO2e"].sum().reset_index()
    return emissions_by_commodity.to_dict(orient="records")
