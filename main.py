import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict

app = FastAPI(title="API de Emisiones CO2")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Función auxiliar para convertir tipos numpy
def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj

# Mapeo de países para normalizar nombres
COUNTRY_MAPPING = {
    "UK/Australia": "UK",
    "UK/Netherlands": "UK",
    "Historic": None,
    "Saudi": "Saudi Arabia"
}

# Coordenadas de países
COUNTRIES_COORDS = {
    "UAE": {"lat": 23.4241, "lng": 53.8478, "code": "AE"},
    "India": {"lat": 20.5937, "lng": 78.9629, "code": "IN"},
    "Indonesia": {"lat": -0.7893, "lng": 113.9213, "code": "ID"},
    "USA": {"lat": 37.0902, "lng": -95.7129, "code": "US"},
    "UK": {"lat": 55.3781, "lng": -3.4360, "code": "GB"},
    "Thailand": {"lat": 15.8700, "lng": 100.9925, "code": "TH"},
    "Bahrain": {"lat": 26.0667, "lng": 50.5577, "code": "BH"},
    "Germany": {"lat": 51.1657, "lng": 10.4515, "code": "DE"},
    "Australia": {"lat": -25.2744, "lng": 133.7751, "code": "AU"},
    "Canada": {"lat": 56.1304, "lng": -106.3468, "code": "CA"},
    "Mexico": {"lat": 23.6345, "lng": -102.5528, "code": "MX"},
    "China": {"lat": 35.8617, "lng": 104.1954, "code": "CN"},
    "Ireland": {"lat": 53.1424, "lng": -7.6921, "code": "IE"},
    "Cyprus": {"lat": 35.1264, "lng": 33.4299, "code": "CY"},
    "Czechia": {"lat": 49.8175, "lng": 15.4730, "code": "CZ"},
    "Russia": {"lat": 61.5240, "lng": 105.3188, "code": "RU"},
    "Colombia": {"lat": 4.5709, "lng": -74.2973, "code": "CO"},
    "Egypt": {"lat": 26.8206, "lng": 30.8025, "code": "EG"},
    "Italy": {"lat": 41.8719, "lng": 12.5674, "code": "IT"},
    "Norway": {"lat": 60.4720, "lng": 8.4689, "code": "NO"},
    "South Africa": {"lat": -30.5595, "lng": 22.9375, "code": "ZA"},
    "Switzerland": {"lat": 46.8182, "lng": 8.2275, "code": "CH"},
    "Japan": {"lat": 36.2048, "lng": 138.2529, "code": "JP"},
    "Iraq": {"lat": 33.2232, "lng": 43.6793, "code": "IQ"},
    "Kazakhstan": {"lat": 48.0196, "lng": 66.9237, "code": "KZ"},
    "Kuwait": {"lat": 29.3759, "lng": 47.9774, "code": "KW"},
    "Libya": {"lat": 26.3351, "lng": 17.2283, "code": "LY"},
    "Ukraine": {"lat": 48.3794, "lng": 31.1656, "code": "UA"},
    "Iran": {"lat": 32.4279, "lng": 53.6880, "code": "IR"},
    "Nigeria": {"lat": 9.0820, "lng": 8.6753, "code": "NG"},
    "Korea": {"lat": 35.9078, "lng": 127.7669, "code": "KR"},
    "Austria": {"lat": 47.5162, "lng": 14.5501, "code": "AT"},
    "Poland": {"lat": 51.9194, "lng": 19.1451, "code": "PL"},
    "Brazil": {"lat": -14.2350, "lng": -51.9253, "code": "BR"},
    "Ecuador": {"lat": -1.8312, "lng": -78.1834, "code": "EC"},
    "Venezuela": {"lat": 6.4238, "lng": -66.5897, "code": "VE"},
    "Oman": {"lat": 21.4735, "lng": 55.9754, "code": "OM"},
    "Malaysia": {"lat": 4.2105, "lng": 101.9758, "code": "MY"},
    "Qatar": {"lat": 25.3548, "lng": 51.1839, "code": "QA"},
    "Spain": {"lat": 40.4637, "lng": -3.7492, "code": "ES"},
    "Saudi Arabia": {"lat": 23.8859, "lng": 45.0792, "code": "SA"},
    "Slovakia": {"lat": 48.6690, "lng": 19.6990, "code": "SK"},
    "Angola": {"lat": -11.2027, "lng": 17.8739, "code": "AO"},
    "Algeria": {"lat": 28.0339, "lng": 1.6596, "code": "DZ"},
    "Syria": {"lat": 34.8021, "lng": 38.9968, "code": "SY"},
    "France": {"lat": 46.2276, "lng": 2.2137, "code": "FR"},
    "Turkmenistan": {"lat": 38.9697, "lng": 59.5563, "code": "TM"},
    "Argentina": {"lat": -38.4161, "lng": -63.6167, "code": "AR"},
    "Netherlands": {"lat": 52.1326, "lng": 5.2913, "code": "NL"}
}

# Cargar datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join("app", "data", "df_co2_countrys.csv")

try:
    df = pd.read_csv(file_path)
    # Normalizar nombres de países
    df['normalized_country'] = df['country'].map(lambda x: COUNTRY_MAPPING.get(x, x))
    # Filtrar países inválidos
    df = df[df['normalized_country'].notna()]
except FileNotFoundError:
    df = pd.DataFrame()
    print(f"Error: Archivo no encontrado en la ruta {file_path}")

@app.get("/")
def read_root():
    """Endpoint principal con información de la API"""
    return {
        "message": "API de Emisiones CO2",
        "version": "1.0",
        "endpoints": {
            "countries": "/countries/list",
            "country_stats": "/stats/country/{country}",
            "emissions_by_sector": "/stats/emissions_by_sector/{country}",
            "historical_data": "/stats/historical/{country}",
            "companies": "/stats/companies/{country}"
        }
    }

@app.get("/countries/list")
def get_countries():
    """Obtener lista de países disponibles con sus coordenadas"""
    if df.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")
    
    available_countries = convert_numpy_types(df['normalized_country'].unique().tolist())
    
    valid_countries = []
    for country in available_countries:
        if country in COUNTRIES_COORDS:
            country_data = COUNTRIES_COORDS[country].copy()
            country_data["name"] = country
            valid_countries.append(country_data)
    
    return {"countries": valid_countries}

@app.get("/stats/country/{country}")
def get_country_stats(country: str):
    """Obtener estadísticas generales de un país"""
    if df.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")
    
    normalized_country = COUNTRY_MAPPING.get(country, country)
    if not normalized_country:
        raise HTTPException(status_code=404, detail=f"País no válido: {country}")
    
    country_data = df[df['normalized_country'] == normalized_country]
    if country_data.empty:
        raise HTTPException(status_code=404, detail=f"No hay datos para {country}")
    
    return {
        "country": normalized_country,
        "total_emissions": float(country_data['total_emissions_MtCO2e'].sum()),
        "average_emissions": float(country_data['total_emissions_MtCO2e'].mean()),
        "total_production": float(country_data['production_value'].sum()),
        "number_of_companies": int(len(country_data['parent_entity'].unique())),
        "number_of_sectors": int(len(country_data['parent_type'].unique())),
        "latest_year": int(country_data['year'].max()),
        "earliest_year": int(country_data['year'].min()),
        "total_commodities": int(len(country_data['commodity'].unique()))
    }

@app.get("/stats/emissions_by_sector/{country}")
def get_emissions_by_sector(country: str):
    """Obtener emisiones por sector para un país"""
    if df.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")
    
    normalized_country = COUNTRY_MAPPING.get(country, country)
    if not normalized_country:
        raise HTTPException(status_code=404, detail=f"País no válido: {country}")
    
    country_data = df[df['normalized_country'] == normalized_country]
    if country_data.empty:
        raise HTTPException(status_code=404, detail=f"No hay datos para {country}")
    
    sector_stats = country_data.groupby('parent_type').agg({
        'total_emissions_MtCO2e': 'sum',
        'production_value': 'sum',
        'parent_entity': 'nunique',
        'commodity': 'nunique'
    }).reset_index()
    
    return {
        "country": normalized_country,
        "sectors": [{
            "name": str(row['parent_type']),
            "total_emissions": float(row['total_emissions_MtCO2e']),
            "total_production": float(row['production_value']),
            "number_of_companies": int(row['parent_entity']),
            "number_of_commodities": int(row['commodity'])
        } for _, row in sector_stats.iterrows()]
    }

@app.get("/stats/historical/{country}")
def get_historical_data(country: str):
    """Obtener datos históricos de emisiones por país"""
    if df.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")
    
    normalized_country = COUNTRY_MAPPING.get(country, country)
    if not normalized_country:
        raise HTTPException(status_code=404, detail=f"País no válido: {country}")
    
    country_data = df[df['normalized_country'] == normalized_country]
    if country_data.empty:
        raise HTTPException(status_code=404, detail=f"No hay datos para {country}")
    
    historical = country_data.groupby('year').agg({
        'total_emissions_MtCO2e': 'sum',
        'production_value': 'sum',
        'parent_entity': 'nunique',
        'parent_type': 'nunique',
        'commodity': 'nunique'
    }).reset_index()
    
    return {
        "country": normalized_country,
        "timeline": [{
            "year": int(row['year']),
            "total_emissions": float(row['total_emissions_MtCO2e']),
            "total_production": float(row['production_value']),
            "number_of_companies": int(row['parent_entity']),
            "number_of_sectors": int(row['parent_type']),
            "number_of_commodities": int(row['commodity'])
        } for _, row in historical.iterrows()]
    }
@app.get("/stats/production/{country}")
def get_production_data(country: str):
    """Obtener los productos principales producidos por un país"""
    if df.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")

    # Normalizar el nombre del país
    normalized_country = COUNTRY_MAPPING.get(country, country)
    if not normalized_country:
        raise HTTPException(status_code=404, detail=f"País no válido: {country}")

    # Filtrar los datos del país
    country_data = df[df['normalized_country'] == normalized_country]
    if country_data.empty:
        raise HTTPException(status_code=404, detail=f"No hay datos para {country}")

    # Agrupar por el tipo de producto y sumar la cantidad producida
    production_stats = country_data.groupby('commodity').agg({
        'production_value': 'sum',
        'production_unit': 'first'
    }).reset_index()

    # Ordenar por valor de producción descendente y limitar a los principales productos
    top_products = production_stats.sort_values(by='production_value', ascending=False).head(5)

    return {
        "country": normalized_country,
        "products": [
            {
                "name": str(row['commodity']),
                "volume": float(row['production_value']),
                "unit": str(row['production_unit'])
            } for _, row in top_products.iterrows()
        ]
    }

@app.get("/stats/companies/{country}")
def get_companies_data(country: str):
    """Obtener información detallada de empresas por país"""
    if df.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")
    
    normalized_country = COUNTRY_MAPPING.get(country, country)
    if not normalized_country:
        raise HTTPException(status_code=404, detail=f"País no válido: {country}")
    
    country_data = df[df['normalized_country'] == normalized_country]
    if country_data.empty:
        raise HTTPException(status_code=404, detail=f"No hay datos para {country}")
    
    # Estadísticas por empresa
    company_stats = country_data.groupby('parent_entity').agg({
        'total_emissions_MtCO2e': 'sum',
        'production_value': 'sum',
        'parent_type': 'first',
        'commodity': lambda x: list(x.unique()),
        'year': lambda x: list(x.unique())
    }).reset_index()
    
    # Calcular estadísticas generales para comparativas
    country_total_emissions = float(country_data['total_emissions_MtCO2e'].sum())
    country_avg_emissions = float(country_data['total_emissions_MtCO2e'].mean())
    
    # Tendencias anuales por empresa
    company_trends = {}
    for company in company_stats['parent_entity']:
        company_data = country_data[country_data['parent_entity'] == company]
        yearly_data = company_data.groupby('year').agg({
            'total_emissions_MtCO2e': 'sum',
            'production_value': 'sum'
        }).reset_index()
        company_trends[company] = [{
            'year': int(row['year']),
            'emissions': float(row['total_emissions_MtCO2e']),
            'production': float(row['production_value'])
        } for _, row in yearly_data.iterrows()]
    
    # Convertir todos los datos a tipos Python nativos
    response_data = {
        "country": normalized_country,
    "summary": {
        "total_companies": int(len(company_stats)),
        "total_country_emissions": float(country_total_emissions),
        "average_company_emissions": float(country_avg_emissions),
        "sectors_represented": int(len(country_data['parent_type'].unique())),
        "years_covered": convert_numpy_types(sorted(country_data['year'].unique().tolist()))
    },
    "companies": [{
        "name": str(row['parent_entity']),
        "sector": str(row['parent_type']),
        "total_emissions": float(row['total_emissions_MtCO2e']),
        "total_production": float(row['production_value']),
        "commodities": convert_numpy_types(row['commodity']),
        "years_active": convert_numpy_types(sorted(row['year'])),
        "emissions_percentage": float(row['total_emissions_MtCO2e'] / country_total_emissions * 100),
        "emissions_vs_average": float(row['total_emissions_MtCO2e'] - country_avg_emissions),
        "historical_data": convert_numpy_types(company_trends[row['parent_entity']]),
        "metrics": {
            "emissions_per_production": float(row['total_emissions_MtCO2e'] / row['production_value']) if row['production_value'] > 0 else 0,
            "commodities_count": int(len(row['commodity'])),
            "years_count": int(len(row['year']))
        }
    } for _, row in company_stats.iterrows()],
    "sector_breakdown": convert_numpy_types(country_data.groupby('parent_type').agg({
        'parent_entity': 'nunique',
        'total_emissions_MtCO2e': 'sum'
    }).reset_index().to_dict('records')),
    "metadata": {
        "last_updated": int(country_data['year'].max()),
        "data_completeness": float(len(country_data) / len(df) * 100),
        "total_records": int(len(country_data))
    }
}
    return response_data
       