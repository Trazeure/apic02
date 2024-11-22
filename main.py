import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time


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
            "companies": "/stats/companies/{country}",
            # Nuevos endpoints
            "production": "/stats/production/{country}",
            "global_stats": "/stats/global",
            "global_emissions_by_sector": "/stats/emissions_by_sector/global",
            "global_historical": "/stats/historical/global",
            "global_companies": "/stats/companies/global",
            "global_production": "/stats/production/global",
            "model_predictions": "/model/predictions"
        },
        "documentation": {
            "global_endpoints": {
                "stats": "Obtiene estadísticas globales agregadas",
                "emissions_by_sector": "Obtiene emisiones globales por sector",
                "historical": "Obtiene datos históricos globales de emisiones",
                "companies": "Obtiene información de las top 10 empresas globalmente",
                "production": "Obtiene los principales productos a nivel global"
            },
            "country_endpoints": {
                "countries_list": "Lista de países disponibles con coordenadas",
                "country_stats": "Estadísticas generales de un país específico",
                "emissions_by_sector": "Emisiones por sector para un país",
                "historical_data": "Datos históricos de emisiones por país",
                "companies": "Información detallada de empresas por país",
                "production": "Productos principales producidos por país"
            },
            "model_endpoints": {
                "predictions": "Predicciones y métricas del modelo de emisiones",
                "country_predictions": "Predicciones y análisis específico por país",
                "simulation": "Simulación de reducción de emisiones por país y commodity"
                
            }
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
@app.get("/stats/global")
def get_global_stats():
    """Obtener estadísticas globales agregadas"""
    if df.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")

    total_emissions = df['total_emissions_MtCO2e'].sum()
    total_production = df['production_value'].sum()
    number_of_companies = df['parent_entity'].nunique()
    number_of_sectors = df['parent_type'].nunique()

    return {
        "total_emissions": float(total_emissions),
        "total_production": float(total_production),
        "number_of_companies": int(number_of_companies),
        "number_of_sectors": int(number_of_sectors)
    }

@app.get("/stats/emissions_by_sector/global")
def get_global_emissions_by_sector():
    """Obtener emisiones globales por sector"""
    if df.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")

    sector_stats = df.groupby('parent_type').agg({
        'total_emissions_MtCO2e': 'sum'
    }).reset_index()

    return {
        "sectors": [
            {
                "name": str(row['parent_type']),
                "total_emissions": float(row['total_emissions_MtCO2e'])
            } for _, row in sector_stats.iterrows()
        ]
    }

@app.get("/stats/historical/global")
def get_global_historical_data():
    """Obtener datos históricos globales de emisiones"""
    if df.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")

    historical = df.groupby('year').agg({
        'total_emissions_MtCO2e': 'sum'
    }).reset_index()

    return {
        "timeline": [
            {
                "year": int(row['year']),
                "total_emissions": float(row['total_emissions_MtCO2e'])
            } for _, row in historical.iterrows()
        ]
    }

@app.get("/stats/companies/global")
def get_global_companies_data():
    """Obtener información detallada de las empresas a nivel global"""
    if df.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")

    company_stats = df.groupby('parent_entity').agg({
        'total_emissions_MtCO2e': 'sum',
        'parent_type': 'first'
    }).reset_index()

    top_companies = company_stats.sort_values(by='total_emissions_MtCO2e', ascending=False).head(10)

    return {
        "companies": [
            {
                "name": str(row['parent_entity']),
                "sector": str(row['parent_type']),
                "total_emissions": float(row['total_emissions_MtCO2e'])
            } for _, row in top_companies.iterrows()
        ]
    }

@app.get("/stats/production/global")
def get_global_production_data():
    """Obtener los principales productos a nivel global"""
    if df.empty:
        raise HTTPException(status_code=500, detail="Datos no disponibles")

    production_stats = df.groupby('commodity').agg({
        'production_value': 'sum',
        'production_unit': 'first'
    }).reset_index()

    top_products = production_stats.sort_values(by='production_value', ascending=False).head(5)

    return {
        "products": [
            {
                "name": str(row['commodity']),
                "volume": float(row['production_value']),
                "unit": str(row['production_unit'])
            } for _, row in top_products.iterrows()
        ]
    }

@app.get("/model/predictions")
def get_model_predictions():
    """Endpoint para obtener predicciones y métricas del modelo"""
    try:
        # Cargar datos
        df = pd.read_csv("app/data/df_co2_countrys.csv")
        
        # Preparar datos para el modelo
        features = ['production_value', 'commodity', 'parent_entity', 'country']
        target = 'total_emissions_MtCO2e'
        
        X = df[features]
        y = df[target]
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocesamiento y modelo
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['production_value']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), 
                 ['commodity', 'parent_entity', 'country'])
            ])
        
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Pipeline
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', rf)
        ])
        
        # Entrenamiento y predicciones
        start_time = time.time()
        model_pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model_pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Predicciones futuras
        df['year'] = pd.to_datetime(df['year'], format='%Y')
        emissions_by_year = df.groupby('year')['total_emissions_MtCO2e'].sum().sort_index()
        
        model = ExponentialSmoothing(
            emissions_by_year,
            seasonal='add',
            seasonal_periods=12,
            trend='add'
        ).fit()
        
        # Generar predicciones para los próximos 5 años
        future_dates = pd.date_range(
            emissions_by_year.index[-1], 
            periods=6, 
            freq='Y'
        )[1:]
        forecast = model.forecast(5)
        
        # Preparar datos históricos
        historical_data = [{
            'year': date.year,
            'emissions': float(value),
            'type': 'historical'
        } for date, value in zip(emissions_by_year.index, emissions_by_year.values)]
        
        # Preparar predicciones
        forecast_data = [{
            'year': date.year,
            'emissions': float(value),
            'type': 'forecast'
        } for date, value in zip(future_dates, forecast)]
        
        return {
            "metrics": {
                "rmse": float(rmse),
                "r2_score": float(r2),
                "training_time": float(training_time)
            },
            "predictions": {
                "historical": historical_data,
                "forecast": forecast_data
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/model/country_predictions/{country}")
def get_country_model_predictions(country: str):
    """Endpoint para obtener predicciones específicas por país"""
    try:
        # Cargar datos
        if df.empty:
            raise HTTPException(status_code=500, detail="Datos no disponibles")
        
        country_data = df[df['country'] == country].copy()
        if country_data.empty:
            raise HTTPException(status_code=404, detail=f"No hay datos para {country}")

        # Entrenar modelo específico para el país
        features = ['production_value', 'commodity', 'parent_entity', 'country']
        target = 'total_emissions_MtCO2e'
        
        X = country_data[features]
        y = country_data[target]
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocesamiento y modelo
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['production_value']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), 
                 ['commodity', 'parent_entity', 'country'])
            ])
        
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Pipeline
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', rf)
        ])
        
        # Entrenamiento y métricas del modelo
        start_time = time.time()
        model_pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model_pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Preparar datos temporales para predicciones
        country_data['year'] = pd.to_datetime(country_data['year'], format='%Y')
        emissions_by_year = country_data.groupby('year')['total_emissions_MtCO2e'].sum().sort_index()
        
        # Modelo de predicción temporal
        model = ExponentialSmoothing(
            emissions_by_year,
            seasonal='add',
            seasonal_periods=12,
            trend='add'
        ).fit()
        
        # Generar predicciones
        periods = 5
        future_dates = pd.date_range(emissions_by_year.index[-1], periods=periods+1, freq='Y')[1:]
        forecast = model.forecast(periods)
        
        # Análisis de tendencia
        recent_values = emissions_by_year.tail(5)
        recent_change = (recent_values.iloc[-1] - recent_values.iloc[0]) / recent_values.iloc[0] * 100
        forecast_change = (forecast[-1] - emissions_by_year.iloc[-1]) / emissions_by_year.iloc[-1] * 100
        
        # Calcular tendencia
        trend = np.polyfit(range(len(recent_values)), recent_values.values, 1)[0]
        annual_change = trend / recent_values.mean() * 100
        
        # Determinar estado
        threshold = 2
        if annual_change < -threshold:
            status = "POSITIVO"
            message = "El país muestra una tendencia positiva de reducción"
            recommendations = [
                "Mantener políticas actuales de reducción",
                "Establecer objetivos más ambiciosos",
                "Compartir mejores prácticas con otros países",
                "Invertir en innovación verde",
                "Monitorear y ajustar estrategias existentes"
            ]
        elif annual_change > threshold:
            status = "ALERTA"
            message = "Se requieren acciones inmediatas para reducir emisiones"
            recommendations = [
                "Implementar medidas urgentes de reducción",
                "Actualizar tecnologías industriales",
                "Acelerar transición a energías limpias",
                "Revisar y fortalecer regulaciones",
                "Considerar incentivos económicos verdes"
            ]
        else:
            status = "ESTABLE"
            message = "Se mantiene estable pero hay espacio para mejoras"
            recommendations = [
                "Identificar oportunidades de mejora",
                "Establecer objetivos más específicos",
                "Mejorar sistemas de monitoreo",
                "Desarrollar planes de transición",
                "Fomentar innovación en tecnologías limpias"
            ]

        # Preparar respuesta
        return {
            "country": country,
            "metrics": {
                "rmse": float(rmse),
                "r2_score": float(r2),
                "training_time": float(training_time)
            },
            "status": {
                "current_state": status,
                "message": message,
                "annual_trend": float(annual_change),
                "recent_change": float(recent_change),
                "forecast_change": float(forecast_change)
            },
            "predictions": {
                "historical": [
                    {
                        "year": date.year,
                        "emissions": float(value)
                    } for date, value in zip(emissions_by_year.index, emissions_by_year.values)
                ],
                "forecast": [
                    {
                        "year": date.year,
                        "emissions": float(value)
                    } for date, value in zip(future_dates, forecast)
                ]
            },
            "recommendations": recommendations,
            "model_info": {
                "total_years": len(emissions_by_year),
                "last_known_year": int(emissions_by_year.index[-1].year),
                "last_known_value": float(emissions_by_year.iloc[-1]),
                "forecast_horizon": periods,
                "data_points": len(country_data),
                "features_used": features
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/model/simulation/{country}/{commodity}")
def get_simulation_predictions(country: str, commodity: str):
    """Endpoint para obtener simulaciones de reducción de emisiones"""
    try:
        if df.empty:
            raise HTTPException(status_code=500, detail="Datos no disponibles")

        # Verificar país y commodity
        country_data = df[(df['country'] == country) & (df['commodity'] == commodity)]
        if country_data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No hay datos para {country} y {commodity}"
            )

        # Crear y entrenar modelo usando la función original
        model, X, metrics = create_simulation_model(df)
        
        # Definir rangos de reducción (igual que en el original)
        reduction_percentages = np.arange(5, 51, 5)
        
        # Ejecutar simulación usando la función original
        results = simulate_reduction(
            df=df,
            model=model,
            X=X,
            country=country,
            commodity=commodity,
            reduction_percentages=reduction_percentages,
            year_target=2030
        )
        
        if results is None:
            raise HTTPException(
                status_code=404,
                detail="No se pudieron generar resultados de simulación"
            )

        # Obtener emisiones base (igual que en el original)
        base_emissions = df[
            (df['country'] == country) & 
            (df['commodity'] == commodity)
        ]['total_emissions_MtCO2e'].iloc[-1]
        
        # Encontrar la mejor reducción
        best_reduction = results.loc[results['reduction_efficiency'].idxmax()]

        # Preparar respuesta manteniendo el formato original de los datos
        return {
            "analysis_results": {
                "country": country,
                "commodity": commodity,
                "current_emissions": float(base_emissions)
            },
            "model_metrics": {
                "rmse": float(metrics['rmse']),
                "mae": float(metrics['mae']),
                "r2_score": float(metrics['r2'])
            },
            "simulation_results": results.to_dict('records'),
            "recommendations": {
                "optimal_reduction": float(best_reduction['reduction_percentage']),
                "emissions_reduction": float(best_reduction['emissions_reduction']),
                "reduction_efficiency": float(best_reduction['reduction_efficiency']),
                "projected_emissions": float(best_reduction['predicted_emissions']),
                "improvement_percentage": float(
                    (best_reduction['emissions_reduction']/base_emissions*100)
                )
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Agregar las funciones originales necesarias
def create_simulation_model(df):
    """Crear y entrenar modelo de simulación"""
    features = ['production_value', 'commodity', 'parent_entity', 'country']
    target = 'total_emissions_MtCO2e'
    
    X = df[features]
    y = df[target]
    
    # División para evaluación
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['production_value']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), 
             ['commodity', 'parent_entity', 'country'])
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Calcular métricas
    y_pred = model.predict(X_test)
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return model, X, metrics

def simulate_reduction(df, model, X, country, commodity, reduction_percentages, year_target=2030):
    """Simular reducciones de producción y su impacto"""
    base_data = df[(df['country'] == country) & (df['commodity'] == commodity)].copy()
    
    if len(base_data) == 0:
        return None
    
    latest_data = base_data.iloc[-1]
    latest_production = latest_data['production_value']
    base_emissions = latest_data['total_emissions_MtCO2e']
    
    results = []
    for reduction in reduction_percentages:
        new_production = latest_production * (1 - reduction/100)
        sim_data = pd.DataFrame([latest_data])
        sim_data['production_value'] = new_production
        
        X_sim = sim_data[X.columns]
        predicted_emissions = model.predict(X_sim)[0]
        
        results.append({
            'reduction_percentage': reduction,
            'new_production': new_production,
            'predicted_emissions': predicted_emissions,
            'emissions_reduction': base_emissions - predicted_emissions,
            'reduction_efficiency': (base_emissions - predicted_emissions) / (reduction/100)
        })
    
    return pd.DataFrame(results)

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
       