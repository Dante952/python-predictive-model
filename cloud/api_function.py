import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask import jsonify

try:
    print("Loading models and historical data...")
    temperature_model = joblib.load('regional_model_temperature.joblib')
    rain_model = joblib.load('regional_model_rain.joblib')
    wind_u_model = joblib.load('regional_model_wind_u.joblib')
    wind_v_model = joblib.load('regional_model_wind_v.joblib')
    historical_means = pd.read_pickle('historical_means_regional.pkl')
    print(" Models loaded successfully.")
except FileNotFoundError as e:
    print(f" Critical error loading files: {e}")

def get_closest_historical_data(lat, lon, day_of_year, hour):
    try:
        subset = historical_means.loc[(day_of_year, hour)]
        if subset.empty:
            raise KeyError
        subset = subset.reset_index()
        distances = (subset['lat'] - lat)**2 + (subset['lon'] - lon)**2
        closest_index = distances.idxmin()
        return subset.iloc[closest_index]
    except KeyError:
        temp_df = historical_means.reset_index()
        distances = (temp_df['lat'] - lat)**2 + (temp_df['lon'] - lon)**2 + (temp_df['dayofyear'] - day_of_year)**2
        closest_index = distances.idxmin()
        return temp_df.iloc[closest_index]

def convert_wind_to_speed_and_direction(u, v):
    speed_mps = np.sqrt(u**2 + v**2)
    speed_kmh = speed_mps * 3.6
    direction_degrees = (270 - np.rad2deg(np.arctan2(v, u))) % 360
    directions = ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"]
    index = int(round(direction_degrees / 45))
    cardinal_direction = directions[index % 8]
    return speed_kmh, direction_degrees, cardinal_direction

def create_forecast(lat, lon, timestamp_string):
    try:
        date_object = datetime.fromisoformat(timestamp_string)
        day_of_year = date_object.timetuple().tm_yday
        hour = date_object.hour
        
        original_features = ["TLML", "PRECTOTCORR", "QLML", "ULML", "VLML"]
        
        base_features = get_closest_historical_data(lat, lon, day_of_year, hour)
        prediction_features_full = pd.DataFrame([base_features])
        prediction_features = prediction_features_full[original_features]
        
        temperature_prediction = temperature_model.predict(prediction_features.drop('TLML', axis=1))[0]
        rain_probability_prediction = rain_model.predict_proba(prediction_features.drop('PRECTOTCORR', axis=1))
        wind_u_prediction = wind_u_model.predict(prediction_features.drop('ULML', axis=1))[0]
        wind_v_prediction = wind_v_model.predict(prediction_features.drop('VLML', axis=1))[0]

        if rain_probability_prediction.shape[1] > 1:
            rain_probability = rain_probability_prediction[0, 1]
        else:
            rain_probability = 1.0 if rain_model.classes_[0] == 1 else 0.0
        
        wind_speed_kmh, direction_degrees, cardinal_direction = convert_wind_to_speed_and_direction(wind_u_prediction, wind_v_prediction)
        temperature_celsius = temperature_prediction - 273.15

        return {
            "success": True,
            "forecast": {
                "location": {"lat": lat, "lon": lon},
                "timestamp_utc": date_object.strftime('%Y-%m-%d %H:%M:%S'),
                "temperature_celsius": float(f"{temperature_celsius:.1f}"),
                "rain_probability": float(f"{rain_probability:.2f}"),
                "wind": {
                    "speed_kmh": float(f"{wind_speed_kmh:.1f}"),
                    "direction": cardinal_direction,
                    "degrees": float(f"{direction_degrees:.0f}")
                }
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_forecast_api(request):
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return (jsonify({"success": False, "error": "Invalid JSON"}), 400, headers)
        
        lat = request_json['lat']
        lon = request_json['lon']
        timestamp_string = request_json['timestamp']

        result = create_forecast(lat, lon, timestamp_string)
        
        status_code = 200 if result['success'] else 500
        return (jsonify(result), status_code, headers)

    except KeyError as e:
        return (jsonify({"success": False, "error": f"Missing key in JSON payload: {str(e)}"}), 400, headers)
    except Exception as e:
        return (jsonify({"success": False, "error": f"An internal error occurred: {str(e)}"}), 500, headers)