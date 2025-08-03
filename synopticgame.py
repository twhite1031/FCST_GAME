import pandas as pd
import math
from datetime import datetime, timedelta
import requests
from io import StringIO
import re
import geopandas as gpd
from shapely.geometry import Point
from haversine import haversine, Unit
import os
import numpy as np

# Change names to match id number to real forecaster, you can add more forecasters
id_to_forecaster = {1:("Greene", "human"),2:("Lamsma", "human"),3:("Caldon", "human"),4:("Tolsma", "human"),
                    5:("Gryskewicz", "human"),6:("Thirlwall", "human"),7:("Knudsen", "human"),8:("Farrell", "human"),
                    9:("Biedron", "human"),10:("Crane", "human"),11:("Schiller", "human"),12:("Ragland", "human"), 13: ("Camille", "human"),
                    14: ("Ruiz", "human"), 15: ("Steiger", "human"), 16: ("Starr", "human"), 17: ("SREF", "model"), 18: ("NWS", "model")}

chosen_states = ["California", "Oregon", "Washington", "Idaho", "Montana"]
fcst_city = "Oswego"
# Get Stations that exceeded threshold (hundreths of inch) and the times you would like to look at for ASOS precip data
precip_threshold = 50 # Threshold in hundreths of an inch

# Storing dictionaries
forecasts = {}  # Dictionary to store forecasts keyed by file name
error_data = {}  # Dictionary to store error data keyed by file name
consensus_data = {} # Dictionary to store consensus data keyed by file name
flood_data = {}  # Dictionary to store flood game data keyed by file name

# Constant for wind calculations
pie18 = (math.pi)/18.
angle = 180/(math.pi)

# Helper function to calculate u and v components from wind direction and speed
def calculate_uv_components(direction, speed_knots):
    arg = (27 - direction) * pie18
    speed_mps = speed_knots / 1.94
    u = speed_mps * math.cos(arg)
    v = speed_mps * math.sin(arg)
    return u, v

# Function to read the forecast file and add u/v wind components
def read_forecast_file(forecast_file):
    column_names = ["Number_id", "ProbFL", "Lat", "Lon", "High_T", "Low_T",
            "Wind_dir", "Wind_speed", "Nil_p", "Lght_p", "Mod_p",
            "High_p", "Amount_p"
        ]
    fcst = pd.read_csv(
        forecast_file,
        delimiter=r'\s*,\s*|\s+',
        engine='python',
        header=None,
        names=column_names
    )

    # Set Number_id as the index for easier access later
    fcst.set_index("Number_id", inplace=True)

    # Add u and v wind components
    if not fcst.empty:
        uv = fcst.apply(lambda forecaster_row: calculate_uv_components(forecaster_row['Wind_dir'], forecaster_row['Wind_speed']), axis=1)
        fcst[['u_wind', 'v_wind']] = pd.DataFrame(uv.tolist(), index=fcst.index)

    return fcst

# Function to read the forecast verification file and add u/v wind components
def read_forecast_ver_file(forecast_ver_file, forecast_files):
    column_names = [
            "ProbFL", "High_T", "Low_T", 
            "Wind_dir", "Wind_speed", "Nil_p", "Lght_p", "Mod_p", "High_p", "Amount_p"
        ]
    fcst = pd.read_csv(
        forecast_ver_file,
        delimiter=r'\s*,\s*|\s+',
        engine='python',
        header=None,
        skiprows=2,
        names=column_names
    )

    # Add u and v wind components
    if not fcst.empty:
        uv = fcst.apply(lambda forecaster_row: calculate_uv_components(forecaster_row['Wind_dir'], forecaster_row['Wind_speed']), axis=1)
        fcst[['u_wind', 'v_wind']] = pd.DataFrame(uv.tolist(), index=fcst.index)

    # Index by forecast_files (truncate if necessary)
    fcst.index = forecast_files[:len(fcst)]

    # Reindex to force exact forecast_files order (with potential NaNs if any are missing)
    fcst = fcst.reindex(forecast_files)

    return fcst

# Function to check what forecast files exist in the directory
def check_forecast_files(directory='.'):
    """
    Check the specified directory for forecast files and return a list of valid files.
    Defaults to the current working directory if no directory is provided.

    Parameters:
        directory (str): The path to the directory containing forecast files.

    Returns:
        list: A list of valid forecast file names found in the directory.
    """
    expected_order = ["fcst.m24", "fcst.m48", "fcst.t24", "fcst.t48", "fcst.w24", "fcst.w48", "fcst.r24", "fcst.r48"]
    try:
        existing_files = set(os.listdir(directory))
        forecast_files = [fname for fname in expected_order if fname in existing_files]
        return forecast_files

    except FileNotFoundError:
        print(f"Directory {directory} not found.")
        return []
    
def check_missed_forecasts(forecast_files, forecasts, id_to_forecaster):

    missed_game_status = {}
    total_missed = 0

    for file in forecast_files:
        missed_game_status[file] = {}

        df = forecasts[file]
        if df.empty:
            continue

        for forecaster_id, (name, ftype) in id_to_forecaster.items():
            try:
                forecaster_row = df.loc[forecaster_id]
            except KeyError:
                continue

            #print(f"{name} data stored for {file}")

            missed = forecaster_row["ProbFL"] > 10
            if missed:
                total_missed += 1
                #print(f"{name} has missed the forecast game, flood probability was greater than 10")

            missed_game_status[file][forecaster_id] = missed

    return missed_game_status, total_missed



# Helper function to adjust precipitation probabilities to ensure they sum to 10
def adjust_precip_probs(row):
    precip_cols = ["Nil_p", "Lght_p", "Mod_p", "High_p"]
    total = row[precip_cols].sum()

    if total == 10:
        return row
    elif total > 10.0:
        for col in precip_cols:
            row[col] = (10.0 * row[col]) / total
    else:
        row["Nil_p"] = 10.0
        row["Lght_p"] = 0.0
        row["Mod_p"] = 0.0
        row["High_p"] = 0.0
    return row

# Used in consensus/error calculation to determine how many models are present
num_of_models = sum(1 for id_num, (name, ftype) in id_to_forecaster.items() if ftype == "model")

# Function to clean/process forecasts and calculate consensus data for an individual file (day)
def process_consensus_day(forecast_file, forecasts, fcst_ver, missed_game_status, id_to_forecaster):
    """
    Processes forecasts for a given day (forecast_file) and returns:
    - processed_data_day: {forecaster_id: updated Series}
    - consensus: Series of averaged forecast values
    """
    forecasts_df = forecasts[forecast_file]  # DataFrame for this file
    consensus = pd.Series(0.0, index=forecasts_df.columns)

    # Count missed forecasts
    num_missed = sum(1 for missed in missed_game_status[forecast_file].values() if missed)
    total_human_forecasters = len(id_to_forecaster) - num_of_models
    num_human_active_fcsts = total_human_forecasters - num_missed
    num_human_active_fcsts_flood = num_human_active_fcsts
    
    for forecaster_id, (name, ftype) in id_to_forecaster.items():
        try:
            forecaster_row = forecasts_df.loc[forecaster_id].copy()
        except KeyError:
            print(f"[!] No forecast for ID {forecaster_id} ({name}) in {forecast_file}")
            continue

        # Ensure longitude is negative
        if forecaster_row["Lon"] > 0:
            forecaster_row["Lon"] = -forecaster_row["Lon"]

 
        # Explicitly cast only modified fields
        forecaster_row["Lon"] = float(forecaster_row["Lon"])
        forecaster_row["Amount_p"] = float(forecaster_row["Amount_p"])

        # Normalize precip probabilities if not missed
        if not missed_game_status[forecast_file][forecaster_id]:
            forecaster_row = adjust_precip_probs(forecaster_row)
        # Set updated row safely 

        for col in ["Lon", "Amount_p", "Nil_p", "Lght_p", "Mod_p", "High_p"]:
            forecasts_df.at[forecaster_id, col] = forecaster_row[col]

        # Add to consensus if it's a human forecast and not missed
        if ftype == "human" and not missed_game_status[forecast_file][forecaster_id]:
            
            consensus["ProbFL"] += forecaster_row["ProbFL"]

            if forecaster_row["Lat"] > 0 and forecaster_row["Lon"] < 0:
                consensus["Lat"] += forecaster_row["Lat"]
                consensus["Lon"] += forecaster_row["Lon"]
            else:
                num_human_active_fcsts_flood -= 1

            for var in ["High_T", "Low_T", "Wind_speed", "Nil_p", "Lght_p",
                        "Mod_p", "High_p", "Amount_p", "u_wind", "v_wind"]:
                consensus[var] += forecaster_row[var]

    # Final consensus averaging
    for col in consensus.index:
        if col in ["Lat", "Lon"]:
            consensus[col] /= max(num_human_active_fcsts_flood, 1)
        else:
            consensus[col] /= max(num_human_active_fcsts, 1)

    # Calculate consensus wind direction using atan2 for correctness
    u, v = consensus["u_wind"], consensus["v_wind"]
    if u == 0 and v == 0:
        consensus["Wind_dir"] = np.nan  # Calm wind
    else:
        wind_rad = math.atan2(v, u)
        consensus["Wind_dir"] = (270 - angle * wind_rad) % 360  # Meteorological convention

    return consensus

def get_flood_times(today=None):
    """
    Returns list of (forecast_label, list of valid times), where:
    - .24 = 6Z and 12Z on the base day
    - .48 = 18Z (same day), 0Z, 6Z, 12Z on next day
    """
    if today is None:
        today = datetime.utcnow()

    # Find most recent Monday at 00Z
    days_since_monday = today.weekday()  # Monday = 0
    monday = datetime(today.year, today.month, today.day) - timedelta(days=days_since_monday)
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)

    # Labels: m = Tue, t = Wed, w = Thu, r = Fri
    labels = ["m", "t", "w", "r"]
    expected_order = []

    for i, label in enumerate(labels):
        base_day = monday + timedelta(days=i + 1)       # m = Tue, ..., r = Fri
        next_day = base_day + timedelta(days=1)         # used for .48

        # .24 → 6Z, 12Z on base_day
        times_24 = [
            base_day + timedelta(hours=6),
            base_day + timedelta(hours=12)
        ]
        expected_order.append((f"fcst.{label}24", times_24))

        # .48 → 18Z on base_day, 0Z, 6Z, 12Z on next_day
        times_48 = [
            base_day + timedelta(hours=18),
            next_day + timedelta(hours=0),
            next_day + timedelta(hours=6),
            next_day + timedelta(hours=12)
        ]
        expected_order.append((f"fcst.{label}48", times_48))

    return expected_order

# Function to retrieve station data, returns all stations that exceed the threshold and lat/lon
def get_asos_precipitation(threshold, start_date, end_date):
    """
    Retrieves ASOS station observations with 6-group codes above a threshold,
    returning a list of dictionaries containing station metadata and precip amount.
    """
    base_url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
    params = {
        'year1': start_date.year, 'month1': start_date.month, 'day1': start_date.day, 'hour1': start_date.hour, 'minute1': start_date.minute,
        'year2': end_date.year, 'month2': end_date.month, 'day2': end_date.day, 'hour2': end_date.hour, 'minute2': end_date.minute,
        'tz': 'UTC',
        'format': 'csv',
        'data': 'metar,',
        'latlon': True
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code}, Could not retrieve ASOS data")
        return []

    # Parse CSV
    column_names = ['Station_ID', 'Datetime', 'lon', 'lat', 'metar']
    data = StringIO(response.text)
    df = pd.read_csv(data, names=column_names, header=5, on_bad_lines='skip')
    df.replace('M', pd.NA, inplace=True)

    results = []

    for idx, value in enumerate(df["metar"]):
        pattern = r'\s6(\d{4})\s'
        match = re.search(pattern, value)

        if match:
            six_group = int(match.group(1))

            if threshold < six_group < 3000:
                obs = {
                    "station_id": df.loc[idx, 'Station_ID'],
                    "lat": df.loc[idx, 'lat'],
                    "lon": df.loc[idx, 'lon'],
                    "amount": six_group,
                    "datetime": df.loc[idx, 'Datetime']
                }
                results.append(obs)

    return results

def verify_flood_game(observations, chosen_states, shapefile_path='state_shapefiles/cb_2018_us_state_500k.shp'):
    # Load and filter the shapefile
    states = gpd.read_file(shapefile_path)
    chosen_states_gdf = states[states['NAME'].isin(chosen_states)]

    # Check each observation
    verified_obs = []
    for obs in observations:
        point = Point(obs["lon"], obs["lat"])
        if any(chosen_states_gdf.contains(point)):
            verified_obs.append(obs)

    return verified_obs

# Values for precip error weight calculation
scale = 0
max_tracker = {
    "50_100": 0,
    "100_150": 0,
    "150_200": 0,
    "200_250": 0,
    "250_above": 0
}

# This finds the closest station to your lat/lon and then applies the scale based on the precip the station recieved
def calc_flood_error(lat_fcst, lon_fcst, flood_data, max_tracker, file):

    if not flood_data[file]["verified_observations"]:
        print("[!] No verified observations found for flood game.")
        return None

    distance, best_obs = find_closest_station(lat_fcst, lon_fcst, flood_data[file]["verified_observations"])
    amount_precip = best_obs["amount"]

    #print(f"Precipitation amount at closest station: {amount_precip} hundredths of an inch")

    # Update max_tracker
    if 50 < amount_precip < 100 and amount_precip > max_tracker["50_100"]:
        max_tracker["50_100"] = amount_precip
    elif 100 < amount_precip < 150 and amount_precip > max_tracker["100_150"]:
        max_tracker["100_150"] = amount_precip
    elif 150 < amount_precip < 200 and amount_precip > max_tracker["150_200"]:
        max_tracker["150_200"] = amount_precip
    elif 200 < amount_precip < 250 and amount_precip > max_tracker["200_250"]:
        max_tracker["200_250"] = amount_precip
    elif amount_precip >= 250 and amount_precip > max_tracker["250_above"]:
        max_tracker["250_above"] = amount_precip

    # Determine scaling factor
    if max_tracker["250_above"] > 0:
        scale = 30
    elif max_tracker["200_250"] > 0:
        scale = 25
    elif max_tracker["150_200"] > 0:
        scale = 20
    elif max_tracker["100_150"] > 0:
        scale = 15
    elif max_tracker["50_100"] > 0:
        scale = 10
    else:
        scale = 1

    return distance / scale

# Finds the shortest_distance to each verified station using the Haversine formula
# Find closest station from verified_obs
def find_closest_station(lat, lon, obs_list):
    min_dist = float('inf')
    best_station = None
    for obs in obs_list:
        dist = haversine((lat, lon), (obs["lat"], obs["lon"]), unit=Unit.KILOMETERS)
        if dist < min_dist:
            min_dist = dist
            best_station = obs

    return min_dist, best_station

# Computes the error for the consensus data against the verification data for all forecast files
def compute_consensus_error(consensus_file, forecasts_ver,flood_data):
    """
    Computes error metrics for each forecast day using pandas,
    returning a DataFrame where rows are forecast files (e.g., 'fcst.m24').

    Parameters:
        consensus_file (dict): Keys are filenames; values are Series of consensus values.
        forecasts_ver (pd.DataFrame): Verification data, indexed by forecast file.

    Returns:
        pd.DataFrame: DataFrame of error metrics indexed by forecast file name.
    """
    error_records = []

    for file, consensus in consensus_file.items():
        
        try:
            verification = forecasts_ver.loc[file]
        except KeyError:
            print(f"[!] Verification data not found for {file}")
            continue

        # Skip row if consensus data is missing required columns
        if "ProbFL" not in consensus or "Lat" not in consensus:
            print(f"[!] Missing required fields in consensus data for {file}")
            continue

        # Prepare error row as a dict
        error_row = {"File": file}

        # 1. Flood probability error
        error_row["ProbFL"] = (consensus["ProbFL"] - verification["ProbFL"]) ** 2

        # 2. Flood location (can be used later in distance calc)
        error_row["Lat"] = consensus["Lat"]
        error_row["Lon"] = consensus["Lon"]

        if flood_data[file]["flood_game_verify"]:
            error_row["FloodLoc"] = calc_flood_error(consensus["Lat"], consensus["Lon"], flood_data, max_tracker, file)
        else:
            error_row["FloodLoc"] = 0
        

        # 3. High/Low Temp error
        error_row["High_T"] = abs(consensus["High_T"] - verification["High_T"])
        error_row["Low_T"] = abs(consensus["Low_T"] - verification["Low_T"])

        # 4. Wind vector error (u/v)
        u_diff = consensus["u_wind"] - verification["u_wind"]
        v_diff = consensus["v_wind"] - verification["v_wind"]
        error_row["Wind"] = np.sqrt(u_diff ** 2 + v_diff ** 2)

        # 5. Precip amount error (in inches)
        error_row["Precip_Amount"] = abs(consensus["Amount_p"] - verification["Amount_p"])

        # 6. Precip category error (cumulative probabilities)
        f1 = consensus["Nil_p"]
        f2 = f1 + consensus["Lght_p"]
        f3 = f2 + consensus["Mod_p"]
        v1 = verification["Nil_p"]
        v2 = v1 + verification["Lght_p"]
        v3 = v2 + verification["Mod_p"]
        error_row["Precip_Cat"] = (f1 - v1) ** 2 + (f2 - v2) ** 2 + (f3 - v3) ** 2

        # Add the error row to the records list
        error_records.append(error_row)

    # Convert list of dicts to DataFrame and set file as index
    error_df = pd.DataFrame(error_records).set_index("File")
    return error_df

penal = 1.5 # Penalty for missed forecasts
locpen = 1.4 # Location penalty for flood location error

# All the error variables we will be calculating and using
error_vars = ["ProbFL", "FloodLoc", "High_T", "Low_T", "Wind", "Precip_Amount", "Precip_Cat"]

# Function to process the error points for an individual file
def calc_error_points(file, forecasts, fcst_ver,flood_data,missed_game_status,id_to_forecaster):
    error_data = {}
    verification_data = fcst_ver.loc[file] # Forecaster verification data for this file
    for forecaster_id, (name, ftype) in id_to_forecaster.items():
        error_series = pd.Series(0.0, index=error_vars)
        
        try:
            missed = missed_game_status[file][forecaster_id]
        except KeyError:
            print(f"[!] Forecaster ID {forecaster_id} missing from missed_game_status[{file}]")
            continue

        if not missed:
            
            try:
                current_data = forecasts[file].loc[forecaster_id]
            except KeyError:
                print(f"[!] Missing data for {name} in {file}")
                continue

            # 1. Flood probability error
            error_series["ProbFL"] = (current_data["ProbFL"] - verification_data["ProbFL"]) ** 2
            # 2. Flood location (can be used later in distance calc)
            if flood_data[file]["flood_game_verify"]:
                if current_data["Lat"] == 0 or current_data["Lon"] == 0:
                    error_series["FloodLoc"] = locpen * consensus_error_data.loc[file]["ProbFL"]
                else:
                    error_series["FloodLoc"] = calc_flood_error(current_data["Lat"], current_data["Lon"], flood_data, max_tracker, file)
            else:
                error_series["FloodLoc"] = 0

            # 3. High/Low Temp error
            error_series["High_T"] = abs(current_data["High_T"] - verification_data["High_T"])
            error_series["Low_T"] = abs(current_data["Low_T"] - verification_data["Low_T"])

            # 4. Wind vector error (u/v)
            u_err = current_data["u_wind"] - verification_data["u_wind"]
            v_err = current_data["v_wind"] - verification_data["v_wind"]
            error_series["Wind"] = math.sqrt(u_err ** 2 + v_err ** 2)

            # 5. Precip amount error (in inches)
            error_series["Precip_Amount"] = abs(current_data["Amount_p"] - verification_data["Amount_p"])

            # 6. Precip category error (cumulative probabilities)
            f1 = current_data["Nil_p"]
            f2 = f1 + current_data["Lght_p"]
            f3 = f2 + current_data["Mod_p"]

            v1 = verification_data["Nil_p"]
            v2 = v1 + verification_data["Lght_p"]
            v3 = v2 + verification_data["Mod_p"]

            a = (f1 - v1) ** 2
            b = (f2 - v2) ** 2
            c = (f3 - v3) ** 2

            error_series["Precip_Cat"] = a + b + c

        else:
            # Apply penalties for missed forecasts
            for var in error_vars:
                error_series[var] = penal * consensus_error_data.loc[file, var]
                
        error_data[forecaster_id] = error_series.copy()

    return error_data

# Constant weights
pscx = 1.0
psn = .15
psx = 1.0
erday = 25.0
localfloodscalex = 4
localfloodscalen = .25
locprbscalex = 4
locprbscalen = .25
gush = 1.25
gust = .30
gulp = .80
gulf = .5

# === Precipitation Probability Scaling ===
def prbsc(con_p_prob_error, con_p_amount_error, gush, psn, psx):
    if con_p_prob_error < 0.05:
        prbsc = 0
    else:
        prbsc = gush * (con_p_amount_error / con_p_prob_error)
    prbsc = max(psn, min(prbsc, psx))
    print(f"[Scale] Precip Prob Scaling (prbsc): {prbsc}")
    return prbsc

# === Precip vs Temperature Scaling ===
def presc(con_temp_error, con_p_amount_error, con_p_prob_error, prbsc, pscx):
    scaled_con_p_prob_error = prbsc * con_p_prob_error
    con_p_total_error = scaled_con_p_prob_error + con_p_amount_error
    print("ErPrCn", con_p_total_error)
    if con_p_total_error < 0.05:
        presc = 0
    else:
        presc = con_temp_error / con_p_total_error
    presc = min(presc, pscx)
    print(f"[Scale] Precip vs Temp Scaling (presc): {presc}")
    return presc

# === Wind vs Temperature Scaling ===
def windsc(con_temp_error, con_wind_error, gust):
    if con_wind_error < 0.05:
        windsc = 0
    else:
        windsc = gust * (con_temp_error / con_wind_error)
    print(f"[Scale] Wind vs Temp Scaling (windsc): {windsc}")
    return windsc

# === Flood Location vs Probability Scaling ===
def locprbsc(con_flood_loc_error, con_flood_prob_error, gulp, locprbscalex, locprbscalen):
    if con_flood_prob_error < 0.05:
        locprbscale = 0
    else:
        locprbscale = gulp * (con_flood_loc_error / con_flood_prob_error)

    # Clamp the scaling value between locprbscalen and locprbscalex
    locprbscale = max(locprbscalen, min(locprbscale, locprbscalex))

    print(f"[Scale] Flood Loc vs Prob Scaling (locprbscale): {locprbscale}")
    return locprbscale

# === Local Game vs Flood Game Scaling ===
def locfloodsc(con_local_error, con_flood_error, gulf, localfloodscalen, localfloodscalex):

    if con_flood_loc_error <= 0:
        localfloodscale = 1
    else:
        localfloodscale = gulf * (con_local_error / con_flood_error)
        localfloodscale = max(localfloodscalen, min(localfloodscale, localfloodscalex))
    print(f"[Scale] Local vs Flood Scaling (localfloodscale): {localfloodscale}")
    return localfloodscale

# === Weekly Total Scaling ===
def weekly_total_scaling(numday, erday, con_total_week_error):
    total_con_error= sum(con_total_week_error) # All errors summed up
    print(erday)
    erweek = numday * erday
    weeksc = erweek / total_con_error if total_con_error != 0 else 0
    print(f"[Scale] Weekly Total Scaling (weeksc): {weeksc}")
    return weeksc

# === Apply Scaling to Each Forecaster ===
def apply_first_scaling(con_total_week_error, week_error_df, id_to_forecaster,locprbsc, locfloodsc, prbsc, presc, windsc):
    
    # Consensus scaling
    con_total_week_error["ProbFL"] *= locprbsc 
    con_total_week_error["ProbFL"] *= locfloodsc 
    con_total_week_error["FloodLoc"] *= locfloodsc
    con_total_week_error["Precip_Cat"] *= prbsc * presc
    con_total_week_error["Precip_Amount"] *= presc
    con_total_week_error["Wind"] *= windsc

    # Forecaster scaling
    for forecaster_id in id_to_forecaster:
        current_error_data = week_error_df.loc[forecaster_id].copy()

        current_error_data["ProbFL"] *= locprbsc
        current_error_data["ProbFL"] *= locfloodsc
        current_error_data["FloodLoc"] *= locfloodsc   
        current_error_data["Precip_Cat"] *= prbsc * presc
        current_error_data["Precip_Amount"] *= presc
        current_error_data["Wind"] *= windsc
        
        week_error_df.loc[forecaster_id] = current_error_data

def apply_second_scaling(con_total_week_error, week_error_df, id_to_forecaster, numday, erday):
    con_total_error = sum(con_total_week_error)
    erweek = numday * erday
    weeksc = erweek / con_total_error if con_total_error != 0 else 0

    
    for forecaster_id in id_to_forecaster:
        current_error_data = week_error_df.loc[forecaster_id].copy()
        for var in current_error_data.index:
            current_error_data[var] *= weeksc
        week_error_df.loc[forecaster_id] = current_error_data

    for var in con_total_week_error.index:
        con_total_week_error[var] *= weeksc

    return weeksc

if __name__ == "__main__":

    print("Starting synoptic game processing...")

    # Step 1: Load forecast files
    forecast_files = check_forecast_files()
    fcst_ver = read_forecast_ver_file("fcst.ver", forecast_files)

    numday = 0
    forecasts = {}

    # Step 2: Read individual forecast files
    for file in forecast_files:
        try:
            forecasts[file] = read_forecast_file(file)
            print(f"[✓] Successfully read {file}")
            numday += 1
        except FileNotFoundError:
            print(f"[!] {file} not found. Make sure it exists in the current directory.")
            forecasts[file] = pd.DataFrame()
        except Exception as e:
            print(f"[X] Error reading {file}: {e}")
            forecasts[file] = pd.DataFrame()

    # Step 3: Check for missed forecasts
    missed_game_status, num_missed_fcst = check_missed_forecasts(
        forecast_files, forecasts, id_to_forecaster
    )

    # Step 4: Compute consensus for each forecast day
    consensus_data = {}
    for file in forecast_files:
        consensus_data[file] = process_consensus_day(
            file, forecasts, fcst_ver, missed_game_status, id_to_forecaster
        )

    # Step 5: Run Flood Game verification
    flood_game_times = dict(get_flood_times(today=datetime(2024, 11, 20, 12, 0)))
    print("Expected flood game times for this week:", flood_game_times)

    flood_data = {}
    for file in forecast_files:
        flood_data[file] = {"verified_observations": []}

        for time in flood_game_times[file]:
            # Use ±8-minute window to simulate synoptic obs
            start = time - timedelta(minutes=8)
            end = time + timedelta(minutes=8)

            observations = get_asos_precipitation(precip_threshold, start, end)

            verified_observations = verify_flood_game(
                observations,
                chosen_states,
                shapefile_path='state_shapefiles/cb_2018_us_state_500k.shp'
            )

            flood_data[file]["verified_observations"].extend(verified_observations)

        flood_data[file]["flood_game_verify"] = len(flood_data[file]["verified_observations"]) > 0
        print(f"[✓] Flood game verification complete: {len(flood_data[file]['verified_observations'])} valid obs for {file}")

    # Step 6: Compute consensus error
    consensus_error_data = compute_consensus_error(
        consensus_data, fcst_ver, flood_data
    )

    # Step 7: Compute error points for each forecaster
    error_data = {}
    for file in forecast_files:
        error_data[file] = calc_error_points(
            file, forecasts, fcst_ver, flood_data, missed_game_status, id_to_forecaster
        )
    # Sum up all errors for the week
    week_error_df = pd.DataFrame(columns=error_vars, dtype=float)
    con_total_week_error = pd.Series(0.0, index=error_vars)

    # Loop over files and accumulate error
    for file in forecast_files:
        # Convert list to Series for readable operations
        con_daily_errors = pd.Series(consensus_error_data.loc[file], index=error_vars)
        con_total_week_error += con_daily_errors

        for forecaster_id in id_to_forecaster:
            try:
                daily_errors = pd.Series(error_data[file][forecaster_id], index=error_vars)  # Grab consensus errors for the day

                if forecaster_id in week_error_df.index:
                    week_error_df.loc[forecaster_id] += daily_errors # Sum each variable consensus errors that day
                else:
                    week_error_df.loc[forecaster_id] = daily_errors

            except KeyError as e:
                print(f"[!] No error data for forecaster ID {forecaster_id} on {file}")
                print(f"Error: {e}")

    # Add weights to each error using consensus for the week
    con_p_amount_error     = con_total_week_error["Precip_Amount"] # ErAmCn
    con_p_prob_error       = con_total_week_error["Precip_Cat"]
    con_temp_error         = con_total_week_error["High_T"] + con_total_week_error["Low_T"]
    con_wind_error         = con_total_week_error["Wind"]
    con_flood_prob_error   = con_total_week_error["ProbFL"]
    con_flood_loc_error    = con_total_week_error["FloodLoc"]
    con_local_error = con_p_amount_error + con_temp_error + con_wind_error + con_p_amount_error
    con_flood_error = con_flood_prob_error + con_flood_loc_error

    print("ErAmCn", con_p_amount_error)
    print("ErPbCn", con_p_prob_error)
    print("ErTmpCn", con_temp_error)
    print("ErWndCn", con_wind_error)
    print("ErPFLPrCn", con_flood_prob_error)
    print("ErLFLCn", con_flood_loc_error)

    # Create scaling factors to the consensus errors
    prbsc_scale = prbsc(con_p_prob_error, con_p_amount_error, gush, psn, psx)
    presc_scale = presc(con_temp_error, con_p_amount_error, con_p_prob_error, prbsc_scale, pscx)
    windsc_scale = windsc(con_temp_error, con_wind_error, gust)
    locprbsc_scale = locprbsc(con_flood_loc_error, con_flood_prob_error, gulp, locprbscalex, locprbscalen)
    locfloodsc_scale = locfloodsc(con_local_error, con_flood_error, gulf, localfloodscalen, localfloodscalex)
    #weekly_scales = weekly_total_scaling(numday, erday, con_total_week_error)

    # Apply scaling to each forecaster's error data
    weekly_scales = apply_first_scaling(con_total_week_error,week_error_df, id_to_forecaster, locprbsc_scale, locfloodsc_scale, prbsc_scale, presc_scale, windsc_scale)
    weeksc = apply_second_scaling(con_total_week_error, week_error_df, id_to_forecaster, numday, erday)
    

    # Error data for each forecaster is stored in error_data_{file} for easy reading
    for file in forecast_files:
        try:
            df = pd.DataFrame.from_dict(error_data[file]).T # Transpose for names to be the rows
            df.index = df.index.map(lambda idx: id_to_forecaster.get(idx, ("Unknown",))[0])  # map index to name
            df.to_csv(f"raw_error_data_{file}.csv")
            df = pd.DataFrame.from_dict(consensus_data[file])
            df.to_csv(f"raw_consensus_data_{file}.csv")
            df = pd.DataFrame.from_dict(forecasts[file])
            df.index = df.index.map(lambda idx: id_to_forecaster.get(idx, ("Unknown",))[0]) # map index to name
            df.to_csv(f"forecasts_{file}.csv")
        except KeyError:
            print(f"[!] No error data found for {file}")


    consensus_error_data.to_csv("consensus_error_scores.csv")

    # Save weekly weighted forecaster scores with forecaster names
    week_error_df.index = week_error_df.index.map(
        lambda idx: id_to_forecaster.get(idx, ("Unknown",))[0]
    )
    week_error_df.to_csv("weighted_weekly_error_scores.csv")