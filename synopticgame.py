import pandas as pd
import math
from datetime import datetime
import requests
from io import StringIO
import re
import geopandas as gpd
from shapely.geometry import Point
from haversine import haversine, Unit
import os
import numpy as np
from collections import defaultdict

# Change names to match id number to real forecaster, you can add more forecasters
id_to_forecaster = {1:("Thomas", "human"),2:("Bob", "human"),3:("Jeff", "human"),4:("Tim", "human"),
                    5:("Tom", "human"),6:("Wang", "human"),7:("Steiger", "human"),8:("Pete", "human"),
                    9:("Ruan", "human"),10:("Sand", "human"),11:("Bobby", "human"),12:("Nat", "human"), 13: ("Tash", "human"),
                    14: ("Dave", "human"), 15: ("Tyler", "human"), 16: ("Yong", "human"), 17: ("NWS", "model"), 18: ("SREF", "model")}

chosen_states = ["California", "Oregon", "Washington", "Idaho", "Montana"]
fcst_city = "Oswego"

# Get Stations that exceeded threshold (hundreths of inch) and the times you would like to look at for ASOS precip data
precip_threshold = 50 # Threshold in hundreths of an inch
start_date, end_date = datetime(2024, 11, 20, 11,52), datetime(2024,11,20,11,59)

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
   
forecast_files = check_forecast_files()  # Check current directory for forecast files
fcst_ver = read_forecast_ver_file("fcst.ver", forecast_files) # Reading forecast verification file

forecasts = {} # Dictionary to store forecast data keyed by file name
numday = 0 # Counter used for scaling error values, changes based on number of days processed

# Read each forecast file and store in a dictionary for processing
for file in forecast_files:
    try:
        forecasts[file] = read_forecast_file(file)  # Store data keyed by file name, reading individually
        print(f"[✓] Successfully read {file}")
        numday += 1

    except FileNotFoundError:
        print(f"[!] {file} not found. Make sure it exists in the current directory.")
        forecasts[file] = pd.DataFrame()  # Use empty placeholder for downstream logic

    except Exception as e:
        print(f"[X] Error reading {file}: {e}")
        forecasts[file] = pd.DataFrame()

error_data = {}  # Dictionary to store error data keyed by file name
consensus_data = {}

# Dictionary to store each proccessed forecaster for a day
processed_data = {}

# Values for precip error weight calculation
scale = 0
max_tracker = {
    "50_100": 0,
    "100_150": 0,
    "150_200": 0,
    "200_250": 0,
    "250_above": 0
}

def check_missed_forecasts(forecast_files, forecasts, id_to_forecaster):
    """
    Check which forecasters missed the game for each forecast file based on ProbFL > 10.
    
    Parameters:
        forecast_files (list): List of forecast file names.
        forecasts (dict): Dictionary of DataFrames keyed by forecast file.
        id_to_forecaster (dict): Dictionary mapping forecaster_id to (name, type).

    Returns:
        dict: missed_game_status[file][forecaster_id] = True/False
        int: Total number of missed forecasts across all files
    """
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

            print(f"{name} data stored for {file}")

            missed = forecaster_row["ProbFL"] > 10
            if missed:
                total_missed += 1
                print(f"{name} has missed the forecast game, flood probability was greater than 10")

            missed_game_status[file][forecaster_id] = missed

    return missed_game_status, total_missed

missed_game_status, num_missed_fcst = check_missed_forecasts(forecast_files, forecasts, id_to_forecaster)

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

# Function check lat lon is valid for game
def is_within_chosen_states(lat_lon_tuple, chosen_states_gdf):
    point = Point(lat_lon_tuple[1], lat_lon_tuple[0])  # Note that Point expects (lon, lat)
    return any(chosen_states_gdf.contains(point))


# Function to retrieve station data
def get_asos_precipitation(threshold, start_date, end_date):
    # Array to store tuples of verified lat/lon for stations
    asos_lat_lon = []
    asos_amount = []
   
    base_url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
    params = {
        'year1': start_date.year, 'month1': start_date.month, 'day1': start_date.day, 'hour1': start_date.hour, 'minute1': start_date.minute, # Start date
        'year2': end_date.year, 'month2': end_date.month, 'day2': end_date.day, 'hour2': end_date.hour,  'minute2':end_date.minute, # End date
        'tz': 'UTC',  # Timezone
        'format': 'csv',  # Format of the returned data
        'data': 'metar,',  # Precipitation data
        'latlon' : True,
           }
  
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code}, Could not retrieve ASOS data")
        return False, None

    # Parse the CSV data
    column_names = ['Station_ID', 'Datetime', 'lon', 'lat', 'metar']
    data = StringIO(response.text)
    df = pd.read_csv(data, names=column_names, header=5, on_bad_lines='skip')
    df.replace('M', pd.NA, inplace=True)

    

    for idx, value in enumerate(df["metar"]):
        pattern = r'\s6(\d{4})\s'
        match = re.search(pattern, value)

        if match:
            six_group = int(match.group(1))

            if threshold < six_group < 3000:
                station_id = df.loc[idx, 'Station_ID']
                lat = df.loc[idx, 'lat']
                lon = df.loc[idx, 'lon']

                asos_lat_lon.append((lat, lon))
                asos_amount.append(six_group)

    return asos_lat_lon, asos_amount

# Run function to find Flood game verification at a specific time period
asos_lat_lon, asos_amount = get_asos_precipitation(precip_threshold, start_date, end_date)
print(f"asos_lat_lon {asos_lat_lon}")
print(f"asos_amount: {asos_amount}")
states = gpd.read_file('state_shapefiles/cb_2018_us_state_500k.shp')
# Filter the states GeoDataFrame to only include the chosen states
chosen_states_gdf = states[states['NAME'].isin(chosen_states)]
# Filter the list of tuples
verified_asos_idx_lat_lon = [(idx, lat_lon) for idx, lat_lon in enumerate(asos_lat_lon) if is_within_chosen_states(lat_lon, chosen_states_gdf)]

# Extract the indices and values separately if needed
verified_indices = [item[0] for item in verified_asos_idx_lat_lon]
verified_lat_lon = [item[1] for item in verified_asos_idx_lat_lon]

print(f"Verified ASOS stations latlon: {verified_lat_lon}")
print(f"Verified ASOS stations indices: {verified_indices}")
# Determine if flood game is verifiable based on filtered station data
flood_game_verify = len(verified_asos_idx_lat_lon) > 0

if flood_game_verify:
    print("✅ Flood game verification PASSED: At least one station meets criteria within chosen states.")
    # Optionally show verified stations
    for i, (idx, latlon) in enumerate(verified_asos_idx_lat_lon):
        print(f"  Station {i+1}: Index={idx}, Lat/Lon={latlon}, Precip={asos_amount[idx]}")
else:
    print("❌ Flood game verification FAILED: No valid stations found in chosen states.")

# This finds the closest station to your lat/lon and then applies the scale based on the precip the station recieved
def calc_flood_error(lat_fcst, lon_fcst, filtered_lat_lon, asos_amount, max_tracker):
    distance, station_coords = find_shortest_distance(lat_fcst, lon_fcst, filtered_lat_lon)
    coord_idx = asos_lat_lon.index(station_coords) # Find index of the closest station in the original asos_lat_lon list
    
    #print(f"Closest station to forecast location ({lat_fcst}, {lon_fcst}) is at {station_coords} with distance {distance:.2f} km")
    if coord_idx is None:
        return 0

    amount_precip = asos_amount[coord_idx]
    print(f"Precipitation amount at closest station: {amount_precip} hundreths of an inch")

    # Find the maximum precipitation amount in the tracker for scaling, rewarded for higher amounts
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

    # Determine scale based on highest bin that has a value
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
        scale = 1  # Default
    
    return distance / scale

# Finds the shortest_distance to each verified station using the Haversine formula
def find_shortest_distance(lat, lon, coordinates):
    
    # Convert the reference point to a tuple
    reference_point = (lat, lon)

    # Initialize the minimum distance with a large number
    min_distance = float('inf')
    closest_coord = None

    # Iterate over each coordinate tuple in the list
    for idx, coord in enumerate(coordinates):
        # Calculate the distance using haversine
        distance = haversine(reference_point, coord, unit=Unit.KILOMETERS)
        
        # Update the minimum distance if a shorter distance is found
        if distance < min_distance:
            min_distance = distance
            closest_coord = coord
                      
    return min_distance, closest_coord

# Computes the error for the consensus data against the verification data for all forecast files
def compute_consensus_error(consensus_file, forecasts_ver):
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
        print(file)
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

        if flood_game_verify:
            error_row["FloodLoc"] = calc_flood_error(consensus["Lat"], consensus["Lon"], verified_lat_lon, asos_amount, max_tracker)
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
def calc_error_points(file, forecasts, fcst_ver,missed_game_status,id_to_forecaster):
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
            if flood_game_verify:
                if current_data["Lat"] == 0 or current_data["Lon"] == 0:
                    error_series["FloodLoc"] = locpen * consensus_error_data.loc[file]["ProbFL"]
                else:
                    error_series["FloodLoc"] = calc_flood_error(current_data["Lat"], current_data["Lon"], verified_lat_lon, asos_amount, max_tracker)
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
        
# Loop through each forecast file (day) and compute consensus data
for idx, file in enumerate(forecast_files):
    consensus_data[file] = process_consensus_day(file, forecasts, fcst_ver, missed_game_status, id_to_forecaster) 

consensus_error_data = compute_consensus_error(consensus_data,fcst_ver)

for idx, file in enumerate(forecast_files):
    error_data[file] = calc_error_points(file, forecasts, fcst_ver, missed_game_status, id_to_forecaster)  # Calculate error points for each forecaster 

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

# Constant weights
pscx = 1.0
psn = .15
psx = 1.0
erday = 25.0
fmin = 80
localfloodscalex = 4
localfloodscalen = .25
locprbscalex = 4
locprbscalen = .25
gush = 1.25
gust = .30
gulp = .80
gulf = .5

# Add weights to each error using consensus for the week
con_p_amount_error     = con_total_week_error["Precip_Amount"] # ErAmCn
con_p_prob_error       = con_total_week_error["Precip_Cat"]
con_temp_error         = con_total_week_error["High_T"] + con_total_week_error["Low_T"]
con_wind_error         = con_total_week_error["Wind"]
con_flood_prob_error   = con_total_week_error["ProbFL"]
con_flood_loc_error    = con_total_week_error["FloodLoc"]
print("ErAmCn", con_p_amount_error)
print("ErPbCn", con_p_prob_error)
print("ErTmpCn", con_temp_error)
print("ErWndCn", con_wind_error)
print("ErPFLPrCn", con_flood_prob_error)
print("ErLFLCn", con_flood_loc_error)

'''
Issues
ErPrCn: Precipitation Category Error
presc: Precipitation vs Temperature Scaling
ErLFLCn: Flood Location Error
ErPFLPrCn: Flood Probability Error
weeksc: Weekly Total Scaling

'''
# === Precipitation Probability Scaling ===
if con_p_prob_error < 0.05:
    prbsc = 0
else:
    prbsc = gush * (con_p_amount_error / con_p_prob_error)
prbsc = max(psn, min(prbsc, psx))
print(f"[Scale] Precip Prob Scaling (prbsc): {prbsc}")

# === Precip vs Temperature Scaling ===
scaled_con_p_prob_error = prbsc * con_p_prob_error
con_p_total_error = scaled_con_p_prob_error + con_p_amount_error
print("ErPrCn", con_p_total_error)
if con_p_total_error < 0.05:
    presc = 0
else:
    presc = con_temp_error / con_p_total_error
presc = min(presc, pscx)
print(f"[Scale] Precip vs Temp Scaling (presc): {presc}")

# === Wind vs Temperature Scaling ===
if con_wind_error < 0.05:
    windsc = 0
else:
    windsc = gust * (con_temp_error / con_wind_error)
print(f"[Scale] Wind vs Temp Scaling (windsc): {windsc}")

# === Flood Location vs Probability Scaling ===
if con_flood_prob_error < 0.05:
    locprbscale = 0
else:
    locprbscale = gulp * (con_flood_loc_error / con_flood_prob_error)
print(f"[Scale] Flood Loc vs Prob Scaling (locprbscale): {locprbscale}")

# === Local Game vs Flood Game Scaling ===
con_local_error = con_p_amount_error + con_p_prob_error + con_temp_error + con_wind_error
con_flood_error = con_flood_prob_error + con_flood_loc_error

if con_flood_loc_error <= 0:
    localfloodscale = 1
else:
    localfloodscale = gulf * (con_local_error / con_flood_error)
    localfloodscale = max(localfloodscalen, min(localfloodscale, localfloodscalex))
print(f"[Scale] Local vs Flood Scaling (localfloodscale): {localfloodscale}")

# === Weekly Total Scaling ===
total_con_error = sum(con_total_week_error)
erweek = numday * erday
weeksc = erweek / total_con_error if total_con_error != 0 else 0
print(f"[Scale] Weekly Total Scaling (weeksc): {weeksc}")

# === Apply Scaling to Each Forecaster ===
for forecaster_id in id_to_forecaster:
    current_error_data = week_error_df.loc[forecaster_id].copy()

    current_error_data["Precip_Cat"] *= prbsc * presc
    current_error_data["Precip_Amount"] *= presc
    current_error_data["Wind"] *= windsc
    
    for var in current_error_data.index:
        current_error_data[var] *= weeksc

    week_error_df.loc[forecaster_id] = current_error_data

# Error data for each forecaster is stored in error_data_{file} for easy reading
for file in forecast_files:
    try:
        df = pd.DataFrame.from_dict(error_data[file])
        df.to_csv(f"raw_error_data_{file}.csv")
        df = pd.DataFrame.from_dict(consensus_data[file])
        df.to_csv(f"raw_consensus_data_{file}.csv")
        df = pd.DataFrame.from_dict(forecasts[file])
        df.to_csv(f"forecasts_{file}.csv")
    except KeyError:
        print(f"[!] No error data found for {file}")


consensus_error_data.to_csv("consensus_error_scores.csv")

week_error_df.to_csv("weighted_weekly_error_scores.csv")