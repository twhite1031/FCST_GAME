import pandas as pd
import math
from datetime import datetime
import requests
from io import StringIO
import re
import geopandas as gpd
from shapely.geometry import Point
from haversine import haversine, Unit

# Change names to match id number to real forecaster, you can add more forecasters
id_to_forecaster = {1:("Thomas", "human"),2:("Bob", "human"),3:("Jeff", "human"),4:("Tim", "human"),
                    5:("Tom", "human"),6:("Wang", "human"),7:("Steiger", "human"),8:("Pete", "human"),
                    9:("Ruan", "human"),10:("Sand", "human"),11:("Bobby", "human"),12:("Nat", "human"), 13: ("Tash", "human"),
                    14: ("Dave", "human"), 15: ("Tyler", "human"), 16: ("Yong", "human"), 17: ("NWS", "model"), 18: ("SREF", "model")}
num_of_models = 2
chosen_states = ["California", "Oregon", "Washington", "Idaho", "Montana"]
fcst_city = "Oswego"

# Get Stations that exceeded threshold (hundreths of inch) and the times you would like to look at for ASOS precip data
precip_threshold = 50
start_date, end_date = datetime(2024, 11, 20, 11,52), datetime(2024,11,20,11,59)

# Read the file with custom delimiter to handle spaces

def read_forecast_file(forecast_file):
    fcst = pd.read_csv(forecast_file, delimiter=r'\s*,\s*|\s+', engine='python', header=None, names=[
    "Number_id", "ProbFL", "Lat", "Lon", "High_T", "Low_T", 
    "Wind_dir", "Wind_speed", "Nil_p", "Lght_p", "Mod_p", "High_p", "Amount_p"])
    return fcst

def read_forecast_ver_file(forecast_ver_file):
    fcst = pd.read_csv(forecast_ver_file, delimiter=r'\s*,\s*|\s+', engine='python', header=None, skiprows=2, names=[
     "ProbFL", "High_T", "Low_T", 
    "Wind_dir", "Wind_speed", "Nil_p", "Lght_p", "Mod_p", "High_p", "Amount_p"])
    return fcst

# Define the file names
forecast_files = ["fcst.m24", "fcst.m48", "fcst.t24", "fcst.t48", "fcst.w24", "fcst.w48", "fcst.r24", "fcst.r48"]
forecasts = {}
numday = 0
for file in forecast_files:
    try:
        forecasts[file] = read_forecast_file(file)
        numday += 1
    except FileNotFoundError:
        print(f"{file} not found, make sure it exists in the same directory")


fcst_ver = read_forecast_ver_file("fcst.ver")

# Dictionary to store each verification day to value
forecasts_ver = {}
consensus_data = {}
error_data = {}

# Array to store tuples of verified lat/lon for stations
asos_lat_lon = []
asos_amount = []

# Dictionary to store each proccessed forecaster for a day
processed_data = {}
missed_game_status = {}

# Constant for wind calculations
pie18 = (math.pi)/18.
angle = 180/(math.pi)
num_missed_fcst = 0

# Values for precip error weight calculation
scale = 0
max_value_250_above = 0
max_value_200_250 = 0
max_value_150_200 = 0
max_value_100_150 = 0
max_value_50_100 = 0
gulf = .5


# Loop through each day and forecaster to figure out when forecasts were missed
for idx, file in enumerate(forecast_files):
    missed_game_status[file] = {}

    for forecaster_id in id_to_forecaster:
        # Retrieves one row of data based on the forecaster_id
        try:
            forecaster_data = forecasts[file].iloc[forecaster_id-1].to_list()
        except KeyError:
            continue
        print(f"{id_to_forecaster[forecaster_id][0]} data stored for {file}")

        # Check Flood game probability, if >10 then missed game
        if forecaster_data[1] > 10:
            num_missed_fcst += 1
            missed_game = True
            print(f"{id_to_forecaster[forecaster_id][0]} has missed the forecast game, flood probabilty was greater than 10")
        else:
            missed_game = False
        # Update the missed game status dictionary
        missed_game_status[file][forecaster_id] = missed_game

# Retrieve verification and calculate components
for idx,file in enumerate(forecast_files):
    
    # A list of zeros to match each forecast variable, will be used to create consensus data for each day
    consensus_data_list = [0] * 14

    # Check to see who missed forecast and how many humans to models
    num_missed_fcst = sum(1 for status in missed_game_status[file].values() if status)

    #print(f"Number of missed games on {file}: {num_missed_fcst}")
    total_human_forecasters = len(id_to_forecaster) - num_of_models

    #print("Total Human Forecasters: ", total_human_forecasters)
    num_human_active_fcsts = total_human_forecasters - num_missed_fcst
    num_human_active_fcsts_flood = num_human_active_fcsts

    # Will match each row with the respective day it is verifying
    forecasts_ver[file] = fcst_ver.iloc[idx].to_list()
    
    # Converts the wind speed to u and v componnts
    arg = (27 - forecasts_ver[file][3]) * pie18

    # Converts to m/s
    spd = forecasts_ver[file][4] / 1.94

    # Determine u and v components 

    ucomp_ver = spd * math.cos(arg)
    vcomp_ver = spd * math.sin(arg)
    
    forecasts_ver[file].append(ucomp_ver)
    forecasts_ver[file].append(vcomp_ver)
    #print("Forecasts ver", forecasts_ver)

    for forecaster_id in id_to_forecaster:
        # Retrieves one row of data based on the forecaster_id
        try:
            forecaster_data = forecasts[file].iloc[forecaster_id-1].to_list()
        except KeyError:
            #print(f"{file} not found")
            break
        #print(f"{id_to_forecaster[forecaster_id]} proccesed for {file}")

        # Check Longitude, make positive if negative
        if forecaster_data[3] > 0:
            forecaster_data[3] = -forecaster_data[3]
        name, ftype = id_to_forecaster[forecaster_id]

        # Check if the forecaster has missed the game for each day (file), if not compute precipitation categories
        if not missed_game_status[file][forecaster_id]:
            precip_prob_sum = forecaster_data[8] +  forecaster_data[9] +  forecaster_data[10] +  forecaster_data[11]
            #print("precip sum", precip_prob_sum)
            if(precip_prob_sum != 10):
                if (precip_prob_sum > 10):
                    forecaster_data[8] = (10.0 * forecaster_data[8]) / precip_prob_sum                
                    forecaster_data[9] = (10.0 * forecaster_data[9]) / precip_prob_sum
                    forecaster_data[10] = (10.0 * forecaster_data[10]) / precip_prob_sum
                    forecaster_data[11] = (10.0 * forecaster_data[11]) /   precip_prob_sum 

                else:
                    
                    # Assume forecaster has chosen 10 in NIL
                    forecaster_data[8] = 10
                    forecaster_data[9] = 0
                    forecaster_data[10] = 0
                    forecaster_data[11] = 0

        # Wind Direction Computations, same as verification file
        arg = (27 - forecaster_data[6]) * pie18
        spd = forecaster_data[7] / 1.94
        ucomp_fcst = spd * math.cos(arg)
        vcomp_fcst = spd * math.sin(arg)
        forecaster_data.append(ucomp_fcst)
        forecaster_data.append(vcomp_fcst)
        
        # Add data to consensus if a human forecaster did not miss the game, sum up each forecast
        if ftype == "human" and not missed_game_status[file][forecaster_id]:
            consensus_data_list[0] = consensus_data_list[0] + forecaster_data[1]
            #print("First prob:", consensus_data_list)

            if forecaster_data[2] > 0 and forecaster_data[3] < 0:
                consensus_data_list[1] = consensus_data_list[1] + forecaster_data[2]
                consensus_data_list[2] = consensus_data_list[2] + forecaster_data[3]
            else:
                num_human_active_fcsts_flood -= 1
            consensus_data_list[3] = consensus_data_list[3] + forecaster_data[4]
            consensus_data_list[4] = consensus_data_list[4] + forecaster_data[5]

            # U and V components instead of direction
            consensus_data_list[12] = consensus_data_list[12] + forecaster_data[13]
            consensus_data_list[13] = consensus_data_list[13] + forecaster_data[14]

            consensus_data_list[6] = consensus_data_list[6] + forecaster_data[7]

            consensus_data_list[7] = consensus_data_list[7] + forecaster_data[8]
            consensus_data_list[8] = consensus_data_list[8] + forecaster_data[9]
            consensus_data_list[9] = consensus_data_list[9] + forecaster_data[10]
            consensus_data_list[10] = consensus_data_list[10] + forecaster_data[11]
            consensus_data_list[11] = consensus_data_list[11] + forecaster_data[12]

            #print("Consensus Amount", consensus_data_list[11])
            #print("Forecast Amount", forecaster_data[12])
            
            # Convert Precip to correct amount per inch
            forecaster_data[12] = forecaster_data[12] * .01
            
        # Store the forecasters specific day in processed data in the dictionary
        if file not in processed_data:
            processed_data[file] = {}
        

        processed_data[file][forecaster_id] = forecaster_data

        # Store the consensus data before being averaged for each period in a dictionary
        consensus_data[file] = consensus_data_list
        
        
    # Loop to each value and compute consensus average using active forecasts
    for idx, value in enumerate(consensus_data[file]):

        # Uses index value to compute either flood or local consensus, since some people may not put values for flood location
        if idx > 0 and idx <= 2:
            #print("consensus_data_file array:", consensus_data[file])
            consensus_data[file][idx] = consensus_data[file][idx] / num_human_active_fcsts_flood 
        else:
            #print("value of consensus" + str(value) + 'idx' + str(idx))
            consensus_data[file][idx] = consensus_data[file][idx] / num_human_active_fcsts

    # Special calculation for wind direction using u and v
    if consensus_data[file][12] < 0:
        consensus_data[file][5] = 90 - (angle* (math.atan(consensus_data[file][13]/ consensus_data[file][12]))) 
    elif consensus_data[file][12] > 0:
        consensus_data[file][5] = 270 - (angle* (math.atan(consensus_data[file][13]/ consensus_data[file][12]))) 
    elif consensus_data[file][13] > 0:
        consensus_data[file][5] = 180
    elif consensus_data[file][13] < 0:
        consensus_data[file][5] = 360
    
    #Convert Precip to correct amount per inch
    #consensus_data[file][11] = consensus_data_list[11] * .01
    #print("Consensus_data: ", consensus_data[file])


# Print the stored data for verification
#for file, data in processed_data.items():
    #print(f"Data for file {file}:")
    #for forecaster_id, forecaster_data in data.items():
        #print(f"Forecaster {forecaster_id} ({id_to_forecaster[forecaster_id]}): {forecaster_data}")

# Function check lat lon is valid for game
def is_within_chosen_states(lat_lon_tuple, chosen_states_gdf):
    point = Point(lat_lon_tuple[1], lat_lon_tuple[0])  # Note that Point expects (lon, lat)
    return any(chosen_states_gdf.contains(point))

# Function to retrieve station data
def get_asos_precipitation(threshold, start_date, end_date):
    # Clear arrays to store more data if we re-run the code
    asos_lat_lon.clear()
    asos_amount.clear()
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
    flood_game_verify = False
    if response.status_code == 200:
        # Define column names based on the provided data
        column_names = ['Station_ID', 'Datetime', 'lon','lat', 'metar']
        # Use StringIO to read the CSV data
        data = StringIO(response.text)
        
        # Parse the CSV data
        df = pd.read_csv(data,names=column_names,header=5,on_bad_lines='skip')
        #print(df)
        #print('data from: ' + str(df['Datetime'].iloc[0]) + " to " + str(df['Datetime'].iloc[-1]))
        # Filter for 6Z and 12Z times
      
        # Clean data, replace 'M' with NaN
        df.replace('M', pd.NA, inplace=True)
        #print("Station Verified: ", df["Station_ID"])
        for idx, value in enumerate(df["metar"]):

            # Define the regex pattern to match "6" followed by digits
            pattern = r'\s6(\d{4})\s'
            
            match = re.search(pattern, value)

            if match:
                six_group = int(match.group(1))
                #print("Found 6 group", six_group)
                # A final check to make sure we are above 1 inch and below 30 inches (ensures we have the correct 6 group)
                if six_group > threshold and six_group < 3000:
                    flood_game_verify = True
                    #print(df['Station_ID'][idx], df['lat'][idx], df['lon'][idx], df['metar'][idx])
                    asos_lat_lon.append((df['lat'][idx],df['lon'][idx]))
                    asos_amount.append(six_group)
            else:
                six_group = None
        return df
    else:
        print(f"Error: {response.status_code}, Could not retrieve ASOS data")
        return None

# Run function to find Flood game verification at a specific time period
get_asos_precipitation(precip_threshold, start_date, end_date)

states = gpd.read_file('cb_2018_us_state_500k.shp')
# Filter the states GeoDataFrame to only include the chosen states
chosen_states_gdf = states[states['NAME'].isin(chosen_states)]
# Filter the list of tuples
filtered_asos_idx_lat_lon = [(idx, lat_lon) for idx, lat_lon in enumerate(asos_lat_lon) if is_within_chosen_states(lat_lon, chosen_states_gdf)]

# Extract the indices and values separately if needed
filtered_indices = [item[0] for item in filtered_asos_idx_lat_lon]
filtered_lat_lon = [item[1] for item in filtered_asos_idx_lat_lon]
#print("Lat/lon of verified stations",filtered_lat_lon)
#print("Allowed values for flood game: " ,filtered_lat_lon)
   
# This finds the closest station to your lat/lon and then applies the scale based on the precip the station recieved
def calc_flood_error(lat_fcst, lon_fcst):
    distance, station_coords, coord_idx = find_shortest_distance(lat_fcst, lon_fcst, filtered_lat_lon)
    #print("distance in km", distance)
    # Check if any verified, if none give 0 for error regarding location
    if coord_idx == None:
        return 0;
    else:
        amount_precip = asos_amount[coord_idx]    
        #print("Amount precip", amount_precip)
        
    global max_value_250_above 
    global max_value_200_250 
    global max_value_150_200 
    global max_value_100_150 
    global max_value_50_100 

    if 50 < amount_precip < 100:
        if amount_precip > max_value_50_100:
            max_value_50_100 = amount_precip
    if 100 < amount_precip < 150:
        if amount_precip > max_value_100_150:
            max_value_100_150 = amount_precip
    if 150 < amount_precip < 200:
        if amount_precip > max_value_150_200:
            max_value_150_200 = amount_precip
    if 200 < amount_precip < 250:
        if amount_precip > max_value_200_250:
            max_value_200_250 = amount_precip
    if 250 < amount_precip:
        if amount_precip > max_value_250_above:
            max_value_250_above = amount_precip
    else:
        amount_precip = None
    if max_value_250_above > 0:
        scale = 30
    elif max_value_200_250 > 0:
        scale = 25
    elif max_value_150_200 > 0:
        scale = 20
    elif max_value_100_150 > 0:
        scale = 15
    elif max_value_50_100 > 0:
        scale = 10
    else:
        scale = 1  # or some default scale value
    #print(scale)
    #print(distance)
    return distance / scale


# Finds the shortest_distance to each verified station using the Haversine formula
def find_shortest_distance(lat, lon, coordinates):
    
    # Convert the reference point to a tuple
    reference_point = (lat, lon)
    #print("ref point fcst", reference_point)
    #print("coordinates", coordinates)
    # Initialize the minimum distance with a large number
    min_distance = float('inf')
    closest_coord = float('inf')
    coord_idx = None

    # Iterate over each coordinate tuple in the list
    for idx, coord in enumerate(coordinates):
        # Calculate the distance using haversine
        distance = haversine(reference_point, coord, unit=Unit.KILOMETERS)
        
        # Update the minimum distance if a shorter distance is found
        if distance < min_distance:
            min_distance = distance
            closest_coord = coord
            coord_idx = idx
                      
    return min_distance, closest_coord, coord_idx

# Computes the error for one day/file if consensus
def compute_consensus_error(consensus_file):
    consensus_error_data = {}

    for day in consensus_file:

        # Initialize a dict to store error for each day
        consensus_error_data[day] = {}

        # Dummy array for error_data
        error_data_list = [0] * 7

        #print("Computing Consensus error for ", day)

        current_data = consensus_file[day] 
        verification_data = forecasts_ver[day]
        flood_prob_error = ((current_data[0]) - verification_data[0]) ** 2
        error_data_list[0] = flood_prob_error

        # Check if there is any value
        if current_data[0] != 0:
            #print("Calculating Flood Location Error)
            flood_loc_error = calc_flood_error(current_data[1], current_data[2])
            error_data_list[1] = flood_loc_error
            
            # Penalty applied to indivduals + consensus error
            high_temp_error = abs(current_data[3] - verification_data[1])
            error_data_list[2] = high_temp_error
            low_temp_error = abs(current_data[4] - verification_data[2])
            error_data_list[3] = low_temp_error
            u_comp_error = current_data[12] - verification_data[10]
            v_comp_error = current_data[13] - verification_data[11]

            wind_error = math.sqrt((u_comp_error ** 2) + (v_comp_error ** 2))
            error_data_list[4] = wind_error
            precip_amnt_error = abs(current_data[11] - verification_data[9])
            error_data_list[5] = precip_amnt_error

        # Copied from original fortran code, calculate precip category errors
            f1 = current_data[7]
            f2 = f1 + current_data[8]
            f3 = f2 + current_data[9]
            v1 = verification_data[5]
            v2 = v1 + verification_data[6]
            v3 = v2 + verification_data[7]
            a = (f1-v1) ** 2
            b = (f2-v2) ** 2
            c = (f3-v3) ** 2
            
            precip_catgry_error = a + b + c
            error_data_list[6] = precip_catgry_error
            consensus_error_data[day] = error_data_list.copy()

    return consensus_error_data

consensus_error_data = compute_consensus_error(consensus_data)
#print("consensus error data" , consensus_error_data)

penal = 1.5
locpen = 1.4

for idx, file in enumerate(forecast_files):
    error_data[file] = {}
    # Start computing errors
    error_data_list = [0] * 7

    # A list of zeros to match each forecast variable, will be used to create error data for each day
    try:
        for forecaster_id in id_to_forecaster:
            print(f"Starting {file} error computation for ", id_to_forecaster[forecaster_id][0])
            #print("missed_game_status: ", missed_game_status[file][forecaster_id])
            if not missed_game_status[file][forecaster_id]:
                #print("Running calculations")
                current_data = processed_data[file][forecaster_id] 
                verification_data = forecasts_ver[file]
                flood_prob_error = ((current_data[1]) - verification_data[0]) ** 2
                error_data_list[0] = flood_prob_error 
                #print("cur data 1", current_data[1])
                #print("cur data 2", current_data[2])
                #print("cur data 3", current_data[3])
                #print("ver data 0", verification_data[0])          

                if current_data[2] == 0 or current_data[3] == 0:
                    #print("Giving penalty for no lat or lon for flood game")
                    error_data_list[1] = locpen * consensus_error_data[file][0]
                else:
                    #print("Calculating flood location error")
                    flood_loc_error = calc_flood_error(current_data[2], current_data[3])
                    error_data_list[1] = flood_loc_error
                    
                high_temp_error = abs(current_data[4] - verification_data[1])
                error_data_list[2] = high_temp_error
                low_temp_error = abs(current_data[5] - verification_data[2])
                error_data_list[3] = low_temp_error
                #print("fcster id:", forecaster_id)
                u_comp_error = current_data[13] - verification_data[10]
                v_comp_error = current_data[14] - verification_data[11]

                wind_error = math.sqrt((u_comp_error ** 2) + (v_comp_error ** 2))
                error_data_list[4] = wind_error
                precip_amnt_error = abs(current_data[12] - verification_data[9])
                error_data_list[5] = precip_amnt_error
                # Copied from original, calculate precip category errors
                f1 = current_data[8]
                f2 = f1 + current_data[9]
                f3 = f2 + current_data[10]
                v1 = verification_data[5]
                v2 = v1 + verification_data[6]
                v3 = v2 + verification_data[7]
                a = (f1-v1) ** 2
                b = (f2-v2) ** 2
                c = (f3-v3) ** 2
            
                precip_catgry_error = a + b + c
                error_data_list[6] = precip_catgry_error
                error_data[file][forecaster_id] = error_data_list.copy()
            else:
                error_data_list[0] = penal * consensus_error_data[file][0]
                error_data_list[1] = penal * consensus_error_data[file][1]
                error_data_list[2] = penal * consensus_error_data[file][2]
                error_data_list[3] = penal * consensus_error_data[file][3]
                error_data_list[4] = penal * consensus_error_data[file][4]
                error_data_list[5] = penal * consensus_error_data[file][5]
                error_data_list[6] = penal * consensus_error_data[file][6]


                error_data[file][forecaster_id] = error_data_list.copy()


    except KeyError:
        print("Could not find " + file + " for error calculation")

# Sum up all errors for the week
week_error = {}
con_week_error = [0,0,0,0,0,0,0]
for idx, file in enumerate(forecast_files):
    con_daily_errors = consensus_error_data[file]
    for i in range(len(con_daily_errors)):
        con_week_error[i] += con_daily_errors[i]
    for forecaster_id in id_to_forecaster:
        try: 
            if forecaster_id not in week_error:
                # Make an empty array to store weekly error sum if it doesn't exist already
                week_error[forecaster_id] = [0] * len(error_data[file][forecaster_id])

            # Retrieve the error of just one day and forecaster
            daily_errors = error_data[file][forecaster_id]
            #print('len of daily', range(len(daily_errors)))

            # For each error value, sum each index and save it into the week_error dictionary
            for i in range(len(daily_errors)):
                week_error[forecaster_id][i] += daily_errors[i]
                
                #Debugging: Print the current sum and the daily error value
                #print(f"After adding, week_error[{forecaster_id}][{i}] = {week_error[forecaster_id][i]}")
                #print(f"Daily error for index {i} on day {file}: {daily_errors[i]}")
        except KeyError:
            print(f"Likely no data found for forecaster id {forecaster_id} on {file}")
#print("week error", week_error)
#print("con error", con_week_error)

#print(missed_game_status)
#print(processed_data)

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

# Add weights to each error using consensus for the week

con_p_amount_error = con_week_error[5]
con_p_prob_error = con_week_error[6]
con_temp_error = con_week_error[2] + con_week_error[3]
con_wind_error = con_week_error[4]
con_flood_prob_error = con_week_error[0]
con_flood_loc_error = con_week_error[1]
    
# Scaling of amount error vs probability error for precip
if con_p_prob_error < .05:
    prbsc = 0
else:
    prbsc = gush * (con_p_amount_error / con_p_prob_error)
if prbsc < psn:
    prbsc = psn
else:
    prbsc = psx

# Scaling of precipitation error versus temperature erorr
con_p_total_error = con_p_prob_error + con_p_amount_error
if con_p_total_error < .05:
    presc = 0
else: 
    presc = con_temp_error / con_p_total_error
if presc > pscx:
    presc = pscx
# Scaling of wind error versus temperature error
if con_wind_error < .05:
    windsc = 0
else:
    windsc = gust * (con_temp_error / con_wind_error)

# Scaling of location versus probability for flood game
if (con_flood_prob_error < .05):
    locprbscale = 0
else:
    locprbscale = gulp * (con_flood_loc_error / con_flood_prob_error)

# Scaling of local game vs flood game
con_local_error = con_p_amount_error + con_p_prob_error + con_temp_error + con_wind_error
con_flood_error = con_flood_prob_error + con_flood_loc_error

if con_flood_loc_error <= 0:
    localfloodscale = 1
else:
    localfloodscale = gulf * (con_local_error / con_flood_error)
    if(localfloodscale > localfloodscalex):
        localfloodscale = localfloodscalex
    if(localfloodscale < localfloodscalen):
        localfloodscale = localfloodscalen

# Scaling of the week 
total_con_error = sum(con_week_error)
erweek = numday * erday
weeksc = erweek / total_con_error

for forecaster_id in id_to_forecaster:
    current_error_data = week_error[forecaster_id]
    # Adjust Precip prob
    current_error_data[6] = prbsc * current_error_data[6]
    current_error_data[6] = presc * current_error_data[6]
    current_error_data[5] = presc * current_error_data[5]

    # Adjust wind error
    current_error_data[4] =  windsc * current_error_data[4]

    # Adjust all values
    for i in range(len(current_error_data)):
        current_error_data[i] = weeksc * current_error_data[i]
    # Now get total of sum of every value after scale for individual
    error_total = sum(current_error_data)

#print("Data from files", forecasts)
#print("Raw error data", error_data)
#print("Consensus data", consensus_error_data)       
#print("Missed game status: ", missed_game_status)

