from flask import Flask, render_template, request, redirect, url_for
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
import pandas as pd
from io import StringIO

app = Flask(__name__)

# Google Sheets API setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('certain-tendril-430414-n8-7d615e996d05.json', scope)
client = gspread.authorize(creds)
spreadsheet = client.open("Forecasting_Game")

# Access specific sheets by name
sheet = spreadsheet.worksheet("Submissions")
sheet2 = spreadsheet.worksheet("Results")
params = {
    'station': 'KSYR',  # Station ID for Syracuse Airport
    'data': 'tmpf,dwpf,',  # Temperature and dew point data
    'year1': '2024', 'month1': '7', 'day1': '22', 'hour1':'22',  # Start date
    'year2': '2024', 'month2': '7', 'day2': '22', 'hour2':'23', # End date
    'tz': 'Etc/UTC',  # Timezone
    'format': 'csv',  # Format of the returned data
}
 
# API endpoint
url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'

# Make the GET request
response = requests.get(url, params=params)

# Check the status code of the response
if response.status_code == 200:
    # Parse the CSV data
     # Define column names based on the provided data
    column_names = ['Station_ID', 'Datetime', 'Temperature', 'Dew_Point']
    
    # Use StringIO to read the CSV data
    data = StringIO(response.text)
    
    # Parse the CSV data
    df_observed = pd.read_csv(data, names=column_names, header=None, skip_blank_lines=False)
    
    # Replace 'M' with NaN (missing values)
    df_observed.replace('M', pd.NA, inplace=True)

    # Function to clean a DataFrame column
    def clean_column(column):
        return pd.to_numeric(column, errors='coerce')

    # Apply the cleaning function to specific columns
    df_observed['Temperature'] = clean_column(df_observed['Temperature'])
    df_observed['Dew_Point'] = clean_column(df_observed['Dew_Point'])
    df_observed = df_observed.dropna().reset_index(drop=True)
    print(df_observed['Temperature'].values)
else:
    print(f"Error: {response.status_code}")

# Example: Get all records from the sheet

# Save data to a CSV file
#df.to_csv('asos_data.csv', index=False)


@app.route('/')
def index():
    return render_template('index.html')

#@app.route('/submit', methods=['POST'])
#def submit():
#    forecasted_temp = request.form['forecasted_temp']
#    forecasted_wind = request.form['forecasted_wind']
    
    # Append user input to Google Sheet
#    sheet.append_row([date, station_id, forecasted_temp, forecasted_wind])
    
#    return redirect(url_for('index'))

@app.route('/results')

def results():
    # Retrieve user forecasts from Google Sheets
    user_data = sheet.get_all_records()
    print(user_data)
    df_user = pd.DataFrame(user_data)
    print(df_user)
    # Retrieve observed data and calculate differences

    df_user['Temp_Diff'] = df_user['Forecasted_Temperature'].values[0] - df_observed['Temperature'].values[0]
    print(df_user['Temp_Diff'].values)
    print(df_observed['Temperature'].values)
    print(df_observed['Dew_Point'].values)
    print(float(df_user['Temp_Diff'].values[0]))
    sheet2.append_row([df_user['Temp_Diff'].values[0]])
    # Return the results as a table in HTML
    #return render_template('results.html', tables=[df_user.to_html(classes='data', header="true")])

results()
