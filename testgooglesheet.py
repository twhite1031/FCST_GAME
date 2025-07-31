import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Define the scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Load the credentials
creds = ServiceAccountCredentials.from_json_keyfile_name('/data2/white/PYTHON_SCRIPTS/METCLUB/certain-tendril-430414-n8-7d615e996d05.json', scope)

# Authorize the client
client = gspread.authorize(creds)

# Open the Google Sheet by name
sheet = client.open("Forecasting_Game").sheet1
print(sheet)
# Example: Get all records from the sheet
records = sheet.get_all_records()
print(records)

# Example: Append a row to the sheet
sheet.append_row(["2024-07-23", "KSYR", "25.0", "15.0"])
