# import pandas as pd
# import numpy as np
# import calendar

# df = pd.read_csv("C:/Users/devik/Downloads/rider_weather_data.csv")

# # Parse datetime with correct format
# df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')

# # Mask for only Nov (11) and Dec (12)
# mask = df['datetime'].dt.month.isin([11, 12])

# # All possible months
# all_months = np.arange(1, 13)

# def change_month_safe(dt):
#     new_month = np.random.choice(all_months)
#     year = dt.year
#     day = dt.day
#     hour = dt.hour
#     minute = dt.minute

#     # Max possible day in the new month
#     max_day = calendar.monthrange(year, new_month)[1]

#     # Clamp day to valid range
#     safe_day = min(day, max_day)

#     return dt.replace(month=new_month, day=safe_day)

# # Apply safe replacement
# df.loc[mask, 'datetime'] = df.loc[mask, 'datetime'].apply(change_month_safe)

# # Save output
# df.to_csv("modified_datetime_data.csv", index=False)

# print("Saved as modified_datetime_data.csv")

#-------------------------------------------------------------
#driver data
#-------------------------------------------------------------

# import pandas as pd
# import numpy as np
# import calendar

# # Load your driver data
# df = pd.read_csv("C:/Users/devik/Downloads/driver_data.csv")

# # Parse the date column
# df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# # All possible months
# all_months = np.arange(1, 13)

# def random_month_safe(dt):
#     """Replace the month with a random month (1-12), keeping the day valid."""
#     new_month = np.random.choice(all_months)
#     year = dt.year
#     day = dt.day
    
#     # Maximum valid day for the new month
#     max_day = calendar.monthrange(year, new_month)[1]
    
#     # Clamp day to valid range
#     safe_day = min(day, max_day)
    
#     return dt.replace(month=new_month, day=safe_day)

# # Apply the function to the date column
# df['date'] = df['date'].apply(random_month_safe)

# # Optional: format back to string
# df['date'] = df['date'].dt.strftime('%d-%m-%Y')

# # Save the modified data
# df.to_csv("modified_driver_data.csv", index=False)

# print("Saved as modified_driver_data.csv")

#-------------------------------------------------------------
#traffic data
#-------------------------------------------------------------

# import pandas as pd
# import numpy as np
# import calendar

# # Load traffic data
# df = pd.read_csv("C:/Users/devik/Downloads/traffic_data.csv")

# # Parse Timestamp column
# df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')

# # All possible months
# all_months = np.arange(1, 13)

# def random_month_safe(dt):
#     """Replace the month with a random month (1-12), keeping day, hour valid."""
#     new_month = np.random.choice(all_months)
#     year = dt.year
#     day = dt.day
#     hour = dt.hour
#     minute = dt.minute
#     second = dt.second

#     # Max day in new month
#     max_day = calendar.monthrange(year, new_month)[1]
#     safe_day = min(day, max_day)

#     return dt.replace(month=new_month, day=safe_day, hour=hour, minute=minute, second=second)

# # Apply the function
# df['Timestamp'] = df['Timestamp'].apply(random_month_safe)

# # Optional: format back to string
# df['Timestamp'] = df['Timestamp'].dt.strftime('%d-%m-%Y %H:%M')

# # Save modified data
# df.to_csv("modified_traffic_data.csv", index=False)

# print("Saved as modified_traffic_data.csv")


########################################
# fraud label
#########################################
import pandas as pd

# Load raw merged data
df = pd.read_csv("C:/Users/devik/Downloads/fraud_classify_data.csv")

print("Original Shape:", df.shape)

# Create fraud column
df["fraud"] = 0

df.loc[
    (df.get("Cancelled Rides by Driver", 0) > 0) |
    (df.get("Cancelled Rides by Customer", 0) > 2) |
    (df.get("Incomplete Rides", 0) > 0) |
    (df.get("Driver_Rating", 5) < 3) |
    (df.get("Customer Rating", 5) < 3) |
    (df.get("Traffic_Congestion_Level", "").astype(str).str.lower() == "high") |
    (df.get("Crime_Severity", 0) > 50) |
    (df.get("ride_count", 0) > 15) |
    (df.get("Traffic_Density", 0) > 70),
    "fraud"
] = 1

print(df["fraud"].value_counts())

# Save fraud-labeled data
df.to_csv("fraud_labeled.csv", index=False)

print("fraud_labeled.csv saved successfully.")
