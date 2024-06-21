import pandas as pd
import torch

# Specify the path to your Excel file
excel_file = 'SWH CP1 vis 20200401-20200531.xlsx'

# Read the Excel file into a pandas DataFrame
data = pd.read_excel(excel_file)

print(data.head(10))

# fillering missing data
missing = []
for i in range(1, len(data)):
    current_time = data.iloc[i, 1]
    previous_time = data.iloc[i-1, 1]
    
    # When the data is in next day
    if current_time - previous_time == pd.Timedelta(hours=14):
        for j in range(4):
            missing.append(i)
        continue

    elif current_time - previous_time > pd.Timedelta(hours=3):
        print(current_time - previous_time)
        difference = int(((current_time - previous_time).total_seconds()/60) - (14 * 60))
        print(((current_time - previous_time).total_seconds()/60))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++',difference)
        print('difference:',difference)
        print(i)
        for x in range(difference+4):
            missing.append(i)
        continue


    # Check if the current time is not continuous with the previous time
    elif current_time - previous_time > pd.Timedelta(minutes=1):
        difference = (current_time - previous_time).total_seconds() / 60
        # Add an empty line
        for z in range(round(difference)-1):
            missing.append(i)
        continue

# descending order
missing = missing[::-1]
print(missing)

# filling the missing data with empty values
for i in missing:
    data = pd.concat([data.iloc[:i], 
                      pd.DataFrame({data.columns[0]: 'CP1', 
                                    data.columns[1]: data.iloc[i,1] - pd.Timedelta(minutes=1), 
                                    data.columns[2]: int(data.iloc[i-1,2]+data.iloc[i,2])/2}, index=[0]), 
                      data.iloc[i:]]).reset_index(drop=True)
print(data.head(10))

# Resample the data per 5 minutes
resampled_data = data.iloc[::5]

# Adding a new column named image_name
# data['image_name'] = 'img' + data.iloc[:, 0].astype(str) + '_' + \
#     data.iloc[:, 1].astype(str).str[2:4] + \
#     data.iloc[:, 1].astype(str).str[5:7] + \
#     data.iloc[:, 1].astype(str).str[8:10] + \
#     '_' + \
#     data.iloc[:, 1].astype(str).str[11:13] + \
#     data.iloc[:, 1].astype(str).str[14:16] + '.jpg'

# Adding a new column named image_name
resampled_data['image_name'] = 'img' + resampled_data.iloc[:, 0].astype(str) + '_' + \
    resampled_data.iloc[:, 1].astype(str).str[2:4] + \
    resampled_data.iloc[:, 1].astype(str).str[5:7] + \
    resampled_data.iloc[:, 1].astype(str).str[8:10] + \
    '_' + \
    resampled_data.iloc[:, 1].astype(str).str[11:13] + \
    resampled_data.iloc[:, 1].astype(str).str[14:16] + '.jpg'

# Print the resampled data
print(resampled_data)


# Save the resampled data to a new Excel file with only two columns
filename = 'data.xlsx'
resampled_data[['image_name', 'VISIBILITY DATA (IN 10 METERS)']].rename(columns={'VISIBILITY DATA (IN 10 METERS)': 'label'}).to_excel(filename, index=False)