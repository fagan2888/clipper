import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def assign_users(group):
    #This function assigns User IDs to each day in the dataset based on the baseline dataset
    #Should be called from Pandas .transform method
    print ("Starting CircadianDayOfWeek {}, RandomWeekID {}".format(int(group['CircadianDayOfWeek'].unique()),
                                                                    int(group['RandomWeekID'].unique())))
    #In the baseline array, the index is the UserID
    X = df_baseline.copy()
    user_ids = []
    for i, row in group.iterrows():
        #Calculate distances in feature space
        d = euclidean_distances(X, [row[feature_cols]])
        
        #Get row number of closest item
        row_number = np.argmin(d)
        
        #Get user ID from row
        user_id = X.iloc[row_number].name
        
        #print("Row {}: ClipperCardID {} matched to UserID {}".format(i, row['ClipperCardID'], user_id))
        
        #Add UserID to return array
        user_ids.append(user_id)
        
        #Drop row with this UserID so we can't assign it again
        X.drop(user_id, inplace=True)
        
    return pd.DataFrame({'UserID': np.array(user_ids), 'ClipperCardID': group['ClipperCardID']})


if __name__ == "__main__":
    #This script takes in the raw Clipper data and determines User IDs for each Clipper Card ID
    #Clipper Card IDs reset everyday, whereas User IDs are consistant across entire dataset for a given transit rider

    #Read in Clipper data
    df = pd.read_csv('data/2014_-_9_Anonymous_Clipper.csv')

    #Strip the column names of white space
    for col in ['AgencyName', 'PaymentProductName', 'RouteName']:
        df[col] = df[col].str.strip()

    #Convert time to floats
    df['TagOnTime_Time'] = pd.to_timedelta(df['TagOnTime_Time'])/pd.offsets.Hour(1)
    df['TagOffTime_Time'] = pd.to_timedelta(df['TagOffTime_Time'])/pd.offsets.Hour(1)

    #Drop unneeded columns
    df.drop(columns=['Year',
                     'Month',
                     'CircadianDayOfWeek_Name',
                     'AgencyName',
                     'PaymentProductName',
                     'TagOnLocationName',
                     'RouteName',
                     'TagOffLocationName'], inplace=True)

    #Create aggregation function dictionary
    agg_funcs = {}
    for col in df.columns:
        if col == 'TagOnTime_Time':
            agg_funcs[col] = 'min'
        elif col in ('TripSequenceID', 'TagOffTime_Time'):
            agg_funcs[col] = 'max'
        elif col not in ('CircadianDayOfWeek', 'RandomWeekID', 'ClipperCardID'):
            agg_funcs[col] = 'sum'

    #Aggregate by ClipperCardID, preserving day of week and week ID
    df = df.groupby(['CircadianDayOfWeek', 'RandomWeekID', 'ClipperCardID']).agg(agg_funcs).reset_index()

    #Fill NaNs, convert IDs to ints
    df['TagOffLocationID'] = df['TagOffLocationID'].fillna(0)
    df['TagOffTime_Time'] = df['TagOffTime_Time'].fillna(0)
    df['TagOffLocationID'] = df['TagOffLocationID'].astype('int')

    #Normalize data
    for col in df.columns:
        if col not in ('CircadianDayOfWeek', 'RandomWeekID', 'ClipperCardID'):
            df[col] = df[col]/df[col].max()

    #Split into baseline vs. everything else. Baseline data is busiest day in dataset.
    df_baseline = df[(df['CircadianDayOfWeek']==4) & (df['RandomWeekID']==8)]
    df = df[~((df['CircadianDayOfWeek']==4) & (df['RandomWeekID']==8))]

    #Drop uneeded columns
    df_baseline.drop(columns=['CircadianDayOfWeek', 'RandomWeekID'], inplace=True)

    #Reset indexes
    df_baseline.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    #Pop baseline Clipper IDs
    df_baseline_IDs = df_baseline.pop('ClipperCardID')

    #Create User IDs from baseline Clipper IDs
    df_baseline_IDs = df_baseline_IDs.reset_index()
    df_baseline_IDs.rename(index=str, columns={'index': 'UserID'}, inplace=True)

    #Assign User Ids to rest of dataset
    feature_cols = ['TripSequenceID', 'AgencyID', 'PaymentProductID', 'FareAmount',
            'TagOnTime_Time', 'TagOnLocationID', 'RouteID', 'TagOffTime_Time',
            'TagOffLocationID']
    
    df_results = df.groupby(['CircadianDayOfWeek', 'RandomWeekID']).apply(assign_users)
    
    #Recombine datasets for Clipper/user ID lookup
    df_clipper_users = pd.concat([df_baseline_IDs, df_results])
    df_clipper_users = df_clipper_users[['ClipperCardID', 'UserID']]
    df_clipper_users.sort_values('ClipperCardID', inplace=True)
    
    #Export to CSV
    df_clipper_users.to_csv('data/clipper_users.csv', index=False)