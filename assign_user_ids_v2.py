import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def assign_users(group):
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


if __name__ == '__main__':
    
    #Read in data
    df = pd.read_csv('data/df.csv')

    #Get feature column names from Payment Product Names
    feature_cols = list(np.unique(df[['PaymentProductName']]))
    
    #Create feature matrix with Payment Product Names
    df = df[['CircadianDayOfWeek', 'RandomWeekID', 'ClipperCardID', 'PaymentProductName', 'Year']].pivot_table(
                                                        index=['CircadianDayOfWeek', 'RandomWeekID', 'ClipperCardID'],
                                                        columns='PaymentProductName',
                                                       aggfunc='count',
                                                       values='Year').fillna(0).reset_index()
    
    #Take week with most unique users as baseline
    df_baseline = df[(df['CircadianDayOfWeek']==4) & (df['RandomWeekID']==7)]
    df = df[~((df['CircadianDayOfWeek']==4) & (df['RandomWeekID']==7))]
    
    #Drop uneeded columns
    df_baseline.drop(columns=['CircadianDayOfWeek', 'RandomWeekID'], inplace=True)
    
    #Reset indexes
    df_baseline.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    #Pop Clipper ID
    df_baseline_IDs = df_baseline.pop('ClipperCardID')
    df_baseline_IDs = df_baseline_IDs.reset_index()
    df_baseline_IDs.rename(index=str, columns={'index': 'UserID'}, inplace=True)
    
    #Assign user IDs
    df_results = df.groupby(['CircadianDayOfWeek', 'RandomWeekID']).apply(assign_users)
    
    #Recombine for Clipper/user ID lookup
    df_clipper_users = pd.concat([df_baseline_IDs, df_results])
    
    df_clipper_users = df_clipper_users[['ClipperCardID', 'UserID']]
    
    df_clipper_users.sort_values('ClipperCardID', inplace=True)
    
    df_clipper_users.to_csv('data/clipper_users.csv', index=False)