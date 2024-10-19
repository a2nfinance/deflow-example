import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
def load_data(df):
    # Read genotype-phenotype data after subsequent data preprocessing
    #data = df.set_index('Unnamed: 0')
    data = df.copy()
    # Split original data to training and testing data
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,0:-1], data.iloc[:,-1], test_size=0.2, random_state=42)
   
    feature_names = list(data.columns)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert all to numpy
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    return X_train, y_train, X_test, y_test, feature_names, scaler