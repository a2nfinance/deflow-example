# Import packages
from __future__ import print_function, division
import pandas as pd
import numpy as np
import argparse
import time
import zipfile
import os

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn import tree
from joblib import dump
from tqdm import tqdm

from math import sqrt
from scipy.special import ndtri
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
def load_data(df):
    # Read genotype-phenotype data after subsequent data preprocessing
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
                 
def eval (model, X_test, y_test):
    # Load and fit model
    pred = model.predict(X_test)
    # Find AUC score
    roc_auc = roc_auc_score (y_test, pred)
    
    TN, FP, FN, TP = confusion_matrix(y_test, pred).ravel()
    print('TN', TN, 'FP', FP, 'FN', FN, 'TP', TP)
    print("Roc_AUC: ", roc_auc)
    ss, sp, acc, mcc, ss_ci, sp_ci, acc_ci, mcc_ci = measure (TN, FP, FN, TP, 0.95)

    return pd.DataFrame({'Sensitivity': [ss, ss_ci], 'Specificity': [sp, sp_ci], \
                        'Accuracy': [acc, acc_ci], 'MCC': [mcc, mcc_ci], 'AUC': roc_auc})
 

def proportion_confidence_interval(r, n, z):
    A = 2*r + z**2
    B = z*sqrt(z**2 + 4*r*(1 - r/n))
    C = 2*(n + z**2)
    return ((A-B)/C, (A+B)/C)

def sensitivity_and_specificity_with_confidence_intervals(TP, FP, FN, TN, alpha=0.95):

    z = -ndtri((1.0-alpha)/2)
    
    # Compute sensitivity using method described in [1]
    sensitivity_point_estimate = TP/(TP + FN)
    sensitivity_confidence_interval = proportion_confidence_interval(TP, TP + FN, z)
    
    # Compute specificity using method described in [1]
    specificity_point_estimate = TN/(TN + FP)
    specificity_confidence_interval = proportion_confidence_interval(TN, TN + FP, z)
    
    # Compute MCC
    mcc = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) + 10**(-16))
    mcc_confidence_interval = proportion_confidence_interval(TP*TN - FP*FN, sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) + 10**(-16)), z)

    # Compute accuracy
    acc = (TP +TN)/(TP+FP+TN+FN)
    acc_confidence_interval = proportion_confidence_interval(TP +TN, TP+FP+TN+FN, z)

    return sensitivity_point_estimate, specificity_point_estimate, \
        sensitivity_confidence_interval, specificity_confidence_interval, \
        acc, acc_confidence_interval, mcc, mcc_confidence_interval 

# Get sensitivity, specificity, accuracy, and Matthews's correlation coefficient.

def measure (TP, FP, FN, TN, a):
    sensitivity_point_estimate, specificity_point_estimate, \
        sensitivity_confidence_interval, specificity_confidence_interval, \
        acc, acc_confidence_interval, mcc, mcc_confidence_interval \
        = sensitivity_and_specificity_with_confidence_intervals(TP, FP, FN, TN, alpha=a)
 
    print("Sensitivity: %f, Specificity: %f, Accuracy: %f, MCC: %f" %(sensitivity_point_estimate, specificity_point_estimate, acc, mcc))
    print("alpha = %f CI for sensitivity:"%a, sensitivity_confidence_interval)
    print("alpha = %f CI for specificity:"%a, specificity_confidence_interval)
    print("alpha = %f CI for accuracy:"%a, acc_confidence_interval)
    print("alpha = %f CI for MCC:"%a, mcc_confidence_interval)
    print("")
    return (sensitivity_point_estimate, specificity_point_estimate, acc, mcc, \
            sensitivity_confidence_interval, specificity_confidence_interval, \
            acc_confidence_interval, mcc_confidence_interval)


# Select features using decision tree model
def rfe_dt(X_train, y_train, X_test, y_test, output):
    
    rfe_dt = tree.DecisionTreeClassifier(random_state=7)
    best_auc = list()
    features = []
    iddd = []
    gr = []


    for i in tqdm(range(1, len(X_train[0]))):
        rfe = RFE(rfe_dt, n_features_to_select=i)
        rfe.fit_transform(X_train, y_train)
        # Reduce X to the selected features.
        X_train_reduce = rfe.transform(X_train)
        X_test_reduce = rfe.transform(X_test)
        # Decision tree model
        roc_auc, dt_grid = train_dt(X_train_reduce, y_train, X_test_reduce, y_test)
        if i % 10 == 0:
            print("Number of features:", i, "AUC: ", roc_auc)
        features.append(rfe.support_)
        best_auc.append(roc_auc)
        gr.append(dt_grid)
        iddd.append(i)

    print("The best AUC of Decision Tree: ", max(best_auc))
    idd = np.argmax(best_auc)
    print("Number of Selected Features is: ", iddd[idd])
    ft = features[idd] 

    # Save the model
    dump(gr[idd], output + "/dt.joblib")
    indice = [i for i, x in enumerate(ft) if x]
    
    pd.DataFrame({'features':indice}).to_csv(output + "/dt_features.csv")


def train_dt(X_train, y_train, X_test, y_test):

    # Create decision-tree cross validation

    grid = {'criterion': ["gini", "entropy"], 
            'splitter': ["best", "random"],
            'max_features': ["sqrt","log2"],
            'max_depth' : [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}
    
    clf = tree.DecisionTreeClassifier(random_state=7)
    dt_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5, n_jobs = -1)
    
    # Train the regressor
    dt_grid.fit(X_train, y_train)
    # Make predictions using the optimised parameters
    dt_pred = dt_grid.predict(X_test)
    roc_auc = round(roc_auc_score (y_test, dt_pred), 3)

    return (roc_auc, dt_grid)

parser = argparse.ArgumentParser(
                    prog='Training models using Machine Learning',
                    description='Logistic Regression, Decision Tree, and SVM',
                    epilog='Help')

parser.add_argument(
    '--input_data', 
    default="../feature_selection/data/combined_files.zip",
    metavar='i', type=str, help='Input path')

parser.add_argument('--output_dir', 
                    default="output",
                    metavar='o', type=str, help='Output path')

args = parser.parse_args()

def read_files_from_zip(zip_path, output_dir=None):
    """Read and process each file in the ZIP archive without extracting."""
    df = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            print(f"Processing {file_name} from the ZIP archive:")

            with zip_ref.open(file_name) as file:  # file is a ZipExtFile object
                # Check file size
                file_size = zip_ref.getinfo(file_name).file_size
                print(f"File size: {file_size} bytes")

                if file_size == 0:
                    print(f"Warning: {file_name} is empty.")
                    continue  # Skip empty files

                # Attempt to process as a CSV file
                try:
                    file.seek(0)  # Reset file pointer to the beginning
                    data = pd.read_csv(file).set_index("Unnamed: 0")
                    print("CSV, TXT file data:")
                    print(data.head())  # Show the first few rows of the CSV file
                    df.append(data)
                except pd.errors.EmptyDataError:
                    print(f"Error: {file_name} is empty.")
                
    return df

if __name__ == '__main__':
    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|         -----               DISEASE PREDICTION              -----          |")
    print("|                                                                            |")
    print("|============================================================================|")
    print("")
    print("")
    print("********************************* INPUT DATA *********************************")
    print("")
    print("Import data may take several minutes, please wait...")
    print("")

    # Ensure the output directory exists if provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    df = read_files_from_zip(args.input_data)
    # Load genotype-phenotype data 
    X_train, y_train, X_test, y_test, feature_names, _ = load_data(df[0])
    indice_features = df[1]['features']

    # Get data with the selected features
    X_train_reduce = X_train[:, indice_features]
    X_test_reduce = X_test[:, indice_features]
    
    # For reduce data
    start_time = time.time()
    dt_auc_reduce, dt_md_reduce = train_dt(X_train_reduce, y_train, X_test_reduce , y_test)
    end_time = time.time()
    sfe_knn_elapsed_time = end_time - start_time
    print("Elapsed time: ", sfe_knn_elapsed_time)
    print('AUC reduce: ', dt_auc_reduce)
    
    # Find the best model
    auc = [dt_auc_reduce]
    
    print("********************************** SAVING **********************************")
    eval(dt_md_reduce, X_test_reduce, y_test).to_csv(args.output_dir + "/dt_reduce_evaluations.csv")
    print("********************************* FINISHED *********************************")
    print("")
    