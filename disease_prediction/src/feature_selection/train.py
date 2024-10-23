# Import packages
import numpy as np
import pandas as pd
import zipfile
import time
import os
import io
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn import tree
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

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

# Select features using decision tree model
def rfe_dt(X_train, y_train, X_test, y_test, feature_output_file):
    
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
    #dump(gr[idd], output + "/dt.joblib")
    indice = [i for i, x in enumerate(ft) if x]
    
    pd.DataFrame({'features':indice}).to_csv(feature_output_file)


def train_dt(X_train, y_train, X_test, y_test):

    # Create decision-tree cross validation

    grid = {'criterion': ["gini"], 
            'splitter': ["best"],
            'max_features': ["sqrt"],
            'max_depth' : [2, 10]}
    
    clf = tree.DecisionTreeClassifier(random_state=7)
    dt_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5, n_jobs = -1)
    
    # Train the regressor
    dt_grid.fit(X_train, y_train)
    # Make predictions using the optimised parameters
    dt_pred = dt_grid.predict(X_test)
    roc_auc = round(roc_auc_score (y_test, dt_pred), 3)

    return (roc_auc, dt_grid)

def read_vcf(vcf_file):
    """Reads a VCF file from a ZipExtFile and returns it as a pandas DataFrame."""
    vcf_names = None  # Initialize vcf_names

    # Read all lines from the ZipExtFile into a list
    lines = [line.decode('utf-8') for line in vcf_file.readlines()]

    # Check if the VCF header exists
    for line in lines:
        if line.startswith("#CHROM"):
            vcf_names = [x for x in line.strip().split('\t')]
            break

    # Check if vcf_names was set
    if vcf_names is None:
        raise ValueError("VCF header not found in the file.")
    
    # Prepare data for DataFrame, filtering out the header lines
    data = pd.read_csv(io.StringIO(''.join(lines)), comment='#', sep='\s+', header=None, names=vcf_names)
    return data

def combine_files_into_zip(parent_folder, output_zip):
    """Create a ZIP file with all the files in the parent folder."""
    # Ensure the output directory exists
    os.makedirs(output_zip, exist_ok=True)

    # Define the full path for the output ZIP file
    zip_file_path = os.path.join(output_zip, "combined_files.zip")

    # Collect all file paths from the parent folder
    file_paths = []
    for root, _, files in os.walk(parent_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    print("Input file paths:", file_paths)  # Debugging output

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for file in file_paths:
            # Check if the file exists and is a valid file
            if isinstance(file, str) and os.path.isfile(file):
                zipf.write(file, os.path.basename(file))
            else:
                print(f"Skipping {file}: not a valid file path.")
    
    print(f"Files combined into {zip_file_path}")

def concatenate_data (csv_file1, csv_file2):

    # Read the CSV files into DataFrames
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    # Find the columns in df2 that are not in df1
    columns_to_add = [col for col in df2.columns if col not in df1.columns]

    # Concatenate only the unique columns from df2 to df1
    df_combined = pd.concat([df1.iloc[:, -2], df2[columns_to_add]], axis=1)
    df_combined["SEX"] = df1["SEX"]
    df_combined["PHENOTYPE"] = df1["PHENOTYPE"]
    # Replace NaN values with 0
    df_combined.fillna(0, inplace=True)
    print('Files concatenated without duplicates successfully.')
    return df_combined

def run_feature_selection(local: False):
    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|         -----      MACHINE LEARNING FEATURE SELECTIONS     -----           |")
    print("|                                                                            |")
    print("|============================================================================|")
    print("")
    print("")
    print("********************************* INPUT DATA *********************************")
    print("")
    print("Import data may take several minutes, please wait...")
    print("")

    input_data_file = "data/disease.csv" if local else "/data/inputs/disease.csv"
    input_generated_data_file = "data/generated_data.csv" if local else "/data/inputs/generated_data.csv"
    processing_output_dir = "data" if local else "/data/inputs" 
    feature_output_file = "data/dt_features.csv" if local else "/data/inputs/dt_features.csv"
    combine_data_output_folter = "." if local else "/data/outputs"
    # Combine data and generated datay
    combined_df = concatenate_data(input_data_file, input_generated_data_file)
    print ("Combined data: ", combined_df.head())
    
    # Load genotype-phenotype data 
    X_train, y_train, X_test, y_test, _, _ = load_data(combined_df)

    print("Decision-Tree RFE")
    # Start timer
    start_time = time.time()
    rfe_dt(X_train, y_train, X_test, y_test, feature_output_file)
    end_time = time.time()
    # Calculate elapsed time
    dt_elapsed_time = end_time - start_time
    print("Elapsed time: ", dt_elapsed_time)
    print("")
 

    print("********************************** SAVING **********************************")
    #pd.DataFrame({'DT':[dt_elapsed_time]}).to_csv(args.output_dir + "/Time.csv")
    combined_df.to_csv(processing_output_dir + "/combined_data.csv")
    # Clean unused data here
    os.remove(input_data_file)
    os.remove(input_generated_data_file)
    # Get all files in the specified folder
    combine_files_into_zip(processing_output_dir, combine_data_output_folter) # combined_files.zip
    print("")
    print("********************************* FINISHED *********************************")
    print("")
    
if __name__ == '__main__':
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_feature_selection(local)
    
    