# Import packages
import pandas as pd
import numpy as np
import models.dt as dt
from data_loader import load_data
from metrics import eval
import argparse
import time
import zipfile
import os

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
    dt_auc_reduce, dt_md_reduce = dt.train_dt(X_train_reduce, y_train, X_test_reduce , y_test)
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
    