# Import packages
from data_loader import load_data
import pandas as pd
import zipfile
import models.dt as dt
import argparse
import time
import os
import io

parser = argparse.ArgumentParser(
                    prog='Training models using Machine Learning',
                    description='Logistic Regression, Decision Tree, and SVM',
                    epilog='Help')

parser.add_argument(
    '--meta', 
    default="../meta_data_collection/output/combined_files.zip", 
    help="Path to the ZIP file (default: combined_files.zip)",
    type=str
    )

parser.add_argument(
        "--data", 
        default="../data_preprocessing/output/data.csv", 
        help="Data after preprocessing for training",
        type = str
    )

parser.add_argument(
        "--gen_data", 
        default="../data_generation/output/generated_data.csv", 
        help="Generated data for training",
        type = str
    )

parser.add_argument(
        "--output_dir", 
        default="output", 
        help="Directory to save processed files (default: output)"
    )

parser.add_argument(
        "--output_data", 
        default="data", 
        help="Directory to save processed files (default: data)"
    )
args = parser.parse_args()

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

def read_files_from_zip(zip_path, output_dir=None):
    """Read and process each file in the ZIP archive without extracting."""
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
                    data_1 = pd.read_csv(file)
                    print("CSV, TXT file data:")
                    print(data_1.head())  # Show the first few rows of the CSV file

                except pd.errors.EmptyDataError:
                    print(f"Error: {file_name} is empty.")
                except pd.errors.ParserError:
                    # If it's not a CSV, TXT, try to process as a VCF file
                    try:
                        file.seek(0)  # Reset file pointer to the beginning
                        data_2 = read_vcf(file)  # Reading as VCF from the ZipExtFile
                        print("VCF file data:")
                        print(data_2.head())  # Show the first few rows of the VCF file
                    except ValueError as e:
                        print(f"Failed to process {file_name} as a VCF file: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred while processing {file_name}: {e}")
    return data_1, data_2

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
    
if __name__ == '__main__':

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

    # Ensure the output directory exists if provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.output_data:
        os.makedirs(args.output_data, exist_ok=True)

    # Process files from the ZIP
    meta, _ = read_files_from_zip(args.meta)
    # Combine data and generated data
    combined_df = concatenate_data(args.data, args.gen_data)
    print ("Combined data: ", combined_df.head())
    
    # Load genotype-phenotype data 
    X_train, y_train, X_test, y_test, feature_names, _ = load_data(combined_df)

    print("Decision-Tree RFE")
    # Start timer
    start_time = time.time()
    dt.rfe_dt(X_train, y_train, X_test, y_test, args.output_dir)
    end_time = time.time()
    # Calculate elapsed time
    dt_elapsed_time = end_time - start_time
    print("Elapsed time: ", dt_elapsed_time)
    print("")
 

    print("********************************** SAVING **********************************")
    
    #pd.DataFrame({'DT':[dt_elapsed_time]}).to_csv(args.output_dir + "/Time.csv")
    combined_df.to_csv(args.output_dir + "/combined_data.csv")
    # Get all files in the specified folder
    combine_files_into_zip(args.output_dir, args.output_data)
    print("")
    print("********************************* FINISHED *********************************")
    print("")
    