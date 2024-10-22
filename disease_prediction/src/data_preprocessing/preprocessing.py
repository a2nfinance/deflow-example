import zipfile
import pandas as pd
import io
import sys
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

# Encode bi-allelic genotypes by 0, 1, 2
def get_genotype(geno): 
    geno = geno.set_index('ID')
    # Encode genotype to 0, 1, 2
    pd.set_option('future.no_silent_downcasting', True)
    df = geno.replace(['0|0', '0|1', '1|0', '1|1'], [0, 1, 1, 2])
    df = df.iloc[:, 8::]
    df = df.T
    return (df)

# Add phenotype to genotype data
def add_phenotype(obs, geno):
    # Collect sex data
    s = list(obs['sex'])
    se = []
    for x in s:
        if x == 'male':
            se.append(0)
        elif x == 'female':
            se.append(1)
        else:
            se.append(-9)
    # Collect phenotype data
    pn = list(obs['Sample Name'])
    pheno = []
    for x in pn:
        if 'OBL' in x:
            pheno.append(0)
        if 'OBH' in x:
            pheno.append(1)

    geno['SEX'] = se
    geno['PHENOTYPE'] = pheno
    
    return geno

def get_input(local=False):
    if local:
        print("Reading local file disease.zip.")
        return "disease.zip"
    else:
        return "/data/inputs/disease.zip"  

def run_process(local=False):
    input_data = get_input(local)
    filename = "disease.csv" if local else "/data/outputs/disease.csv"
     # Command-line argument parsing with defaults    
    # Process files from the ZIP
    data_1, data_2 = read_files_from_zip(input_data)
    # Data including encoded genotypes
    print ("Encode bi-allelic genotypes by 0, 1, 2")
    geno = get_genotype(data_2)
    df = add_phenotype(data_1, geno)
    print ("Save the training data to ", filename)
    df.to_csv(filename)

if __name__ == "__main__":
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_process(local)
    


    
    
