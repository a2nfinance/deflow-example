import random
import sys
import pandas as pd

def generate_data(local=False):
    # Get data from a local file.
    file_path = get_input(local)

    print("Reading local file.")
    # Attempt to process as a file
    data = pd.read_csv(file_path, header = None)

    columns = []

    for x in data.iloc[:, 0]:
        columns.append(x)
    # Define the columns (Chromosome, Position, Reference, Alternate)
    columns *= 10

    # Add all other columns here...
    columns.append("SEX")
    columns.append("PHENOTYPE")

    # Generate random data for 300 rows (can be increased)
    rows = []
    for _ in range(300):
        row = [random.choice([0, 1, 2]) for _ in range(len(columns) - 2)]  # Random genotypes
        row.append(random.choice([0, 1]))  # SEX: 0 or 1
        row.append(random.choice([0, 1]))  # PHENOTYPE: 0, 1
        rows.append(row)
    
    filename = "generated_data.csv" if local else "/data/outputs/generated_data.csv"
    
    with open(filename, "w") as f:
        # Write header
        f.write(",".join(columns) + "\n")
        # Write rows
        for i, row in enumerate(rows):
            f.write(f"SRR{i+6996662}," + ",".join(map(str, row)) + "\n")

    print(f"File saved to {filename}")

def get_input(local=False):
    if local:
        return "position.txt"
    else:
        return "/data/inputs/position.txt"  

if __name__ == "__main__":
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    generate_data(local)
