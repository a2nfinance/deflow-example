import os
import random
import argparse

def generate_data(output_dir, filename="generated_data.csv"):
    # Define the columns (Chromosome, Position, Reference, Alternate)
    columns = [
        "chr1_18236545_A_G", "chr1_72372846_A_G", "chr1_72372878_G_T", "chr1_72559191_G_A",
        "chr1_73198389_T_G", "chr1_100355937_C_G", "chr1_103268762_T_C", "chr1_109604665_C_T",
        "chr1_118961220_C_G", "chr1_177920345_A_G", "chr1_177944384_T_C", "chr1_219470882_A_G",
        "chr1_223728074_G_A", "chr2_622723_A_G", "chr2_622857_A_G", "chr2_632028_C_T",
        "chr2_634905_T_C", "chr2_635004_C_T", "chr2_638107_G_A", "chr2_638144_G_T"
    ]
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
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save to CSV
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        # Write header
        f.write(",".join(columns) + "\n")
        # Write rows
        for i, row in enumerate(rows):
            f.write(f"SRR{i+6996662}," + ",".join(map(str, row)) + "\n")

    print(f"File saved to {output_path}")


if __name__ == "__main__":
    # Command-line argument parsing with defaults
    parser = argparse.ArgumentParser(description="Process files from a ZIP archive.")

    
    # Adding output_dir with a default value
    parser.add_argument(
        "--output_dir", 
        default="output", 
        help="Directory to save processed files (default: output_data)"
    )

    args = parser.parse_args()

    # Ensure the output directory exists if provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    print ("Generating data")
    generate_data(args.output_dir) 
    print("Finish")
