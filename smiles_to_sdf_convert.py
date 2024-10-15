"""
This script allows users to convert SMILES strings in a CSV File to an output SDF File. 
There are two conversion options: 
1. Convert using OpenBabel 
2. Covert using RDKit 

---------------------
Input: 
- a CSV file containing SMILES Strings 
- The column that contains SMILES Strings in the CSV file 
- Output file name 
Output: 
- An SDF File containing SDF format of all converted structures from SMILES Strings 

----------------------
The following code is implemented using ChatGPT's generation. 
"""

import subprocess
from dask import delayed
import dask
import dask.dataframe as dd
import os

from rdkit import Chem
from rdkit.Chem import AllChem

# OpenBabel conversion
def convert_with_openbabel(smiles_batch): 
    """ 
    Convert SMILES to SDF using OpenBabel.
    """     
    try:
        # Create a multiline SMILES string (simulating file content in memory)
        smiles_string = '\n'.join(smiles_batch)

        # Command to run OpenBabel using the SMILES string directly as input via a pipe
        command = "obabel -i smi -o sdf -h --gen3d"

        # Run OpenBabel and feed the SMILES string to stdin
        process = subprocess.run(command, input=smiles_string, text=True, shell=True, check=True, capture_output=True)

        # Capture the stdout (SDF output)
        return process.stdout  # Return the SDF output as a string
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while converting batch with OpenBabel: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def parallel_convert_using_openbabel(smiles_list, output_sdf): 
    batch_size = 10  # Set the batch size
    batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]
    
    # Use Dask to convert batches in parallel
    delayed_tasks = [dask.delayed(convert_with_openbabel)(batch) for batch in batches]
    
    # Compute the results
    sdf_results = dd.compute(*delayed_tasks)

    # Write all the SDF results to the output SDF file
    write_sdf_file(sdf_results, output_sdf)

    print(f"Successfully converted SMILES to SDF. Output saved to {output_sdf}")

# RDKit conversion
def smiles_to_sdf(smiles):
    """Convert a single SMILES string to SDF format."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: Invalid SMILES '{smiles}' encountered, skipping.")
        return ""  # Return empty string for invalid SMILES
    mol = Chem.AddHs(mol)

    try:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        sdf_block = Chem.MolToMolBlock(mol)
        return sdf_block
    except ValueError as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return ""  # Skip problematic molecules
    except RuntimeError as e:
        # Handle UFF parameter failure
        print(f"UFF optimization failed for SMILES '{smiles}' due to atom types: {e}")
        return ""  # Skip problematic molecules

def parallel_convert_using_rdkit(smiles_series):
    """Convert SMILES strings to SDF format in parallel."""
    # Create a list of delayed tasks
    delayed_tasks = [dask.delayed(smiles_to_sdf)(smiles) for smiles in smiles_series]
    
    # Compute the results in parallel
    sdf_results = dask.compute(*delayed_tasks)
    return sdf_results

def write_sdf_file(sdf_results, output_file):
    """Write the SDF results into a single output SDF file."""
    with open(output_file, 'w') as f:
        for sdf in sdf_results:
            if sdf:  # Check if sdf is not None
                f.write(sdf)
                f.write("\n$$$$\n") 

def main():
    # Get user input 
    input_csv = input("Enter the SMILES CSV file: ") 
    output_sdf = input("Enter the output SDF file name (e.g., output.sdf): ")
    smiles_column = input("Enter SMILES column to be converted: ")
    print("Choose the conversion method:") 
    print("1. OpenBabel") 
    print("2. RDKit") 
    choice = input("Enter 1 or 2: ")

    # Extract SMILES Strings from CSV to a list
    df = dd.read_csv(input_csv)
    smiles_list = df[smiles_column].compute().tolist()  # Convert to list

    if choice == '1': 
        parallel_convert_using_openbabel(smiles_list, output_sdf) 
    elif choice == '2': 
        # Convert SMILES to SDF
        sdf_results = parallel_convert_using_rdkit(smiles_list)

        # Write to output SDF file
        write_sdf_file(sdf_results, output_sdf)
    else: 
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__": 
    main()
