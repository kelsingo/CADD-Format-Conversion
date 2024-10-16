"""
This script retries the failed conversion that occurred 3 times during attempts by RDKit. 
The molecules that are failed to convert are: 
1. CN(c1[nH]cnc2nccc1-2)C1CCC(C[SH]2(=O)NC3CCCCC3CO2)CC1 (Mol_1)
2. C=C=C
3. CC(C)(C)c1cc(CCC(=O)OCC(COC(=O)CCc2cc(C(C)(C)C)c(O)c(C(C)(C)C)c2)(COC(=O)CCc2cc(C(C)(C)C)c(O)c(C(C)(C)C)c2)COC(=O)CCc2cc(C(C)(C)C)c(O)c(C(C)(C)C)c2)cc(C(C)(C)C)c1O
4. c1cc2ccc1CCc1ccc(cc1)CC2
5. CN(c1[nH]cnc2nccc1-2)C1CCC(C[SH]2(=O)NC3CCCCC3CO2)CC1
"""

import subprocess 
import rdkit
from rdkit import Chem 
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem


def convert_with_openbabel(input_smiles, output_file): 
    """ 
    Convert SMILES to SDF using OpenBabel.
    """     
    try:
        smile = input_smiles.strip()
        command = f'echo "{smile}" | obabel -ismi -O {output_file} -h --gen3d'
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully converted using OpenBabel. Output saved to {output_file}")

    except subprocess.CalledProcessError: 
        print("Error occurred while converting with OpenBabel.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def convert_with_rdkit(input_smiles, output_sdf): 
    """ 
    Convert SMILES to SDF using RDKit using RDKit parallel. 
    """ 
    writer = Chem.SDWriter(output_sdf)
    # try:
    mol = Chem.MolFromSmiles(input_smiles.strip())
    if mol is None:
        print(f"Warning: Invalid SMILES '{input_smiles}' encountered, skipping.")
        return ""  # Return empty string for invalid SMILES
    mol = Chem.AddHs(mol) 
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol) 
    try:
        writer.write(mol)
    except Exception as e:
        print(f"Error writing to SDF: {e}")
    print(f"Successfully converted using RDKit. Output saved to {output_sdf}") 
    writer.close()
def main():
    # Get user input 
    input_csv = input("Enter the SMILES String: ") 
    output_sdf = input("Enter the output SDF file name (e.g., output.sdf): ")
    print("Choose the conversion method:") 
    print("1. OpenBabel") 
    print("2. RDKit") 
    choice = input("Enter 1 or 2: ")

    # Extract SMILES Strings from CSV to a list
    # smiles = df[df.columns[smiles_column]].compute().tolist()

    if choice == '1': convert_with_openbabel(input_csv, output_sdf) 
    elif choice == '2': convert_with_rdkit(input_csv, output_sdf) 
    else: print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__": 
    main()