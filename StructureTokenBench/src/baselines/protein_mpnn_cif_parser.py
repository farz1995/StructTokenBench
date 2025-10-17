"""
A helper function to parse CIF file to the format required by
ProteinMPNN's featurize function. 
"""

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa


alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
            'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

three_to_one_mapping = dict(zip(alpha_3, alpha_1))

def three_to_one(x: str) -> str:
    return three_to_one_mapping[x] if x in three_to_one_mapping else "X"

def parse_cif_pdb_biounits(prot_path, atoms=['N','CA','C'], chain=None):
    '''
    input:  x = CIF filename
            atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x,y,z)), sequence
    '''

    sequence = []
    all_coords = []
    residue_index = []

    if prot_path.lower().endswith('cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", prot_path)
    model = structure[0]
    # find the right chain:
    selected_chain = None
    for peptide_chain in model.get_chains():
        if peptide_chain.id == chain:
            selected_chain = peptide_chain
        
    for residue in selected_chain.get_unpacked_list():
        if not is_aa(residue):
            continue
        if residue.get_resname() == "HOH":
            continue
        if residue.id[2] != ' ':
            continue

        res_coords = [None] * len(atoms)
        for atom in residue.get_unpacked_list():
            if atom.get_name() in atoms:
                res_coords[atoms.index(atom.get_name())] = atom.get_coord()
        coords_nan_flag = np.any([x is None for x in res_coords])
        
        if not coords_nan_flag:
            all_coords.append(res_coords)
            residue_index.append(residue.id[1])
            sequence.append(three_to_one(residue.get_resname())) 

    all_coords = np.asarray(all_coords)
    sequence = np.array(["".join(sequence)])
    residue_index = np.asarray(residue_index)
    return all_coords, sequence, residue_index

## Test
# cif_path = '/home/ec2-user/efs/structure_prediction_data/mmcifs/pdb_mmcif/mmcif_files/102l.cif'
# xyz, seq = parse_cif_pdb_biounits(cif_path, chain_id='A')
