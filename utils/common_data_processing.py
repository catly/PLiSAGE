import torch
import numpy as np
from Bio.PDB import PDBParser

# Common mapping for element types to a numeric index.
ELE2NUM = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}

def load_structure_from_pdb(fname):
    """Extracts atom coordinates and types from a PDB file."""
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    for atom in atoms:
        coords.append(atom.get_coord())
        # Map element to an index, defaulting to -1 for unknown elements.
        types.append(ELE2NUM.get(atom.element, -1))

    coords = np.stack(coords)
    # Create a one-hot encoding for the atom types.
    types_array = np.zeros((len(types), len(ELE2NUM)))
    for i, t in enumerate(types):
        if t != -1:  # Skip unknown types
            types_array[i, t] = 1.0

    atom_coords = torch.tensor(coords)
    atom_types = torch.tensor(types_array)

    return atom_coords, atom_types
