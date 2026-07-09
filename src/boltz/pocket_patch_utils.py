from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from Bio.PDB import MMCIFParser
import torch

from boltz.data.types import StructureV2
from boltz.data.tokenize.boltz2 import tokenize_structure
from boltz.data import const

def _is_hydrogen(atom) -> bool:
    el = getattr(atom, "element", None)
    if el is not None and str(el).strip():
        return str(el).strip().upper() == "H"
    return atom.get_name().strip().upper().startswith("H")


def _is_water(residue) -> bool:
    return residue.get_resname().strip().upper() in {"HOH", "WAT", "H2O"}


from typing import List, Set, Tuple, Dict, Sequence
import numpy as np
from Bio.PDB import MMCIFParser


ATOM_TYPE_VOCAB: Tuple[str, ...] = (
    "C", "N", "O", "S", "P",
    "F", "Cl", "Br", "I",
    "B", "Si",
    "Na", "K", "Mg", "Ca",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Se",
    "UNK",
)


def _get_atom_type(atom) -> str:
  
    elem = getattr(atom, "element", None)
    if elem is not None:
        elem = str(elem).strip()

    if not elem:
        name = atom.get_name().strip()
        letters = "".join([c for c in name if c.isalpha()])

        if len(letters) == 0:
            elem = "UNK"
        elif len(letters) == 1:
            elem = letters.upper()
        else:
            elem = letters[:2].capitalize()
    else:
        elem = elem.capitalize()

    return elem


def _atom_types_to_onehot(
    atom_types: Sequence[str],
    vocab: Sequence[str],
) -> np.ndarray:
  
    vocab_to_idx = {t: i for i, t in enumerate(vocab)}
    unk_idx = vocab_to_idx["UNK"]

    out = np.zeros((len(atom_types), len(vocab)), dtype=np.float32)
    for i, t in enumerate(atom_types):
        j = vocab_to_idx.get(str(t), unk_idx)
        out[i, j] = 1.0
    return out


def extract_interface_patch_and_index_from_prior_complex(
    cif_path: str,
    protein_chain_names: List[str],
    partner_chain_names: List[str],
    cutoff: float = 4.5,
    atom_type_vocab: Sequence[str] = ATOM_TYPE_VOCAB,
) -> Tuple[
    Set[Tuple[str, int]],
    np.ndarray,  # protein_atom_indices
    np.ndarray,  # partner_atom_indices
    np.ndarray,  # interface_protein_atom_indices
    np.ndarray,  # interface_partner_atom_indices
    np.ndarray,  # interface_contact_pair_index, shape [2, M], local indices
    np.ndarray,  # protein_atom_types
    np.ndarray,  # partner_atom_types
    np.ndarray,  # interface_protein_atom_types
    np.ndarray,  # interface_partner_atom_types
    np.ndarray,  # protein_atom_type_onehot
    np.ndarray,  # partner_atom_type_onehot
    np.ndarray,  # interface_protein_atom_type_onehot
    np.ndarray,  # interface_partner_atom_type_onehot
    np.ndarray,  # atom_type_vocab
]:

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("known", cif_path)
    model = next(structure.get_models())

    protein_chain_set = set(protein_chain_names)
    partner_chain_set = set(partner_chain_names)

    cutoff2 = cutoff * cutoff

    global_atom_idx = 0

    protein_atom_indices: List[int] = []
    partner_atom_indices: List[int] = []

    protein_atom_types: List[str] = []
    partner_atom_types: List[str] = []

  
    global_atom_type_map: Dict[int, str] = {}

    partner_atom_coords = []
    partner_atom_global_idx = []

    # key=(chain_id, ordinal)
    # value={"coords": np.ndarray[n_atoms,3], "indices": np.ndarray[n_atoms]}
    protein_residue_atoms = {}

    for chain in model:
        chain_id = chain.id
        is_protein_chain = chain_id in protein_chain_set
        is_partner_chain = chain_id in partner_chain_set

        ordinal = -1

        for residue in chain:
            if _is_water(residue):
                continue

            if is_protein_chain:
                ordinal += 1

            current_coords = []
            current_indices = []

            for atom in residue:
                if _is_hydrogen(atom):
                    continue

                coord = atom.coord.astype(np.float32)
                atom_type = _get_atom_type(atom)

                global_atom_type_map[global_atom_idx] = atom_type

                if is_protein_chain:
                    protein_atom_indices.append(global_atom_idx)
                    protein_atom_types.append(atom_type)
                    current_coords.append(coord)
                    current_indices.append(global_atom_idx)

                elif is_partner_chain:
                    partner_atom_indices.append(global_atom_idx)
                    partner_atom_types.append(atom_type)
                    partner_atom_coords.append(coord)
                    partner_atom_global_idx.append(global_atom_idx)

                global_atom_idx += 1

            if is_protein_chain and len(current_coords) > 0:
                protein_residue_atoms[(chain_id, ordinal)] = {
                    "coords": np.asarray(current_coords, dtype=np.float32),
                    "indices": np.asarray(current_indices, dtype=np.int64),
                }

    if len(partner_atom_coords) == 0:
        raise ValueError("No partner heavy atoms found in known complex.")

    partner_atom_coords = np.asarray(partner_atom_coords, dtype=np.float32)          # [N_partner, 3]
    partner_atom_global_idx = np.asarray(partner_atom_global_idx, dtype=np.int64)    # [N_partner]

    patch: Set[Tuple[str, int]] = set()

    interface_protein_atom_set = set()
    interface_partner_atom_set = set()

   
    interface_contact_pair_global_set = set()

    for (chain_id, ordinal), data in protein_residue_atoms.items():
        prot_coords = data["coords"]      # [n_p, 3]
        prot_indices = data["indices"]    # [n_p]

        diff = prot_coords[:, None, :] - partner_atom_coords[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)  # [n_p, n_partner]

        if np.any(d2 < cutoff2):
            patch.add((chain_id, ordinal))

        prot_contact_mask = np.any(d2 < cutoff2, axis=1)
        if np.any(prot_contact_mask):
            for idx in prot_indices[prot_contact_mask]:
                interface_protein_atom_set.add(int(idx))

        partner_contact_mask = np.any(d2 < cutoff2, axis=0)
        if np.any(partner_contact_mask):
            for idx in partner_atom_global_idx[partner_contact_mask]:
                interface_partner_atom_set.add(int(idx))

     
        contact_rows, contact_cols = np.where(d2 < cutoff2)
        for r, c in zip(contact_rows.tolist(), contact_cols.tolist()):
            interface_contact_pair_global_set.add(
                (int(prot_indices[r]), int(partner_atom_global_idx[c]))
            )

    protein_atom_indices_arr = np.asarray(protein_atom_indices, dtype=np.int64)
    partner_atom_indices_arr = np.asarray(partner_atom_indices, dtype=np.int64)

    interface_protein_atom_indices = np.asarray(
        sorted(interface_protein_atom_set), dtype=np.int64
    )
    interface_partner_atom_indices = np.asarray(
        sorted(interface_partner_atom_set), dtype=np.int64
    )

 
    prot_global_to_local = {
        int(gidx): i for i, gidx in enumerate(interface_protein_atom_indices.tolist())
    }
    partner_global_to_local = {
        int(gidx): i for i, gidx in enumerate(interface_partner_atom_indices.tolist())
    }

 
    interface_contact_pair_local = sorted(
        (
            prot_global_to_local[p_g],
            partner_global_to_local[l_g],
        )
        for (p_g, l_g) in interface_contact_pair_global_set
        if p_g in prot_global_to_local and l_g in partner_global_to_local
    )

    if len(interface_contact_pair_local) == 0:
        interface_contact_pair_index = np.zeros((2, 0), dtype=np.int64)
    else:
        interface_contact_pair_index = np.asarray(
            interface_contact_pair_local, dtype=np.int64
        ).T  # [2, M]

    protein_atom_types_arr = np.asarray(protein_atom_types, dtype=object)
    partner_atom_types_arr = np.asarray(partner_atom_types, dtype=object)

    interface_protein_atom_types = np.asarray(
        [global_atom_type_map[int(idx)] for idx in interface_protein_atom_indices],
        dtype=object,
    )
    interface_partner_atom_types = np.asarray(
        [global_atom_type_map[int(idx)] for idx in interface_partner_atom_indices],
        dtype=object,
    )

    protein_atom_type_onehot = _atom_types_to_onehot(
        protein_atom_types_arr, atom_type_vocab
    )
    partner_atom_type_onehot = _atom_types_to_onehot(
        partner_atom_types_arr, atom_type_vocab
    )
    interface_protein_atom_type_onehot = _atom_types_to_onehot(
        interface_protein_atom_types, atom_type_vocab
    )
    interface_partner_atom_type_onehot = _atom_types_to_onehot(
        interface_partner_atom_types, atom_type_vocab
    )

    return (
        patch,
        protein_atom_indices_arr,
        partner_atom_indices_arr,
        interface_protein_atom_indices,
        interface_partner_atom_indices,
        interface_contact_pair_index,
        protein_atom_types_arr,
        partner_atom_types_arr,
        interface_protein_atom_types,
        interface_partner_atom_types,
        protein_atom_type_onehot,
        partner_atom_type_onehot,
        interface_protein_atom_type_onehot,
        interface_partner_atom_type_onehot,
        np.asarray(atom_type_vocab, dtype=object),
    )
    
def extract_interface_patch_and_index_from_prior_complex_v1(
    cif_path: str,
    protein_chain_names: List[str],
    partner_chain_names: List[str],
    cutoff: float = 4.5,
) -> Tuple[
    Set[Tuple[str, int]],
    np.ndarray,  # protein_atom_indices
    np.ndarray,  # partner_atom_indices
    np.ndarray,  # interface_protein_atom_indices
    np.ndarray,  # interface_partner_atom_indices
]:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("known", cif_path)
    model = next(structure.get_models())

    protein_chain_set = set(protein_chain_names)
    partner_chain_set = set(partner_chain_names)

    cutoff2 = cutoff * cutoff


    global_atom_idx = 0

    protein_atom_indices: List[int] = []
    partner_atom_indices: List[int] = []

 
    partner_atom_coords = []
    partner_atom_global_idx = []


    # key=(chain_id, ordinal)
    # value={"coords": np.ndarray[n_atoms,3], "indices": np.ndarray[n_atoms]}
    
    protein_residue_atoms = {}

    for chain in model:
        chain_id = chain.id
        is_protein_chain = chain_id in protein_chain_set
        is_partner_chain = chain_id in partner_chain_set

        ordinal = -1

        for residue in chain:
            if _is_water(residue):
                continue

            if is_protein_chain:
                ordinal += 1

            current_coords = []
            current_indices = []

            for atom in residue:
                if _is_hydrogen(atom):
                    continue

                coord = atom.coord.astype(np.float32)

                if is_protein_chain:
                    protein_atom_indices.append(global_atom_idx)
                    current_coords.append(coord)
                    current_indices.append(global_atom_idx)

                elif is_partner_chain:
                    partner_atom_indices.append(global_atom_idx)
                    partner_atom_coords.append(coord)
                    partner_atom_global_idx.append(global_atom_idx)

                global_atom_idx += 1

            if is_protein_chain and len(current_coords) > 0:
                protein_residue_atoms[(chain_id, ordinal)] = {
                    "coords": np.asarray(current_coords, dtype=np.float32),
                    "indices": np.asarray(current_indices, dtype=np.int64),
                }

    if len(partner_atom_coords) == 0:
        raise ValueError("No partner heavy atoms found in known complex.")

    partner_atom_coords = np.asarray(partner_atom_coords, dtype=np.float32)
    partner_atom_global_idx = np.asarray(partner_atom_global_idx, dtype=np.int64)

    patch: Set[Tuple[str, int]] = set()

    interface_protein_atom_set = set()
    interface_partner_atom_set = set()

  
    for (chain_id, ordinal), data in protein_residue_atoms.items():
        prot_coords = data["coords"]      # [n_p, 3]
        prot_indices = data["indices"]    # [n_p]

        diff = prot_coords[:, None, :] - partner_atom_coords[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)  # [n_p, n_partner]

        if np.any(d2 < cutoff2):
            patch.add((chain_id, ordinal))

     
        prot_contact_mask = np.any(d2 < cutoff2, axis=1)
        if np.any(prot_contact_mask):
            for idx in prot_indices[prot_contact_mask]:
                interface_protein_atom_set.add(int(idx))

        # partner 
        partner_contact_mask = np.any(d2 < cutoff2, axis=0)
        if np.any(partner_contact_mask):
            for idx in partner_atom_global_idx[partner_contact_mask]:
                interface_partner_atom_set.add(int(idx))

    return (
        patch,
        np.asarray(protein_atom_indices, dtype=np.int64),
        np.asarray(partner_atom_indices, dtype=np.int64),
        np.asarray(sorted(interface_protein_atom_set), dtype=np.int64),
        np.asarray(sorted(interface_partner_atom_set), dtype=np.int64),
    )


def extract_interface_patch_from_prior_complex(
    cif_path: str,
    protein_chain_names: List[str],
    partner_chain_names: List[str],
    cutoff: float = 4.5,
) -> Set[Tuple[str, int]]:

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("known", cif_path)
    model = next(structure.get_models())

   
    protein_chain_set = set(protein_chain_names)
    partner_chain_set = set(partner_chain_names)

  
    partner_atoms = []
    for chain in model:
        if chain.id not in partner_chain_set:
            continue
        for residue in chain:
            if _is_water(residue):
                continue
            for atom in residue:
                if not _is_hydrogen(atom):
                    partner_atoms.append(atom.coord.astype(np.float32))

    if len(partner_atoms) == 0:
        raise ValueError("No partner heavy atoms found in known complex.")

    partner_atoms = np.asarray(partner_atoms, dtype=np.float32)
    cutoff2 = cutoff * cutoff

    patch: Set[Tuple[str, int]] = set()

  
    for chain in model:
        if chain.id not in protein_chain_set:
            continue

        ordinal = -1
        for residue in chain:
            if _is_water(residue):
                continue
            ordinal += 1

          
            prot_atoms = []
            for atom in residue:
                if not _is_hydrogen(atom):
                    prot_atoms.append(atom.coord.astype(np.float32))

            if len(prot_atoms) == 0:
                continue

          
            prot_atoms = np.asarray(prot_atoms, dtype=np.float32)
            diff = prot_atoms[:, None, :] - partner_atoms[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)

         
            if np.any(d2 < cutoff2):
                patch.add((chain.id, ordinal))

    return patch


def build_protein_patch_mask_for_boltz2(
    record,
    target_dir: Path,
    patch_chain_ordinals: Set[Tuple[str, int]],
    affinity: bool = False,
) -> torch.Tensor:
    """
    protein_patch_mask:
        torch.BoolTensor, shape [n_tokens]
    """


    if affinity:
        structure_path = target_dir / record.id / f"pre_affinity_{record.id}.npz"
    else:
        structure_path = target_dir / f"{record.id}.npz"

    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    struct = StructureV2.load(structure_path)


    tokenized = tokenize_structure(struct, record.affinity)
    tokens = tokenized.tokens if hasattr(tokenized, "tokens") else tokenized[0]

  
    chain_name_to_chain_id: Dict[str, int] = {
        c.chain_name: int(c.chain_id) for c in record.chains
    }

    protein_type_id = const.chain_type_ids["PROTEIN"]
    patch_pairs = set()  # {(asym_id, res_idx)}

    valid_chains = struct.chains[struct.mask] if hasattr(struct, "mask") else struct.chains
    wanted_chain_names = {x[0] for x in patch_chain_ordinals}

    for chain in valid_chains:
        asym_id = int(chain["asym_id"])

     
        chain_name = None
        for c in record.chains:
            if int(c.chain_id) == asym_id:
                chain_name = c.chain_name
                break

        if chain_name is None:
            continue
        if chain_name not in wanted_chain_names:
            continue

    
        if int(chain["mol_type"]) != protein_type_id:
            continue

        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        chain_residues = struct.residues[res_start:res_end]

        for ordinal, res in enumerate(chain_residues):
            if (chain_name, ordinal) not in patch_chain_ordinals:
                continue
            patch_pairs.add((asym_id, int(res["res_idx"])))

 
    asym_id = np.asarray(tokens["asym_id"]).astype(np.int32)
    res_idx = np.asarray(tokens["res_idx"]).astype(np.int32)
    mol_type = np.asarray(tokens["mol_type"]).astype(np.int32)

    protein_patch_mask = np.zeros(len(asym_id), dtype=bool)

    for i in range(len(protein_patch_mask)):
        if int(mol_type[i]) != protein_type_id:
            continue
        if (int(asym_id[i]), int(res_idx[i])) in patch_pairs:
            protein_patch_mask[i] = True

    return torch.from_numpy(protein_patch_mask)


def make_batched_patch_mask(protein_patch_mask: np.ndarray, batch_size: int = 1) -> np.ndarray:
   
    mask = protein_patch_mask.astype(bool)[None, :]
    if batch_size > 1:
        mask = np.repeat(mask, batch_size, axis=0)
    return mask

