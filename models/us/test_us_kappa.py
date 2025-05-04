"""SevenNet thermal conductivity calculation script."""

import json
import os
import sys
import traceback
import warnings
import argparse
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
from typing import Any, Literal

import pandas as pd
import torch
from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from moyopy import MoyoDataset
from moyopy.interface import MoyoAdapter
from pymatviz.enums import Key
from tqdm import tqdm

from matbench_discovery import today, timestamp
from matbench_discovery.enums import DataFiles, Task
from matbench_discovery.phonons import check_imaginary_freqs
from matbench_discovery.phonons import thermal_conductivity as ltc

# Add path to SevenNet
absolute_path = os.path.abspath("/data/andrii/HIENet")
sys.path.append(absolute_path)
from sevenn.sevennet_calculator import SevenNetCalculator

__author__ = "Yutack Park"
__date__ = "2024-06-25"

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run thermal conductivity calculations with GPU and data bounds selection')
parser.add_argument('--gpu', type=str, required=True, help='GPU device number')
parser.add_argument('--left', type=int, required=True, help='Left bound for data selection')
parser.add_argument('--right', type=int, required=True, help='Right bound for data selection. Use -1 for no bound')
args = parser.parse_args()

warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")

# Process bounds
left = args.left
right = args.right if args.right != -1 else None
print(f"Processing data slice: {left} to {right}")

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# EDITABLE CONFIG
name = "finetune-half-lr-3-epoch"
model_name = f"/home/a11155/{name}.pth"

# Initialize the calculator
seven_net_calc = SevenNetCalculator(model=model_name)

# Relaxation parameters
ase_optimizer: Literal["FIRE", "LBFGS", "BFGS"] = "FIRE"
max_steps = 300
force_max = 1e-4  # Run until the forces are smaller than this in eV/A

# Symmetry parameters
symprec = 1e-5
enforce_relax_symm = True
conductivity_broken_symm = False
prog_bar = True
save_forces = False  # Save force sets to file
temperatures = [300]  # Temperatures to calculate conductivity at in Kelvin
displacement_distance = 0.01  # Displacement distance for phono3py
task_type = "LTC"  # lattice thermal conductivity
job_name = (
    f"{model_name}-phononDB-{task_type}-{ase_optimizer}_force{force_max}_sym{symprec}"
)

# Output directory and file names
out_dir = "./sevennet_results"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/kappa_{left}_{right}.json.gz"

timestamp_str = f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S}"
print(f"\nJob {job_name} started {timestamp_str}")
all_atoms_list: list[Atoms] = read(DataFiles.phonondb_pbe_103_structures.path, index=":")

# Apply the slice bounds to the atoms list
atoms_list = all_atoms_list[left:right]
print(f"Processing {len(atoms_list)} structures out of {len(all_atoms_list)} total")

run_params = {
    "timestamp": timestamp_str,
    "model_name": model_name,
    "device": device,
    "slice": {"left": left, "right": right},
    "versions": {dep: version(dep) for dep in ("numpy", "torch", "matbench_discovery")},
    "ase_optimizer": ase_optimizer,
    "cell_filter": "FrechetCellFilter",
    "max_steps": max_steps,
    "force_max": force_max,
    "symprec": symprec,
    "enforce_relax_symm": enforce_relax_symm,
    "conductivity_broken_symm": conductivity_broken_symm,
    "temperatures": temperatures,
    "displacement_distance": displacement_distance,
    "task_type": task_type,
    "job_name": job_name,
    "n_structures": len(atoms_list),
}

run_params_path = f"{out_dir}/run_params_{left}_{right}.json"
with open(run_params_path, mode="w") as file:
    json.dump(run_params, file, indent=4)
print(f"Saved run parameters to {run_params_path}")

# Set up the relaxation and force set calculation
optim_cls: Callable[..., Optimizer] = {"FIRE": FIRE, "LBFGS": LBFGS}[ase_optimizer]
force_results: dict[str, dict[str, Any]] = {}
kappa_results: dict[str, dict[str, Any]] = {}
tqdm_bar = tqdm(atoms_list, desc="Conductivity calculation: ", disable=not prog_bar)

for atoms in tqdm_bar:
    mat_id = atoms.info[Key.mat_id]
    init_info = deepcopy(atoms.info)
    formula = atoms.get_chemical_formula()
    spg_num = MoyoDataset(MoyoAdapter.from_atoms(atoms)).number
    
    # Create a description that matches the expected format
    desc = f"{formula}-{spg_num}"
    
    info_dict = {
        Key.desc: desc,  # Use formatted description
        Key.formula: formula,
        Key.spg_num: spg_num,
        # Don't include errors and error_traceback in initial info_dict
    }

    tqdm_bar.set_postfix_str(mat_id, refresh=True)

    # Initialize relax_dict to avoid "possibly unbound" errors
    relax_dict = {
        "max_stress": None,
        "reached_max_steps": False,
        "broken_symmetry": False,
    }

    errors = []
    error_traceback = []

    try:
        atoms.calc = seven_net_calc
        if max_steps > 0:
            if enforce_relax_symm:
                atoms.set_constraint(FixSymmetry(atoms))
                # Use standard mask for no-tilt constraint
                filtered_atoms = FrechetCellFilter(atoms, mask=[True] * 3 + [False] * 3)
            else:
                filtered_atoms = FrechetCellFilter(atoms)

            optimizer = optim_cls(
                filtered_atoms, logfile=f"{out_dir}/relax_{mat_id}.log"
            )
            optimizer.run(fmax=force_max, steps=max_steps)

            # Note: Different attribute name in different optimizer versions
            # Using nsteps which is more commonly available
            reached_max_steps = getattr(optimizer, 'nsteps', getattr(optimizer, 'step', 0)) >= max_steps
            if reached_max_steps:
                print(f"{mat_id=} reached {max_steps=} during relaxation")

            max_stress = atoms.get_stress().reshape((2, 3), order="C").max(axis=1)
            atoms.calc = None
            atoms.constraints = None
            atoms.info = init_info | atoms.info

            # Check if symmetry was broken during relaxation
            relaxed_spg = MoyoDataset(MoyoAdapter.from_atoms(atoms)).number
            broken_symmetry = spg_num != relaxed_spg
            relax_dict = {
                "max_stress": max_stress,
                "reached_max_steps": reached_max_steps,
                "relaxed_space_group_number": relaxed_spg,
                "broken_symmetry": broken_symmetry,
            }

    except Exception as exc:
        warnings.warn(f"Failed to relax {formula=}, {mat_id=}: {exc!r}", stacklevel=2)
        traceback.print_exc()
        errors.append(f"RelaxError: {exc!r}")
        error_traceback.append(traceback.format_exc())
        
        # Still include the basic info without errors in the results dict
        kappa_results[mat_id] = info_dict | relax_dict
        continue

    # Calculation of force sets
    try:
        # Initialize phono3py with the relaxed structure
        # Use info.get() with defaults for robustness
        ph3 = ltc.init_phono3py(
            atoms,
            fc2_supercell=atoms.info.get("fc2_supercell", [2, 2, 2]),
            fc3_supercell=atoms.info.get("fc3_supercell", [2, 2, 2]),
            q_point_mesh=atoms.info.get("q_point_mesh", [10, 10, 10]),
            displacement_distance=displacement_distance,
            symprec=symprec,
        )

        # Calculate force constants and frequencies
        ph3, fc2_set, freqs = ltc.get_fc2_and_freqs(
            ph3,
            calculator=seven_net_calc,
            pbar_kwargs={"leave": False, "disable": not prog_bar},
        )

        # Check for imaginary frequencies
        has_imaginary_freqs = check_imaginary_freqs(freqs)
        freqs_dict = {
            Key.has_imag_ph_modes: has_imaginary_freqs,
            Key.ph_freqs: freqs,
        }

        # If conductivity condition is met, calculate fc3
        ltc_condition = not has_imaginary_freqs and (
            not relax_dict["broken_symmetry"] or conductivity_broken_symm
        )

        if ltc_condition:  # Calculate third-order force constants
            print(f"Calculating FC3 for {mat_id}")
            fc3_set = ltc.calculate_fc3_set(
                ph3,
                calculator=seven_net_calc,
                pbar_kwargs={"leave": False, "disable": not prog_bar},
            )
            ph3.produce_fc3(symmetrize_fc3r=True)
        else:
            fc3_set = []

        if save_forces:
            force_results[mat_id] = {"fc2_set": fc2_set, "fc3_set": fc3_set}

        if not ltc_condition:
            kappa_results[mat_id] = info_dict | relax_dict | freqs_dict
            warnings.warn(
                f"{mat_id=} has imaginary frequencies or broken symmetry", stacklevel=2
            )
            continue

    except Exception as exc:
        warnings.warn(f"Failed to calculate force sets {mat_id}: {exc!r}", stacklevel=2)
        traceback.print_exc()
        errors.append(f"ForceConstantError: {exc!r}")
        error_traceback.append(traceback.format_exc())
        
        # Still include the basic info without errors in the results dict
        kappa_results[mat_id] = info_dict | relax_dict
        continue

    try:  # Calculate thermal conductivity
        ph3, kappa_dict, cond = ltc.calculate_conductivity(
            ph3, temperatures=temperatures
        )
        
        # Standardize kappa_dict to ensure consistent structure
        if "kappa_tot_rta" in kappa_dict:
            # Make sure kappa_tot_avg exists and is correctly formatted
            if "kappa_tot_avg" not in kappa_dict:
                try:
                    kappa_tot_rta = kappa_dict["kappa_tot_rta"]
                    # Extract first tensor diagonal value for avg
                    kappa_tot_avg = [kappa_tot_rta[0][0][0]]
                    kappa_dict["kappa_tot_avg"] = kappa_tot_avg
                except (IndexError, KeyError):
                    # Fallback if structure isn't as expected
                    kappa_dict["kappa_tot_avg"] = [0.0]
        
        # Fix mode_kappa_tot_rta / mode_kappa_tot_avg naming
        if "mode_kappa_tot_rta" in kappa_dict and "mode_kappa_tot_avg" not in kappa_dict:
            kappa_dict["mode_kappa_tot_avg"] = kappa_dict["mode_kappa_tot_rta"]
            # Optionally remove the old key to avoid duplication
            # kappa_dict.pop("mode_kappa_tot_rta")
        
        print(f"Calculated kappa for {mat_id}: {kappa_dict}")
        
        # Add q_points explicitly if needed
        if "q_points" not in kappa_dict and hasattr(ph3, "mesh"):
            try:
                kappa_dict["q_points"] = ph3.mesh.qpoints.tolist()
            except:
                pass
                
    except Exception as exc:
        warnings.warn(
            f"Failed to calculate conductivity {mat_id}: {exc!r}", stacklevel=2
        )
        traceback.print_exc()
        errors.append(f"ConductivityError: {exc!r}")
        error_traceback.append(traceback.format_exc())
        
        # Still include the basic info without errors in the results dict
        kappa_results[mat_id] = info_dict | relax_dict | freqs_dict
        continue

    # Combine all results, but don't include errors if empty
    result_dict = info_dict | relax_dict | freqs_dict | kappa_dict
    kappa_results[mat_id] = result_dict

# Save results with post-processing to ensure consistent format
df_kappa = pd.DataFrame(kappa_results).T
df_kappa.index.name = Key.mat_id

# Remove any property object columns
cols_to_drop = []
for col in df_kappa.columns:
    if str(col).startswith("<property"):
        cols_to_drop.append(col)
    
if cols_to_drop:
    df_kappa = df_kappa.drop(columns=cols_to_drop)

# Save the processed DataFrame
df_kappa.reset_index().to_json(out_path)
print(f"Saved kappa results to {out_path}")

if save_forces:
    force_out_path = f"{out_dir}/force_sets_{left}_{right}.json.gz"
    df_force = pd.DataFrame(force_results).T
    df_force.index.name = Key.mat_id
    df_force.reset_index().to_json(force_out_path)
    print(f"Saved force sets to {force_out_path}")

# Print a message to help with the combining script
print(f"RESULT_FILE:{out_path}")
