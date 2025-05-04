# %%
import os
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatviz.enums import Key
#from sevenn.sevennet_calculator import SevenNetCalculator
from tqdm import tqdm
import zipfile
from matbench_discovery import timestamp
from matbench_discovery.data import DataFiles, as_dict_handler, ase_atoms_from_zip
from matbench_discovery.enums import Task
import sys
import os
absolute_path = os.path.abspath("/home/a11155/HIENet")
sys.path.append(absolute_path)
from sevenn.sevennet_calculator import SevenNetCalculator
#from hienet.hienet_calculator import HIENetCalculator

__author__ = "Yutack Park"
__date__ = "2024-06-25"


import argparse

parser = argparse.ArgumentParser(description='Run calculations with GPU and data bounds selection')
parser.add_argument('--gpu', type=str, required=True, help='GPU device number')
parser.add_argument('--left', type=int, required=True, help='Left bound for data selection')
parser.add_argument('--right', type=int, required=True, help='Right bound for data selection. Use -1 for no bound')
args = parser.parse_args()

# %% this config is editable
smoke_test = True


times = 0
left = args.left
right = args.right if args.right != -1 else None
print(left, right)



name = "checkpoint_3"
model_name = f"/home/a11155/{name}.pth"
#zip_filename = "/home/a11155/2024-08-04-wbm-initial-atoms.extxyz.zip"
os.makedirs(out_dir := "./results-finetuned", exist_ok=True)



task_type = Task.IS2RE
job_name = f"{model_name}-wbm-{task_type}"
ase_optimizer = "FIRE"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
ase_filter: Literal["frechet", "exp"] = "frechet"

max_steps = 500
force_max = 0.02  # Run until the forces are smaller than this in eV/A
print(force_max, max_steps)
out_path = f"{out_dir}/{name}_{force_max}_{max_steps}_{left}_{right}.json.gz"

slurm_array_task_count = 32


# %%
slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID", "debug")



data_path = {Task.IS2RE: DataFiles.wbm_initial_atoms.path}[task_type]
zip_filename = data_path
print(f"\nJob {job_name!r} running {timestamp}", flush=True)
print(f"{data_path=}", flush=True)

# Initialize ASE SevenNet Calculator from checkpoint
seven_net_calc = SevenNetCalculator(model=model_name)


# %%
print(f"Read data from {data_path}")
atoms_list = ase_atoms_from_zip(zip_filename)

if slurm_array_job_id == "debug":
    #if smoke_test:
    atoms_list = atoms_list[left:right]
    #else:
     #   pass
elif slurm_array_task_count > 1:
    atoms_list = np.array_split(atoms_list, slurm_array_task_count)[
        slurm_array_task_id - 1
    ]

relax_results: dict[str, dict[str, Any]] = {}

filter_cls: Callable[[Atoms], Atoms] = {
    "frechet": FrechetCellFilter,
    "exp": ExpCellFilter,
}[ase_filter]
optim_cls: Callable[..., Optimizer] = {"FIRE": FIRE, "LBFGS": LBFGS}[ase_optimizer]


# %%
for atoms in tqdm(atoms_list, desc="Relaxing"):
    #if 'energy' in atoms.calc.results:
    #    del atoms.calc.results['energy']  # Remove stored energy
    # Or alternatively:
    #atoms.info.pop('energy', None)  # Remove energy from atoms.info if it exists

    # energy = atoms.get_potential_energy()  # relaxed energy


    mat_id = atoms.info[Key.mat_id]
    if mat_id in relax_results:
        continue
    try:
        atoms.calc = seven_net_calc

        if max_steps > 0:
            atoms = filter_cls(atoms)
            optimizer = optim_cls(atoms, logfile="/dev/null")
            optimizer.run(fmax=force_max, steps=max_steps)
        energy = atoms.get_potential_energy()  # relaxed energy
        # if max_steps > 0, atoms is wrapped by filter_cls, so extract with getattr
        relaxed_struct = AseAtomsAdaptor.get_structure(getattr(atoms, "atoms", atoms))
        relax_results[mat_id] = {"structure": relaxed_struct, "energy": energy}
    except Exception as exc:
        print(f"Failed to relax {mat_id}: {exc!r}")
        continue

df_out = pd.DataFrame(relax_results).T.add_prefix("sevennet_")
df_out.index.name = Key.mat_id


# %%
#if not smoke_test:
df_out.reset_index().to_json(out_path, default_handler=as_dict_handler)
