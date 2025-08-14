# Environment (created in Stage 1)

conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba

conda create -n femgnn python=3.10 -y
conda activate femgnn
conda install -c conda-forge fenics-dolfinx gmsh meshio mpi4py petsc4py slepc4py tqdm -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.6.0 torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

pip install numpy scipy matplotlib pyvista
