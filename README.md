# mechanisms-behind-pvs-flow
Simulation code, meshes and associated data to reproduce numerical examples presented in "Mechanisms behind perivascular fluid flow", C. Daversin-Catty, V. Vinje, K-A. Mardal, and M.E. Rognes (2020)

## Running 2D axi and 3D models<a name="models"></a>

The 2D axi and 3D simulations are performed using CFD models based on the [FEniCS project](https://fenicsproject.org/).

The corresponding Python scripts can be run within the latest [FEniCS Docker container](https://quay.io/repository/fenicsproject/dev)
with the last version of [Scipy](https://www.scipy.org/) installed :
```
git clone https://github.com/cdaversin/mechanisms-behind-pvs-flow.git
docker run -it -v $(pwd)/mechanisms-behind-pvs-flow:/home/fenics/shared quay.io/fenicsproject/dev
sudo pip3 install scipy --upgrade
cd shared
```

### 2D axi model
The 2D axi model can be configured and run using the script `2Daxi/script_2D.py`
```
cd 2Daxi
python3 script_2D.py
```

### 3D model
The 3D model can be configured and run using the script `3D/script_C0075.py`
```
cd 3D
python3 script_C0075.py
```

## Graphs
The graphs presented in the paper can be reproduced using [Jupyter notebook](https://jupyter.org/),
running the corresponding scripts in a Web browser.
```
cd mechanisms-behind-pvs-flow/notebooks
jupyter-notebook
```
Note : The data files used in the notebooks are present in the repository by default, and are re-generated
when running the models as described in the [dedicated section](#models)

## Mesh generation
The generation of the 3D PVS mesh presented in the paper is performed using [VMTK](http://www.vmtk.org/)
on a clipped geometry from [Aneurisk dataset](http://ecm2.mathcs.emory.edu/aneuriskweb/repository) (case id C0075).
The meshes in `mechanisms-behind-pvs-flow/3D/C0075_fine` to be used in our 3D model are generated using the
dedicated bash script
```
cd mechanisms-behind-pvs-flow/MeshGeneration
bash generate_meshes.sh
```
This mesh generation tool requires both [VMTK](http://www.vmtk.org/) and [FEniCS](https://fenicsproject.org/).

## Reporting issues
Any questions regarding this repository can be posted as [Issues](https://github.com/cdaversin/mechanisms-behind-pvs-flow/issues).