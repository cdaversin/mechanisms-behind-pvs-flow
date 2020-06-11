from dolfin import *
import argparse

# Configuration from arguments parsing ###########################################
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', required=True, default="", help="Name of the input file (.xml or .xml.gz)") # XML
parser.add_argument('-o', '--output_file', required=True, default="", help="Name of the output file(s) [without extension]") # XDMF
parser.add_argument('--pvs_only', type=bool, nargs='?', const=True, default=False, help="Extract the mesh of the PVS")
args = parser.parse_args()
input_filename = args.input_file
output_filename = args.output_file
pvs_only = args.pvs_only
pvs_tag = 1
##################################################################################

xml_input = File(input_filename)
mesh = Mesh()
xml_input >> mesh

xdmf = XDMFFile(MPI.comm_world, output_filename + ".xdmf")
xdmf_mc = XDMFFile(MPI.comm_world, output_filename + "_mc.xdmf")
xdmf_mf = XDMFFile(MPI.comm_world, output_filename + "_mf.xdmf")

# MeshFunction used with given mesh.domains() builds MeshFunction of given dim d
# with the values contained in mesh.domains().markers(d), read from the XML file
# with xml node : <domains> <mesh_value_collection, dim=d>
cells_mf = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
facets_mf = MeshFunction("size_t", mesh, mesh.topology().dim()-1, mesh.domains())

if pvs_only:
    print("Extracting PVS mesh")
    mesh_pvs = SubMesh(mesh, cells_mf, pvs_tag)
    pvs_cells_mf = MeshFunction("size_t", mesh_pvs, mesh_pvs.topology().dim(), mesh_pvs.domains())
    pvs_facets_mf = MeshFunction("size_t", mesh_pvs, mesh_pvs.topology().dim()-1, mesh_pvs.domains())

    # Mesh
    xdmf.write(mesh_pvs)
    # Cell markers
    xdmf_mc.write(pvs_cells_mf)
    # Facet markers
    xdmf_mf.write(pvs_facets_mf)
else:
    # Mesh
    xdmf.write(mesh)
    # Cell markers
    xdmf_mc.write(cells_mf)
    # Facet markers
    xdmf_mf.write(facets_mf)
