# !/usr/bin/python

from __future__ import print_function  # Python 2 / 3 support
from vmtkmeshgeneratorpvs import *

from os import path
from os import system
from IPython import embed
from vmtk import vtkvmtk, vmtkscripts
import vtk
import json
import argparse

# Configuration from arguments parsing ###########################################
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', required=True, default="", help="Name of the input file (.vtp)") # VTP
parser.add_argument('-o', '--output_file', required=True, default="", help="Name of the output file") # XML
parser.add_argument('-t', '--pvs_thickness', type=float, default=0.2, help="Thickness of the generated PVS (default : 0.2)")
parser.add_argument('--non_const_thickness', type=bool, nargs='?', const=True, default=False, help="PVS thickness is constant (default) or defined as proportional (target_edge_factor)")
parser.add_argument('--target_edge_length', type=float, default=0.2, help="Constant target edge length (default : 0.2)")
parser.add_argument('--target_edge_factor', type=float, default=0.7, help="Target edge factor (default : 0.7), used when PVS thickness is set to non constant")
parser.add_argument('--nb_sub_layers', type=int, default=2, help="Number of layers (mesh) used in eacg boundary layer, 2*nb_sub_layers layers in PVS thickness (default : 2)")

args = parser.parse_args()
ifile_surface = args.input_file
ofile_mesh = args.output_file
NonConstThickness = args.non_const_thickness
TargetEdgeLength = args.target_edge_length
TargetEdgeFactor = args.target_edge_factor
Thickness_PVS = args.pvs_thickness
NumberOfSubLayers = args.nb_sub_layers

print("ifile = ", ifile_surface)
print("ofile = ", ofile_mesh)
if NonConstThickness:
    print("edge factor = ", TargetEdgeFactor)
else:
    print("edge length = ", TargetEdgeLength)
print("Thickness_PVS = ", Thickness_PVS)

# FIXME
if NonConstThickness:
    BoundaryLayerThicknessFactor = Thickness_PVS / TargetEdgeFactor
else:
    BoundaryLayerThicknessFactor = Thickness_PVS / TargetEdgeLength
# We have 2 boundary layers - half the thickness each (*0.5)
BoundaryLayerThicknessFactor = BoundaryLayerThicknessFactor*0.5
# ################################################################################

# Read vtp surface file (vtp) ##################################################
reader = vmtkscripts.vmtkSurfaceReader()
reader.InputFileName = ifile_surface
reader.Execute()
surface = reader.Surface
################################################################################

# Need to compute centerline if we want a PVS thickness depending on the radius
centerline = vmtkscripts.vmtkCenterlines()
centerline.Surface = surface
centerline.SeedSelectorName = 'openprofiles'
centerline.AppendEndPoints = 1
centerline.Execute()

if NonConstThickness:
    centerlineDistance= vmtkscripts.vmtkDistanceToCenterlines()
    centerlineDistance.Surface = surface
    centerlineDistance.Centerlines = centerline.Centerlines
    centerlineDistance.DistanceToCenterlinesArrayName = 'DistanceToCenterlines'
    centerlineDistance.UseRadiusInformation = 1
    centerlineDistance.Execute()
    surface = centerlineDistance.Surface

    vtp_writer = vtk.vtkXMLPolyDataWriter()
    vtp_writer.SetFileName(path.join(ofile_mesh + "_centerline.vtp"))
    vtp_writer.SetInputData(centerline.Centerlines)
    vtp_writer.Update()
    vtp_writer.Write()

    vtp_writer.SetFileName(path.join(ofile_mesh + "_centerline-distance.vtp"))
    vtp_writer.SetInputData(surface)
    vtp_writer.Update()
    vtp_writer.Write()


# Create FSI mesh ##############################################################
print("--- Creating artery + PVS mesh")
meshGenerator = vmtkMeshGeneratorPVS()
meshGenerator.Surface = surface
meshGenerator.SkipRemeshing = 0

if NonConstThickness:
    print(" -- Non constant PVS thickness -- ")
    meshGenerator.ElementSizeMode = 'edgelengtharray'
    meshGenerator.TargetEdgeLengthFactor = TargetEdgeFactor
    meshGenerator.TargetEdgeLengthArrayName = 'DistanceToCenterlines'
else:
    print(" -- Constant PVS thickness -- ")
    meshGenerator.ElementSizeMode = 'edgelength'
    meshGenerator.TargetEdgeLength = TargetEdgeLength
    meshGenerator.MaxEdgeLength = 1.5*meshGenerator.TargetEdgeLength
    meshGenerator.MinEdgeLength = 0.5*meshGenerator.TargetEdgeLength

# Boundary layers (PVS)
meshGenerator.BoundaryLayer = 1
meshGenerator.NumberOfSubLayers = NumberOfSubLayers # layers in each boundary layer
meshGenerator.BoundaryLayerOnCaps = 0 # No additional layer on the inlet/outlet boundaries
meshGenerator.SubLayerRatio = 1
meshGenerator.BoundaryLayerThicknessFactor = BoundaryLayerThicknessFactor
# mesh
meshGenerator.Tetrahedralize = 1

# Cells and walls numbering
meshGenerator.Vessel = 0
meshGenerator.PVS = 1
# NOTE : <Vessel/PVS>_inlet_outlet are defined as an offset for the markers
meshGenerator.Vessel_inlet_outlet = 10 #11, 12, 13...
meshGenerator.PVS_inlet_outlet = 20 #21, 22, 23, ...
meshGenerator.Vessel_wall = 30
meshGenerator.PVS_wall = 40

meshGenerator.Execute()
mesh = meshGenerator.Mesh
################################################################################

# Write mesh in VTU format #####################################################
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName(path.join(ofile_mesh + ".vtu"))
writer.SetInputData(mesh)
writer.Update()
writer.Write()
################################################################################

# Write mesh to FEniCS to format ###############################################
meshWriter = vmtkscripts.vmtkMeshWriter()
meshWriter.CellEntityIdsArrayName = "CellEntityIds"
meshWriter.CellEntityIdsOffset = 0
meshWriter.Mesh = mesh
meshWriter.OutputFileName = path.join(ofile_mesh + ".xml")
meshWriter.WriteRegionMarkers = 1
meshWriter.Execute()
################################################################################
