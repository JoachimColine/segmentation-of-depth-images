# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:46:49 2020

@author: Joachim
"""
import os
import bpy
from bpy import data as D
from bpy import context as C
from mathutils import *
from math import *
import numpy as np
import random
import blensor
import time

def importNewObject(objectFilePath, objectName, objectSize, objectLocation):
    if bpy.data.objects.get(objectName) is not None:
        objs = bpy.data.objects
        objs.remove(objs[objectName], do_unlink=True)
    if objectFilePath[-3:] == "obj":
        bpy.ops.import_scene.obj(filepath=objectFilePath)
    elif objectFilePath[-3:] == "stl":
        bpy.ops.import_mesh.stl(filepath=objectFilePath)
    elif objectFilePath[-3:] == "off":
        bpy.ops.import_mesh.off(filepath=objectFilePath)
    imported = bpy.context.selected_objects[0]
    imported.name = objectName
    bpy.context.scene.objects.active = bpy.data.objects[objectName]
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'ACTIVE'
    bpy.context.object.rigid_body.collision_shape = 'CONVEX_HULL'
    bpy.context.object.rigid_body.use_margin = True
    bpy.context.object.rigid_body.collision_margin = 0
    bpy.context.object.rigid_body.angular_damping = 0.5
    bpy.context.object.rigid_body.linear_damping = 0.99
    bpy.context.object.rigid_body.friction = 0.8
    bpy.context.object.rigid_body.use_deactivation = False
    bpy.context.object.rigid_body.deactivate_linear_velocity = 0
    bpy.context.object.rigid_body.deactivate_angular_velocity = 0
    imported.location = objectLocation
    currentMaxObjectSize = max(bpy.data.objects[objectName].dimensions)
    scale = objectSize / currentMaxObjectSize
    imported.scale=(scale,scale,scale)
    bpy.context.object.rigid_body.mass = 0.1
    return

def placeObject(objectName, objectLocation, objectOrientation):
    bpy.data.objects[objectName].location = objectLocation
    bpy.data.objects[objectName].rotation_euler = objectOrientation
    bpy.data.objects[objectName].rigid_body.mass = 0.1
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return

# --- Tunable parameters ---
# Objects
objectMinSize = 0.04
objectMaxSize = 0.1
objectHomePosition = (10,10,10) # where the object lands upon import (not important)
maxNumberOfObjects = 20
objectXRange = 0.075
objectYRange = 0.075

# Scanner
scannerMinHeight = 0.2
scannerMaxHeight = 0.4
scannerTiltRange = pi/18

# Simulation
speed = 1
numberOfSimulations = 2
datadirname = "dataset1"
# --------------------------------

# --- Set gravity ---
bpy.data.scenes["Scene"].gravity[2] = -9.81

# Build the ground (if it does not exist yet)
if bpy.data.objects.get("ground") is not None:
    objs = bpy.data.objects
    objs.remove(objs["ground"], do_unlink=True)
bpy.ops.mesh.primitive_cube_add(location=(0,0,-2), radius=2) 
bpy.context.active_object.name = "ground"
bpy.ops.rigidbody.object_add()
bpy.context.object.rigid_body.type = 'PASSIVE'
bpy.context.object.rigid_body.collision_shape = 'BOX'
bpy.context.object.rigid_body.friction = 0.8
bpy.context.object.rigid_body.use_margin = True
bpy.context.object.rigid_body.collision_margin = 0

# --- Place scanner ---
scanner = bpy.data.objects["Camera"]

# --- Load objects ---
# First, list all the paths to an object
rootdir = "./objects"
obj_paths = [os.path.join(dp, f) for dp, dn, filenames in 
              os.walk(rootdir) for f in filenames if 
              ((os.path.splitext(f)[1] == '.obj' and not os.path.splitext(f)[0].endswith('coll')) or
              os.path.splitext(f)[1] == '.off' or 
              os.path.splitext(f)[1] == '.stl') and
              "bookshelf" not in f and
              "laptop" not in f and
              "curtain" not in f and
              "door" not in f and
              "keyboard" not in f and
              "sink" not in f and 
              "chair" not in f and
              "stool" not in f and 
              "guitar" not in f and 
              "airplane" not in f and 
              "tent" not in f and
              "mantel" not in f and 
              "ElectronicBoard" not in f and
              "Brake" not in f and
              "bench" not in f and
              "Clutch" not in f] # these objects were found to be misleading, in that they have a part that is too thin to be detectable on a surface... 

# Run simulation
bpy.data.scenes["Scene"].rigidbody_world.steps_per_second = 60
bpy.data.scenes["Scene"].rigidbody_world.solver_iterations= 10
bpy.context.scene.rigidbody_world.time_scale=speed
numberOfObjects = 0 # Initialize number of objects.
start_time = time.time()
for sim in range(numberOfSimulations):
    # Place scanner
    randomHeight = scannerMinHeight + np.random.rand() * (scannerMaxHeight - scannerMinHeight)
    euler1 = 2*scannerTiltRange*np.random.rand() - scannerTiltRange
    euler2 = 2*scannerTiltRange*np.random.rand() - scannerTiltRange
    scanner.location = (0, 0, randomHeight)
    scanner.rotation_euler = (euler1, euler2, np.random.rand()*2*pi)
    # Remove all objects 
    for i in range(numberOfObjects):
        name = '{:04d}'.format(i+1)
        if bpy.data.objects.get(name) is not None:
            objs = bpy.data.objects
            objs.remove(objs[name], do_unlink=True)
    # Place objects
    numberOfObjects = sim % maxNumberOfObjects
    for i in range(numberOfObjects):
        name = '{:04d}'.format(i+1)
        randomPosition = (2*objectXRange*np.random.rand() - objectXRange, 2*objectYRange*np.random.rand() - objectXRange, 0.1*(i+1))
        randomOrientation = (np.random.rand()*3.14,np.random.rand()*3.14,np.random.rand()*3.14)
        objectSize = objectMinSize + np.random.rand() * (objectMaxSize - objectMinSize)
        print(objectSize)
        successfulImport = False
        while not successfulImport:
            try:
                randomPath = random.choice(obj_paths)
                importNewObject(randomPath, name, objectSize, objectHomePosition)
                successfulImport = True
            except:
                 print('failed import, trying again...') # somehow, some objects can't be loaded.      
        placeObject(name, randomPosition, randomOrientation)
    # Print progress to console:
    print("----------------------------------------------------- " + str(sim+1) + " out of " + str(numberOfSimulations))
    print("----------------------------------------------------- after %s seconds ---" % (time.time() - start_time))
    bpy.ops.ptcache.bake_all(bake=True)
    bpy.context.scene.frame_current = 250
    # Scan and save to file
    blensor.tof.scan_advanced(scanner, max_distance = 10.0, evd_file=datadirname+'/scan_'+'{:06d}'.format(sim)+'_.numpy', add_blender_mesh = False, 
                      add_noisy_blender_mesh = False, tof_res_x = 224, tof_res_y = 171, 
                      lens_angle_w=62, lens_angle_h=45, flength = 10.0,  evd_last_scan=True, 
                      noise_mu=0.0, noise_sigma=0.004, timestamp = 0.0, backfolding=False,
                      world_transformation=Matrix())
    scanData = np.loadtxt("./"+datadirname+"/scan_"+'{:06d}'.format(sim)+"_00000.numpy")
    depth_data = scanData[:,4]
    segmentation = scanData[:,11]
    concatenated = np.array([depth_data, segmentation])
    np.save("./"+datadirname+"/scan_"+str(sim), concatenated)
    # Remove raw file (it's too heavy)
    os.remove('./'+datadirname+'/scan_'+'{:06d}'.format(sim)+'_00000.numpy')
    # Clean scene
    bpy.ops.ptcache.free_bake_all()
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
print("END --- %s seconds ---" % (time.time() - start_time))
