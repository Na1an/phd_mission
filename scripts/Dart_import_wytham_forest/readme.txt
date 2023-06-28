Usage:
All scripts require ABSOLUTE file path

To import "myOBJ.obj" located in "/home/user/3DModels"
in the simulation "mySimulation" located in "/home/user/DART/user_data/simulations/mySimulation"
at position (20,23):
python importOBJintoSimulation_5_9_7.py /home/user/3DModels/myOBJ.obj /home/user/DART/user_data/simulations/mySimulation/input -x 20 -y 23 -o /home/user/DART/user_data/simulations/mySimulation/input/myOBJimport.txt

To import a list of lambertian optical properties in the same simulation (see example ops.txt for format):
python importOrUpdateOpticalProperties.py /home/user/3DModels/ops.txt /home/user/DART/user_data/simulations/mySimulation/input/coeff_diff.xml

To update the optical properties of the previously imported 3D obj (again see example opPerGroup.txt for format):
python updateOBJOpticalProperties.py /home/user/3DModels/opPerGroup.txt /home/user/DART/user_data/simulations/mySimulation/input/myOBJimport.txt /home/user/DART/user_data/simulations/mySimulation/input

Note: "/home/user/DART/user_data/simulations/mySimulation/input/myOBJimport.txt" was created by importOBJintoSimulation_5_9_7.py and is reused by updateOBJOpticalProperties.py
"/home/user/DART/user_data/simulations/mySimulation/input/myOBJimport.txt" can be replaced by any path at creation
