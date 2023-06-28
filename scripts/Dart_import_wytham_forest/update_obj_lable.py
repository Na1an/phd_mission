import sys
import math
import argparse
import os.path
from xml.dom import minidom, Node

#### Utilitaries ####
def getNode(parent, nodeName, nodeNum = -1):
    """ Retourne le noeuf fils (numero:  'nodeNum' and nom 'nodeName') du noeud parent ('parent'); si nodeNum negatif, retourne la liste des noeuds fils du meme nom"""
    currentNodeNum = 0
    noeudsFils = []
    for node in parent.childNodes:
        if node.nodeType == Node.ELEMENT_NODE and node.localName == nodeName:
            if currentNodeNum == nodeNum:
                return node
            elif (nodeNum < 0):
                noeudsFils.append(node)
            currentNodeNum = currentNodeNum + 1
    return noeudsFils
    
def getNodeFromPath(path, rootNode):
    cheminXML = path.split('.')
    currentNode = rootNode
    for noeud in cheminXML:
        currentNode = getNode(currentNode, noeud, 0)
    return currentNode
#### End of utilitaries ####

if __name__ == "__main__":
    ## Get argument parser
    parser = argparse.ArgumentParser(description="Import and/or update optical properties from SOLENE into DART")

    ## Add argument
    parser.add_argument("dart_obj_3d_xml", action="store", type=str,
                        help="DART 3D object XML file path (object_3d.xml)"
                       )

    ## Parse argument
    args = parser.parse_args()
    xml_path = args.dart_obj_3d_xml
    #xml_path = "/home/yuchen/user_data/simulations/simulation_belgique/input/object_3d_no_label.xml"
    doc = minidom.parse(xml_path)
    rootNode = doc.documentElement
    ObjNode = getNodeFromPath("object_3d.ObjectList", rootNode)
    ObjNodeList = ObjNode.getElementsByTagName("Object")

    for obj in ObjNodeList:
        print("> processing :", obj.getAttribute("name"))
        GroupNodeList = getNodeFromPath("Groups", obj)
        GroupNodeList = GroupNodeList.getElementsByTagName("Group")
        
        for g in GroupNodeList:
            group_name = g.getAttribute("name")
            print(">> groupe :", group_name)
            GroupTypeProperties = g.getElementsByTagName("GroupTypeProperties")[0]
            ObjectTypeLink = GroupTypeProperties.getElementsByTagName("ObjectTypeLink")[0]
            print("[identOType] :", ObjectTypeLink.getAttribute("identOType"))
            print("[indexOT] :", ObjectTypeLink.getAttribute("indexOT"))
            if group_name == "Leaves":
                ObjectTypeLink.setAttribute("identOType", "Leaf")
                ObjectTypeLink.setAttribute("indexOT", "102")
            elif group_name == "TrunkAndBranches":
                ObjectTypeLink.setAttribute("identOType", "Trunk")
                ObjectTypeLink.setAttribute("indexOT", "103")
            else:
                print("########### Erreur")
        
    open(xml_path,"w").write(doc.toxml())