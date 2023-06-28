import argparse
import os.path
import sys
import math
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

def loadOPperGroup(opPerGroupPath):
    opPergroup = dict()
    
    with open(opPerGroupPath, 'r') as inputStream:
        currentLine = inputStream.readline()
        while currentLine != None and currentLine != "":
            split = currentLine.strip().split()
            
            if (len(split) == 2):
                opPergroup[split[0]] = split[1]
            
            currentLine = inputStream.readline()
    
    return opPergroup
    
def loadDartOPindexPerName(dartOPXMLPath) :
    doc = minidom.parse(dartOPXMLPath)
    rootNode = doc.documentElement
    lambertiansNode = getNodeFromPath("Coeff_diff.Surfaces.LambertianMultiFunctions", rootNode)
    lambertianNodeList = lambertiansNode.getElementsByTagName("LambertianMulti")
    
    lambertianNameList = dict()
    
    for i in range(len(lambertianNodeList)):
        lambertianNameList[lambertianNodeList[i].getAttribute("ident")] = i
    
    return lambertianNameList
    
def loadXMLLink(linkFilePath):
    f = open(linkFilePath, "r")
    
    lines = f.readlines()
    
    groupXMLLinks = dict()
    
    for line in lines:
        split = line.rstrip().split(';')
        print(split)
        groupXMLLinks[split[0]] = split[1]
        
    f.close()
    return groupXMLLinks
    
def updateOPIndexInObject(objXmlFilePath, OPperGroup, lambertianNameList, groupToXMLLinks):
    print(objXmlFilePath)
    doc = minidom.parse(objXmlFilePath)
    rootNode = doc.documentElement
    objectListNode = getNodeFromPath("object_3d.ObjectList", rootNode)
    
    objectList = objectListNode.getElementsByTagName("Object")
    
    for groupName, opName in OPperGroup.items():
        xmlIdSplit = groupToXMLLinks[groupName].split("_")
        
        if xmlIdSplit[0].startswith("po"):
            punctualObjectId = int(xmlIdSplit[0][2:])
            groupId = int(xmlIdSplit[1][2:])
            
            groupNode = getNode(getNode(objectList[punctualObjectId], "Groups", 0), "Group", groupId)
            #if opName == "Ground":
            #    groupNode.setAttribute("groupDEMMode", "1")
            #else:
            #    groupNode.setAttribute("groupDEMMode", "2")
            
            if opName in lambertianNameList:
                opNode = getNode(getNode(groupNode, "GroupOpticalProperties", 0), "OpticalPropertyLink", 0)
                opNode.setAttribute("ident", opName)
                opNode.setAttribute("indexFctPhase", str(lambertianNameList[opName]))
                
                opNode = getNode(getNode(getNode(groupNode, "GroupOpticalProperties", 0), "BackFaceOpticalProperty", 0), "OpticalPropertyLink", 0)
                opNode.setAttribute("ident", opName)
                opNode.setAttribute("indexFctPhase", str(lambertianNameList[opName]))
    
    open(objXmlFilePath,"w").write(doc.toxml()) # Write the xml

def main(args):
    opPerGroupPath = args.opPerGroup
    if not os.path.exists(opPerGroupPath):
        print(opPerGroupPath, "doesn't designate a valid optical property per group path")
        exit()
    elif not os.path.isfile(opPerGroupPath):
        print(opPerGroupPath, "doesn't designate a optical property per group file")
        exit()
    
    OPperGroup = loadOPperGroup(opPerGroupPath)
    
    simuFolderPath = args.simu
    if not os.path.exists(simuFolderPath):
        print(simuFolderPath, "doesn't designate a valid simulation input folder path")
        exit()
    elif not os.path.isdir(simuFolderPath):
        print(simuFolderPath, "doesn't designate a valid simulation input folder path")
        exit()
    
    coeffdiffXmlFilePath = os.path.join(simuFolderPath, "coeff_diff.xml")
    if not os.path.exists(coeffdiffXmlFilePath):
        print(simuFolderPath, "doesn't contain the optical property XML file")
        exit()
        
    lambertianNameAndIndexList = loadDartOPindexPerName(coeffdiffXmlFilePath)
    
    linkFilePath = args.link
    if not os.path.exists(linkFilePath):
        print(linkFilePath, "doesn't designate a valid link file path")
        exit()
    elif not os.path.isfile(linkFilePath):
        print(linkFilePath, "doesn't designate a link file")
        exit()
        
    groupToXMLLinks = loadXMLLink(linkFilePath)
    
    objXmlFilePath = os.path.join(simuFolderPath, "object_3d.xml")
    if not os.path.exists(objXmlFilePath):
        print(simuFolderPath, "doesn't contain the 3D object XML file")
        exit()
    
    updateOPIndexInObject(objXmlFilePath, OPperGroup, lambertianNameAndIndexList, groupToXMLLinks)

if __name__ == "__main__":
    ## Get argument parser
    parser = argparse.ArgumentParser(description="Update the optical properties of a previously imported OBJ")

    ## Add argument
    parser.add_argument("opPerGroup", action="store", type=str,
                        help="Optical properties per group"
                       )
    parser.add_argument("link", action="store", type=str,
                        help="File of the links between the OBJ and the XML"
                       )
    parser.add_argument("simu", action="store", type=str,
                        help="DART simulation input folder path"
                       )

    ## Parse argument
    args = parser.parse_args()

    ## Launch
    main(args)
