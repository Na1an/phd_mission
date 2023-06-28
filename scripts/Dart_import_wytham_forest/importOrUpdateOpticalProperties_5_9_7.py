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

# Expect lines with:
#    opName opDatabaseName opNameOfModelInDatabase
def loadOpticalProperties(opPath):
    print("Parsing optical properties", opPath)

    opticalProperties = dict()
    
    with open(opPath, 'r') as inputStream:
        currentLine = inputStream.readline()
        while currentLine != None and currentLine != "":
            split = currentLine.strip().split()
            
            if (len(split) == 3):
                opticalProperties[split[0]] = [split[1], split[2]]
            
            currentLine = inputStream.readline()
    
    print(" > Found", len(opticalProperties), "optical properties")
    
    return opticalProperties

def insertOrUpdateDartOpticalProperties(dartXMLPath, opticalProperties):
    print("Adding/udpating", dartXMLPath)
    doc = minidom.parse(dartXMLPath)
    rootNode = doc.documentElement
    lambertiansNode = getNodeFromPath("Coeff_diff.Surfaces.LambertianMultiFunctions", rootNode)
    lambertianNodeList = lambertiansNode.getElementsByTagName("LambertianMulti")
    
    # Recover the list of already defined optical properties
    lambertianNameList = dict()
    for i in range(len(lambertianNodeList)):
        lambertianNameList[lambertianNodeList[i].getAttribute("ident")] = i
    
    for name, op in opticalProperties.items():
        # Recover the LambertianMulti node
        if name in lambertianNameList:
            print("Updating", name)
            # An optical property is already defined. We update it.
            # We assume it is a simple lambertian OP (i.e. not Marmit for example)
            currentLambertianMultiOP = lambertianNodeList[lambertianNameList[name]]
        else :
            print("Adding", name)
            # Add a new optical property
            currentLambertianMultiOP = doc.createElement("LambertianMulti")
            lambertiansNode.appendChild(currentLambertianMultiOP)
            
            # Following is to avoid potential duplicate OPs in the XML (though it should never happen).
            currentId = len(lambertianNameList)
            lambertianNameList[name] = currentId
        
        # Recover the Lambertian node
        currentLambertianOP = getNode(currentLambertianMultiOP, "Lambertian", 0)
        if currentLambertianOP == []:
            currentLambertianOP = doc.createElement("Lambertian")
            currentLambertianMultiOP.appendChild(currentLambertianOP)
            
            currentProspectNode = doc.createElement("ProspectExternalModule")
            currentProspectNode.setAttribute("isFluorescent", "0")
            currentProspectNode.setAttribute("useProspectExternalModule", "0")
            currentLambertianOP.appendChild(currentProspectNode)
        
        # Recover the lambertianNodeMultiplicativeFactorForLUT node
        currentGlobalMultiplicativeFactorsNode = getNode(currentLambertianMultiOP, "lambertianNodeMultiplicativeFactorForLUT", 0)
        if currentGlobalMultiplicativeFactorsNode == []:
            currentGlobalMultiplicativeFactorsNode = doc.createElement("lambertianNodeMultiplicativeFactorForLUT")
            currentLambertianMultiOP.appendChild(currentGlobalMultiplicativeFactorsNode)
        
        # Recover the visible lambertianMultiplicativeFactorForLUT node
        currentVisibleMultiplicativeFactorsNode = getNode(currentGlobalMultiplicativeFactorsNode, "lambertianMultiplicativeFactorForLUT", 0)
        if currentVisibleMultiplicativeFactorsNode == []:
            currentVisibleMultiplicativeFactorsNode = doc.createElement("lambertianMultiplicativeFactorForLUT")
            currentGlobalMultiplicativeFactorsNode.appendChild(currentVisibleMultiplicativeFactorsNode)
        
        # Recover the thermal lambertianMultiplicativeFactorForLUT node
        currentThermalMultiplicativeFactorsNode = getNode(currentGlobalMultiplicativeFactorsNode, "lambertianMultiplicativeFactorForLUT", 1)
        if currentThermalMultiplicativeFactorsNode == []:
            currentThermalMultiplicativeFactorsNode = doc.createElement("lambertianMultiplicativeFactorForLUT")
            currentGlobalMultiplicativeFactorsNode.appendChild(currentThermalMultiplicativeFactorsNode)
        
        # <LambertianMulti>
        currentLambertianMultiOP.setAttribute("ident", name)
        currentLambertianMultiOP.setAttribute("lambertianDefinition", "0")
        currentLambertianMultiOP.setAttribute("useMultiplicativeFactorForLUT", "0")
        currentLambertianMultiOP.setAttribute("roStDev", "0.0")
            
        #   <Lambertian>
        currentLambertianOP.setAttribute("databaseName", op[0])
        currentLambertianOP.setAttribute("ModelName", op[1])
        currentLambertianOP.setAttribute("useSpecular", "0")
    
    open(dartXMLPath,"w").write(doc.toxml()) # Write the xml

def main(args):
    opsPath = args.ops
    if not os.path.exists(opsPath):
        print(opsPath, "doesn't designate a valid optical properties path")
        exit()
    elif not os.path.isfile(opsPath):
        print(opsPath, "doesn't designate a optical properties file")
        exit()
    
    ops = loadOpticalProperties(opsPath)
    
    dartXMLPath = args.dartXML
    if not os.path.exists(dartXMLPath):
        print(dartXMLPath, "doesn't designate a valid DART optical property path")
        exit()
    elif not os.path.isfile(dartXMLPath):
        print(dartXMLPath, "doesn't designate a DART optical property file")
        exit()
    
    insertOrUpdateDartOpticalProperties(dartXMLPath, ops)

if __name__ == "__main__":
    ## Get argument parser
    parser = argparse.ArgumentParser(description="Import and/or update optical properties from SOLENE into DART")

    ## Add argument
    parser.add_argument("ops", action="store", type=str,
                        help="Optical properties file path"
                       )
    parser.add_argument("dartXML", action="store", type=str,
                        help="DART optical properties XML file path (coeff_diff.xml)"
                       )

    ## Parse argument
    args = parser.parse_args()

    ## Launch
    main(args)
