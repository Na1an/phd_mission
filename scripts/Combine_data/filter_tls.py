from utility import *

if __name__ == "__main__":
    
    # start the programme
    print("\n############## merge TLS cubes ##############\n")

    # process the arguments
    parser = argparse.ArgumentParser(description="Merge TLS labelled data.")
    parser.add_argument("folder_path", help="The path of folder which contains many cubes of TLS", type=str)
    parser.add_argument("--field", help="which field do you want to take (ex: WL)", type=str, default="WL")
    parser.add_argument("--range", help="range of field", type=str)

    args = parser.parse_args()
    folder_path = args.folder_path

    las = read_header(folder_path)
    res = las
    res.points = las[np.where(las['TreeID']>0)]
    # read data
    

    res.write(os.getcwd()+"/new_file_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".las")
    # end the programme
    print("\n########### End : merge TLS labelled data ###########\n")    

