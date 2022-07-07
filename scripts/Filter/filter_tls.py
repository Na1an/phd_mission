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

    for i in range(115,170):
        try:
            data_raw_path = folder_path + "/{}.laz".format(i)
            data_res_path = folder_path + "/filtered/{}.las".format(i)
            las = read_header(data_raw_path)
            print("> read raw data:" + data_raw_path)

            index = np.where(las.points['WL']>0)
            if(len(index[0]) < 1):
                print(">>> no points selected, len={}".format(len(index[0])))
                continue

            las.points = las.points[index]

            las.write(data_res_path)
        except ValueError:
            print(">> So tile {} has no Tree identified, skip".format(i))
            continue
        print("> write data:" + data_res_path)

    # end the programme
    print("\n########### End : merge TLS labelled data ###########\n")    

