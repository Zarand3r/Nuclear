import sys
import os

import psd_tools.reader
import psd_tools.decoder
from psd_tools.constants import TaggedBlock, SectionDivider, ImageResourceID

def psd_count_parser(image_path,file_string):
    with open(image_path, 'rb') as fp:
        binary = psd_tools.reader.parse(fp)
        decoded = psd_tools.decoder.parse(binary)

    img_dict = decoded.image_resource_blocks
    for item in img_dict:
        if item.resource_id == 1080:
            count_data=item

    count_level = count_data.data.descriptor.items[1][1][0]
    id_list = [str(item.items[3][1].value) for item in count_level]
    num_labels = len(id_list)

    raw_data_pos_dict = {}
    for i in range(num_labels):
        raw_data_pos_dict[id_list[i]] = count_level[i].items[7][1].items

    data_pos_dict = {};
    for key,value in raw_data_pos_dict.items():
        data_pos_dict[key] = [[float(point.items[0][1].value),float(point.items[1][1].value)] for point in value]

    output_file = "./label_"+file_string+".dat"
    fid = open(output_file,"w")
    for key,value in data_pos_dict.items():
        for pos in value:
            fid.write(str(key)+"\t"+str(pos[0])+"\t"+str(pos[1])+"\n")
            #print key,pos[0],pos[1]
    fid.close()
    print ("labels in ",output_file)


for file in os.listdir("./"):
    if file.endswith(".psd"):
        file_string = file[0:-4]
        psd_count_parser("./"+file,file_string)



    #print data_pos_dict.keys()
    #print [len(data_pos_dict[key]) for key in data_pos_dict]
