import psd_tools.reader
import psd_tools.decoder
from psd_tools.constants import TaggedBlock, SectionDivider, ImageResourceID

def getPositions(image_path):
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
	    data_pos_dict[key] = [[float(point.items[0][1].value),float(point.items[1][1].value), 0, 0] for point in value]

	return data_pos_dict


if __name__ == "__main__":
	data_pos_dict = getPositions('../psd/Case_204_G11_40.psd').items()
	for (key, value) in data_pos_dict:
		value[0][2] = 100
	for (key, value) in data_pos_dict:
		print (value[0])