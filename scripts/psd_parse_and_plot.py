import psd_tools.reader
import psd_tools.decoder
from psd_tools.constants import TaggedBlock, SectionDivider, ImageResourceID
import matplotlib.pyplot as plt

image_path = '../psd/Case_204_G11_40.psd'
#image_path = './Case_271_O6_40.psd'
#image_path = '/./Case_322_I3_40.psd'
im = plt.imread(image_path)

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

data_pos_dict = {}
for key,value in raw_data_pos_dict.items(): #In python 2.7 this was .iteritems()
    data_pos_dict[key] = [[float(point.items[0][1].value),float(point.items[1][1].value)] for point in value]

print (data_pos_dict.keys())
print ([len(data_pos_dict[key]) for key in data_pos_dict])
choice1 = data_pos_dict[list(data_pos_dict.keys())[0]] # In python 2.7 this was choice1 = data_pos_dict[data_pos_dict.keys()[0]]
x1,y1 = [i[0] for i in choice1],[i[1] for i in choice1]
choice2 = data_pos_dict[list(data_pos_dict.keys())[1]]
x2,y2 = [i[0] for i in choice2],[i[1] for i in choice2]
choice3 = data_pos_dict[list(data_pos_dict.keys())[2]]
x3,y3 = [i[0] for i in choice3],[i[1] for i in choice3]
choice4 = data_pos_dict[list(data_pos_dict.keys())[3]]
x4,y4 = [i[0] for i in choice4],[i[1] for i in choice4]

implot = plt.imshow(im)
plt.scatter(x1,y1,s=4,c='red') #micronuclei
plt.scatter(x2,y2,s=4,c='cyan') #acute scar
plt.scatter(x3,y3,s=4,c='violet') #subacute scar
plt.scatter(x4,y4,s=4) #unscarred
plt.show()
