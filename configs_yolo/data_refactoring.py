import os
import glob
import shutil
import xmltodict
import munch
from tqdm import tqdm_gui

def xml_to_string(xml_path):
    txt_name = xml_path.split("/")[-1].split(".")[0] + ".txt"
    # print('hello', mp4_name)
    file = open(xml_path)
    # if os.path.exists(os.path.join(to_save_path, txt_name)):
    #     continue
    doc = xmltodict.parse(file.read())
    doc = munch.munchify(doc)
    # no_annot = None
    try:
        annot = doc.annotation.object
    except:
        return ""

    if isinstance(annot, munch.Munch):
        annot = [annot]

    if len(annot) == 0:
        pass

    ret_string = ''
    for anot in annot:
        coord = anot.bndbox
        label = anot.name
        try:
            label = int(label)
            if label not in [0, 1, 2]:
                print('error occurred in', xml_path)
                print(label, coord)
                label = input("revised label: ")
        except:
            print('error occurred in', xml_path)
            print(label, coord)
            label = input("revised label: ")
        if int(label) == 2:
            continue
        center_x = ((float(coord.xmax) + float(coord.xmin)) / 2) / 1920.0
        center_y = ((float(coord.ymax) + float(coord.ymin)) / 2) / 1080.0
        width = (float(coord.xmax) - float(coord.xmin)) / 1920.0
        height = (float(coord.ymax) - float(coord.ymin)) / 1080.0
        ret_string += '{} {} {} {} {}\n'.format(label, center_x, center_y, width, height)

    return ret_string


target_xml = 'clean_xml' # xml folder name containes label made by 'labelImg'
nips_experiment = '/VFP290K/yolov5/' # data root path
training_condition_path = os.path.join(nips_experiment, 'benchmark/train')
validation_condition_path = os.path.join(nips_experiment, 'benchmark/val')
testing_condition_path = os.path.join(nips_experiment, 'benchmark/test')
#
train_image_paths = sorted(glob.glob(os.path.join(training_condition_path, "*/images/*.jpg")))
train_label_paths = sorted(glob.glob(os.path.join(training_condition_path, "*", target_xml, "*.xml")))
valid_image_paths = sorted(glob.glob(os.path.join(validation_condition_path, "*/images/*.jpg")))
valid_label_paths = sorted(glob.glob(os.path.join(validation_condition_path, "*", target_xml, "*.xml")))
test_image_paths = sorted(glob.glob(os.path.join(testing_condition_path, "*/images/*.jpg")))
test_label_paths = sorted(glob.glob(os.path.join(testing_condition_path, "*", target_xml, "*.xml")))
#
#
src = "_".join(training_condition_path.split("/")[-2:])+"_sample"
dst = "_".join(validation_condition_path.split("/")[-2:])+"_sample"
dst2 = "_".join(testing_condition_path.split("/")[-2:])+"_sample"
#
to_save_folder = os.path.join(os.path.join(nips_experiment, "experiments"))
os.makedirs(to_save_folder, exist_ok=True)
os.makedirs(os.path.join(to_save_folder, src), exist_ok=True)
os.makedirs(os.path.join(to_save_folder, dst), exist_ok=True)
os.makedirs(os.path.join(to_save_folder, dst2), exist_ok=True)
#
train_image_fpath = os.path.join(to_save_folder, src, 'images')
train_label_fpath = os.path.join(to_save_folder, src, 'labels')
valid_image_fpath = os.path.join(to_save_folder, dst, 'images')
valid_label_fpath = os.path.join(to_save_folder, dst, 'labels')
test_image_fpath = os.path.join(to_save_folder, dst2, 'images')
test_label_fpath = os.path.join(to_save_folder, dst2, 'labels')
#
os.makedirs(train_image_fpath, exist_ok=True)
os.makedirs(train_label_fpath, exist_ok=True)
os.makedirs(valid_image_fpath, exist_ok=True)
os.makedirs(valid_label_fpath, exist_ok=True)
os.makedirs(test_image_fpath, exist_ok=True)
os.makedirs(test_label_fpath, exist_ok=True)

assert len(train_label_paths) == len(train_image_paths)
total = len(train_label_paths)
for idx, (image_path, label_path) in enumerate(zip(train_image_paths, train_label_paths)):
    assert image_path.split("/")[-1].split(".")[0] == label_path.split("/")[-1].split(".")[0]

    shutil.copy(src=image_path, dst=train_image_fpath)
    ret_string = xml_to_string(label_path)

    f = open(os.path.join(train_label_fpath, label_path.split("/")[-1].split(".")[0]+".txt"), 'w')
    f.write(ret_string)
    f.close()
    if (idx) % 501 == 0:
        print("Training:", (idx + 1), "/",  total)

print(len(valid_label_paths,), len(valid_image_paths))
print(set([p.split("/")[-1].split(".")[0] for p in valid_label_paths]) - set([p.split("/")[-1].split(".")[0] for p in valid_image_paths]))
# assert len(valid_label_paths) == len(valid_image_paths)
# #
total = len(valid_label_paths)
for idx, (image_path, label_path) in enumerate(zip(valid_image_paths, valid_label_paths)):
    assert image_path.split("/")[-1].split(".")[0] == label_path.split("/")[-1].split(".")[0]

    shutil.copy(src=image_path, dst=valid_image_fpath)
    ret_string = xml_to_string(label_path)

    f = open(os.path.join(valid_label_fpath, label_path.split("/")[-1].split(".")[0]+".txt"), 'w')
    f.write(ret_string)
    f.close()
    if (idx) % 501 == 0:
        print("Validation:", (idx + 1), "/",  total)


test_label_paths = [f.split("/")[-1].split(".")[0] for f in test_label_paths]
test_image_paths = [f.split("/")[-1].split(".")[0] for f in test_image_paths]

total = len(test_label_paths)
for idx, (image_path, label_path) in enumerate(zip(test_image_paths, test_label_paths)):
    print(image_path, label_path)
    assert image_path.split("/")[-1].split(".")[0] == label_path.split("/")[-1].split(".")[0]

    if os.path.exists(os.path.join(test_image_fpath, image_path.split("/")[-1])):
        continue
    shutil.copy(src=image_path, dst=test_image_fpath)
    ret_string = xml_to_string(label_path)

    f = open(os.path.join(test_label_fpath, label_path.split("/")[-1].split(".")[0]+".txt"), 'w')
    f.write(ret_string)
    f.close()
    if (idx) % 501 == 0:
        print("test:", (idx + 1), "/",  total)



