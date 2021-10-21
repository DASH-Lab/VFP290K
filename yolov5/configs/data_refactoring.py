import os
import glob
import shutil
import xmltodict
import munch
import argparse
from tqdm import tqdm

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
parser = argparse.ArgumentParser()
parser.add_argument("--data_root_dir", type=str)
args = parser.parse_args()
data_root_dir = args.data_root_dir
domains = ["benchmark", "low", "high", "day", "night", "street", "park", "building"]
for domain in domains:
    training_condition_path = os.path.join(data_root_dir, f'{domain}/train')
    validation_condition_path = os.path.join(data_root_dir, f'{domain}/val')
    testing_condition_path = os.path.join(data_root_dir, f'{domain}/test')

    train_image_paths = sorted(glob.glob(os.path.join(training_condition_path, "image/*.jpg")))
    train_label_paths = sorted(glob.glob(os.path.join(training_condition_path, "label/*.xml")))
    valid_image_paths = sorted(glob.glob(os.path.join(validation_condition_path, "image/*.jpg")))
    valid_label_paths = sorted(glob.glob(os.path.join(validation_condition_path, "label/*.xml")))
    test_image_paths = sorted(glob.glob(os.path.join(testing_condition_path, "image/*.jpg")))
    test_label_paths = sorted(glob.glob(os.path.join(testing_condition_path, "label/*.xml")))

    assert len(train_image_paths) == len(train_label_paths)
    assert len(valid_image_paths) == len(valid_label_paths)
    assert len(test_image_paths) == len(test_label_paths)

    to_train_label_dir = os.path.join(training_condition_path, "image")
    to_valid_label_dir = os.path.join(validation_condition_path, "image")
    to_test_label_dir = os.path.join(testing_condition_path, "image")

    total = len(train_label_paths)
    for image_path, label_path in tqdm(zip(train_image_paths, train_label_paths), total=len(train_image_paths), leave=False):
        assert image_path.split("/")[-1].split(".")[0] == label_path.split("/")[-1].split(".")[0]
        ret_string = xml_to_string(label_path)
        f = open(os.path.join(to_train_label_dir, label_path.split("/")[-1].split(".")[0]+".txt"), 'w')
        f.write(ret_string)
        f.close()

    for image_path, label_path in tqdm(zip(valid_image_paths, valid_label_paths), total=len(valid_image_paths), leave=False):
        assert image_path.split("/")[-1].split(".")[0] == label_path.split("/")[-1].split(".")[0]
        ret_string = xml_to_string(label_path)
        f = open(os.path.join(to_valid_label_dir, label_path.split("/")[-1].split(".")[0]+".txt"), 'w')
        f.write(ret_string)
        f.close()

    for image_path, label_path in tqdm(zip(test_image_paths, test_label_paths), total=len(test_image_paths), leave=False):
        assert image_path.split("/")[-1].split(".")[0] == label_path.split("/")[-1].split(".")[0]
        ret_string = xml_to_string(label_path)
        f = open(os.path.join(to_test_label_dir, label_path.split("/")[-1].split(".")[0]+".txt"), 'w')
        f.write(ret_string)
        f.close()



