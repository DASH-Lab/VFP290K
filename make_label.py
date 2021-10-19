import os, shutil, json
from os.path import join as opj
import numpy as np
from tqdm import tqdm
from glob import glob

import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List
import re

def read_json(path):
    with open(path, 'r') as json_file:
        json_data = json.load(json_file)
        
    return json_data

def get_split_by_video(path):
    videos = np.array([each.split('/')[-1].split('_')[0] for each in path])
    each_video = np.unique(videos)
    
    video_dict = {}
    for video_name in each_video:
        video_dict[video_name] = path[videos == video_name]
    return video_dict

def make_folders_by_target(base_path, json_file, key):
    """
    base_path: our VFP290K directory
    """
    
    os.makedirs(opj(base_path, key), exist_ok=True)
    for each_video in tqdm(json_file[key]['train'], total=len(json_file[key]['train'])):
        shutil.copytree(f'{base_path}/VFP290K/{each_video}', f'{base_path}/{key}/train/{each_video}')
        
        
    for each_video in tqdm(json_file[key]['val'], total=len(json_file[key]['val'])):
        shutil.copytree(f'{base_path}/VFP290K/{each_video}', f'{base_path}/{key}/val/{each_video}')
        
        
    for each_video in tqdm(json_file[key]['test'], total=len(json_file[key]['test'])):
        shutil.copytree(f'{base_path}/VFP290K/{each_video}', f'{base_path}/{key}/test/{each_video}')
        
        
def make_folders_for_yolov5(base_path, json_file, key):
    """
    base_path: our VFP290K directory
    """
    
    img_paths = sorted([y for x in os.walk(base_path+'/VFP290K/') for y in glob(os.path.join(x[0], '*/images/*.jpg')) ])
    xml_paths = sorted([y for x in os.walk(base_path+'/VFP290K/') for y in glob(os.path.join(x[0], '*/clean_xml/*.xml')) ])

    img_dict = get_split_by_video(np.array(list(img_paths)))
    xml_dict = get_split_by_video(np.array(list(xml_paths)))
    
    for each_video in tqdm(json_file[key]['train'], total=len(json_file[key]['train'])):
        for each_image_file, each_xml_file in zip(img_dict[each_video], xml_dict[each_video]):
            os.makedirs(f'{base_path}/yolov5/{key}/train/image', exist_ok=True)
            os.makedirs(f'{base_path}/yolov5/{key}/train/label', exist_ok=True)
            
            shutil.copy(each_image_file, f'{base_path}/yolov5/{key}/train/image/')
            shutil.copy(each_xml_file, f'{base_path}/yolov5/{key}/train/label/')
        
        
    for each_video in tqdm(json_file[key]['val'], total=len(json_file[key]['val'])):
        for each_image_file, each_xml_file in zip(img_dict[each_video], xml_dict[each_video]):    
            os.makedirs(f'{base_path}/yolov5/{key}/val/image', exist_ok=True)
            os.makedirs(f'{base_path}/yolov5/{key}/val/label', exist_ok=True)
            
            shutil.copy(each_image_file, f'{base_path}/yolov5/{key}/val/image/')
            shutil.copy(each_xml_file, f'{base_path}/yolov5/{key}/val/label/')
        
        
    for each_video in tqdm(json_file[key]['test'], total=len(json_file[key]['test'])):
        for each_image_file, each_xml_file in zip(img_dict[each_video], xml_dict[each_video]):
            os.makedirs(f'{base_path}/yolov5/{key}/test/image', exist_ok=True)
            os.makedirs(f'{base_path}/yolov5/{key}/test/label', exist_ok=True)
            
            shutil.copy(each_image_file, f'{base_path}/yolov5/{key}/test/image/')
            shutil.copy(each_xml_file, f'{base_path}/yolov5/{key}/test/label/')
        

def generate_for_coco(vfp_dir, target, task):
    rootpath = opj(vfp_dir, target, task) + '/'
    res = [y for x in os.walk(rootpath) for y in glob(opj(x[0], "*.jpg"))]
    meta = np.array([each.split("/")[-3:] for each in res])
    video_name = meta[:, 0]
    candidate = np.unique(video_name)
    np.random.seed(1)
    all_frames = []
    for each_video in candidate:
        if each_video == target:
            continue
        xmls = os.listdir(rootpath+each_video+"/clean_xml/")
        all_frames.extend(xmls)
    target_path = opj(vfp_dir, "annotations", f"{target}_{task}.txt")
    with open(target_path, "w") as f:
        for each in all_frames:
            video_name = each[:8]
            file_name = each.split(".")[0]
            target_string = opj(video_name, "clean_xml", file_name)
            f.write(target_string + "\n")
     
def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid+ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': os.path.join(filename.split('\\')[-1][:8], 'images', filename.split('\\')[-1]),
        'height': height,
        'width': width,
        'id': os.path.join(img_id.split('\\')[-1][:8], 'images', img_id.split('\\')[-1])
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    #print(label)
    #if label == '1e': return 0
    #print(label2id)
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1 
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)
        
parser = argparse.ArgumentParser()
parser.add_argument("--data_root_dir", type=str, help="data root directory")
args = parser.parse_args()

data_root_dir = args.data_root_dir
vfp_dir = opj(data_root_dir, "VFP290K")
json_dir = data_root_dir
videos = sorted(os.listdir(opj(vfp_dir, "VFP290K")))
fn_targets = {"light condition.json": ["day", "night"],
           "camera angle.json": ["low", "high"],
           "background.json": ["street", "park", "building"],
           "benchmark.json": ["benchmark"]}
tvt = ["train", "val", "test"]
os.makedirs(opj(vfp_dir, "annotations"), exist_ok=True)     


# split VFP290K by targets
for fn, targets in fn_targets.items():
    json_file = read_json(opj(json_dir, fn))
    for target in targets:
        make_folders_by_target(vfp_dir, json_file, target)
        make_folders_for_yolov5(vfp_dir, json_file, target)

# generate label files for coco
for task in tvt:
    for _, targets in fn_targets.items():
        for target in targets:
            generate_for_coco(vfp_dir, target, task)
            
# voc2coco
label_path = opj(data_root_dir, "labels.txt")
label2id = get_label2id(labels_path=label_path)
for fn, targets in fn_targets.items():
    for target in targets:
        for t in tvt:
            ann_dir = opj(vfp_dir, target, t)
            ann_ids = opj(vfp_dir, "annotations", f"{target}_{t}.txt")
            output_path = opj(vfp_dir, "annotations", f"{target}_{t}.json")
            
            ann_paths = get_annpaths(
                ann_dir_path=ann_dir,
                ann_ids_path=ann_ids,
                ext="xml",
                annpaths_list_path=None
            )
            
            convert_xmls_to_cocojson(
                annotation_paths=ann_paths,
                label2id=label2id,
                output_jsonpath=output_path,
                extract_num_from_imgid=False
            )
