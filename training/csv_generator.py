# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from keras_maskrcnn.preprocessing.generator import Generator

import os.path
import numpy as np
import time
import pandas as pd
import cv2
import random
from PIL import Image
from keras import backend as K


def get_image_size(image_filename):
    im = Image.open(image_filename)
    w, h = im.size
    return w, h


def read_single_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img


def rle_decode(mask_rle, shape=(1024, 1024)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def read_oid_segmentation_annotations(dataset_path, csv_path, classes):
    result = {}
    start_time = time.time()
    print('Reading annotations {}'.format(csv_path))
    csv = pd.read_csv(csv_path, usecols=['MaskPath', 'ImageID', 'LabelName', 'BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax'])
    image_size_cache = dict()
    csv = csv[['MaskPath', 'ImageID', 'LabelName', 'BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax']].values
    for i in range(csv.shape[0]):
        mask_id, img_id, class_name, x1, x2, y1, y2 = csv[i]

        if 'validation' in csv_path:
            img_path = dataset_path + 'validation/' + img_id + '.jpg'
            mask_path = dataset_path + 'masks-validation-rescaled/' + mask_id[0] + '/' + mask_id
        else:
            img_path = dataset_path + 'train/' + img_id[:3] + '/' + img_id + '.jpg'
            mask_path = dataset_path + 'masks-train-rescaled/' + mask_id[0] + '/' + mask_id

        if img_path not in result:
            result[img_path] = []

        # Check that the bounding box is valid.
        if x1 < 0:
            # raise ValueError('line {}: negative x1 ({})'.format(i, x1))
            print('line {}: negative x1 ({})'.format(i, x1))
            x1 = 0
        if y1 < 0:
            # raise ValueError('line {}: negative y1 ({})'.format(i, y1))
            print('line {}: negative y1 ({})'.format(i, y1))
            y1 = 0
        if x2 > 1:
            # raise ValueError('line {}: invalid x2 ({})'.format(i, x2))
            print('line {}: invalid x2 ({})'.format(i, x2))
            x2 = 1
        if y2 > 1:
            # raise ValueError('line {}: invalid y2 ({})'.format(i, y2))
            print('line {}: invalid y2 ({})'.format(i, y2))
            y2 = 1

        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(i, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(i, y2, y1))

        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(i, class_name, classes))

        result[img_path].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name, 'mask_path': mask_path})
    print('Total images: {} Reading time: {:.2f} sec'.format(len(result), time.time() - start_time))
    return result


def get_class_index_arrays(classes_dict, image_data):
    classes = dict()
    # classes['empty'] = []
    for name in classes_dict:
        classes[classes_dict[name]] = set()

    for key in image_data:
        for entry in image_data[key]:
            c = classes_dict[entry['class']]
            classes[c] |= set([key])

    for c in classes:
        classes[c] = list(classes[c])
        print('Class ID: {} Images: {}'.format(c, len(classes[c])))

    return classes


class CSVGenerator(Generator):
    def __init__(
        self,
        csv_data_file,
        csv_class_file,
        dataset_path,
        base_dir=None,
        is_rle=False,
        **kwargs
    ):
        self.image_names = []
        self.image_data = {}
        self.dataset_path = dataset_path
        self.base_dir = base_dir
        self.is_rle = is_rle

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_data_file)

        # parse the provided class file
        clss_table = pd.read_csv(csv_class_file, header=None, names=['id', 'name'])
        self.classes = dict()
        for index, row in clss_table.iterrows():
            self.classes[row['id']] = index

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with MaskPath,LabelName,BoxID,BoxXMin,BoxXMax,BoxYMin,BoxYMax
        self.image_data = read_oid_segmentation_annotations(self.dataset_path, csv_data_file, self.classes)
        self.image_names = list(self.image_data.keys())

        self.id_to_image_id = dict([(i, k) for i, k in enumerate(self.image_names)])
        self.image_id_to_id = dict([(k, i) for i, k in enumerate(self.image_names)])
        self.class_index_array = get_class_index_arrays(self.classes, self.image_data)

        super(CSVGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_path(self, image_index):
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        return read_single_image(self.image_path(image_index))

    def load_annotations(self, image_index):
        path = self.image_names[image_index]
        annots = self.image_data[path]

        annotations     = {
            'labels': np.empty((len(annots),)),
            'bboxes': np.empty((len(annots), 4)),
            'masks': [],
        }

        for idx, annot in enumerate(annots):
            if self.is_rle is False:
                mask = cv2.imread(annot['mask_path'], cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print('Invalid mask: {}'.format(annot['mask_path']))
                    w, h = get_image_size(path)
                    mask = np.zeros((h, w), dtype=np.uint8)
            else:
                mask = rle_decode(annot['mask_path'], (1200, 1200))

            if 1:
                annotations['bboxes'][idx, 0] = float(annot['x1'] * mask.shape[1])
                annotations['bboxes'][idx, 1] = float(annot['y1'] * mask.shape[0])
                annotations['bboxes'][idx, 2] = float(annot['x2'] * mask.shape[1])
                annotations['bboxes'][idx, 3] = float(annot['y2'] * mask.shape[0])
            else:
                annotations['bboxes'][idx, 0] = float(annot['x1'])
                annotations['bboxes'][idx, 1] = float(annot['y1'])
                annotations['bboxes'][idx, 2] = float(annot['x2'])
                annotations['bboxes'][idx, 3] = float(annot['y2'])
            annotations['labels'][idx] = self.name_to_label(annot['class'])

            mask = (mask > 0).astype(np.uint8)  # convert from 0-255 to binary mask
            annotations['masks'].append(np.expand_dims(mask, axis=-1))

        return annotations

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # preprocess the image
        image = self.preprocess_image(image)

        # resize image
        image, image_scale = self.resize_image(image)

        # resize masks
        for i in range(len(annotations['masks'])):
            annotations['masks'][i], _ = self.resize_image(annotations['masks'][i])

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale

        return image, annotations

    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        # show_image(image)
        # print(image.min(), image.max())
        # print(annotations)

        if self.transform_generator and len(annotations['masks']) > 0:
            ann = dict()
            ann['image'] = image.copy()
            ann['masks'] = np.array(annotations['masks'])[:, :, :, 0].copy()
            ann['labels'] = annotations['labels'].copy()
            ann['bboxes'] = list(annotations['bboxes'])
            augm = self.transform_generator(**ann)
            image = augm['image']
            for i in range(len(annotations['masks'])):
                # show_image(255*annotations['masks'][i][:, :, 0])
                annotations['masks'][i][:, :, 0] = augm['masks'][i]
                # show_image(255*annotations['masks'][i][:, :, 0])
            annotations['bboxes'] = np.array(augm['bboxes'])

        # show_image(image)
        # print(annotations)
        # exit()

        return image, annotations

    def group_images(self):
        print('Group images. Method: {}...'.format(self.group_method))
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))
        elif self.group_method == 'random_classes':
            classes = list(range(self.num_classes()))
            self.groups = []
            while 1:
                if len(self.groups) > 1000000:
                    break
                self.groups.append([])
                for i in range(self.batch_size):
                    zz = 1000
                    while zz > 0:
                        random_class = random.choice(classes)
                        # print(random_class, len(self.class_index_array[random_class]))
                        if len(self.class_index_array[random_class]) > 0:
                            random_image = random.choice(self.class_index_array[random_class])
                            break
                        zz -= 1
                    random_image_index = self.image_id_to_id[random_image]
                    self.groups[-1].append(random_image_index)
            print('Grouped by random classes: {}'.format(len(self.groups)))
            return

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        # copy all annotations / masks to the batch
        max_annotations = max(len(a['masks']) for a in annotations_group)
        # masks_batch has shape: (batch size, max_annotations, bbox_x1 + bbox_y1 + bbox_x2 + bbox_y2 + label + width + height + max_image_dimension)
        masks_batch = np.zeros((self.batch_size, max_annotations, 5 + 2 + max_shape[0] * max_shape[1]), dtype=K.floatx())
        for index, annotations in enumerate(annotations_group):
            try:
                masks_batch[index, :annotations['bboxes'].shape[0], :4] = annotations['bboxes']
            except:
                print('Error in compute targets!')
                print(index, annotations_group)

            masks_batch[index, :annotations['labels'].shape[0], 4] = annotations['labels']
            masks_batch[index, :, 5] = max_shape[1]  # width
            masks_batch[index, :, 6] = max_shape[0]  # height

            # add flattened mask
            for mask_index, mask in enumerate(annotations['masks']):
                masks_batch[index, mask_index, 7:7 + (mask.shape[0] * mask.shape[1])] = mask.flatten()

        return list(batches) + [masks_batch]