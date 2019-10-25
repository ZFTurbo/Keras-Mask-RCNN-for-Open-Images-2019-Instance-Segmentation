# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os
    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from keras_maskrcnn import models
import cv2
import time
import glob
import pandas as pd
import numpy as np
import base64
from pycocotools import mask as coco_mask
import zlib


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_single_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img


def get_class_arr(path, type='name'):
    s = pd.read_csv(path, names=['google_name', 'name'], header=None)[type].values
    return s


def encode_binary_mask(mask):
    """Converts a binary mask into OID challenge encoding ascii text."""

    # convert input mask to expected COCO API input --
    mask_to_encode = np.expand_dims(mask, axis=2)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    # binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_SPEED)
    base64_str = base64.b64encode(binary_str)
    return base64_str


def decode_binary_mask(mask, width, height):
    """Converts a binary mask into OID challenge encoding ascii text."""

    compressed_mask = base64.b64decode(mask)
    rle_encoded_mask = zlib.decompress(compressed_mask)
    # print(rle_encoded_mask)
    decoding_dict = {
        'size': [height, width],  # [im_height, im_width],
        'counts': rle_encoded_mask
    }
    mask_tensor = coco_mask.decode(decoding_dict)
    return mask_tensor


def show_image_debug(draw, boxes, scores, labels, masks, classes):
    from keras_retinanet.utils.visualization import draw_box, draw_caption
    from keras_maskrcnn.utils.visualization import draw_mask
    from keras_retinanet.utils.colors import label_color

    # visualize detections
    limit_conf = 0.2
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        # scores are sorted so we can break
        if score < limit_conf:
            break

        color = (0, 255, 0)
        color_mask = (255, 0, 0)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        mask = mask[:, :]
        draw_mask(draw, b, mask, color=color_mask)

        caption = "{} {:.3f}".format(classes[label], score)
        draw_caption(draw, b, caption)
    draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
    show_image(draw)


def get_maskrcnn_single_predictions(model, input_image, classes, show_debug_images):
    from keras_retinanet.utils.image import preprocess_image, resize_image

    image_init = input_image.copy()

    # preprocess image for network
    image = preprocess_image(image_init)

    # Resize image
    image, image_scale = resize_image(image, min_side=800, max_side=1024)
    if show_debug_images:
        # copy to draw on
        draw, draw_scale = resize_image(image_init, min_side=800, max_side=1024)

    start = time.time()
    print('Image shape: {}'.format(image.shape))
    img_rot = image.copy()
    img_rot = np.expand_dims(img_rot, axis=0)
    outputs = model.predict_on_batch(img_rot)

    # Save only needed mask
    boxes = outputs[-4][0].copy()
    masks = outputs[-1][0].copy()
    scores = outputs[-3][0].copy()
    labels = outputs[-2][0].copy()

    # Only save needed mask to save space
    masks_reduced = []
    for i in range(masks.shape[0]):
        masks_reduced.append(masks[i, :, :, labels[i]])
    masks = np.array(masks_reduced)

    print('Detections shape: {} {} {} {}'.format(boxes.shape, scores.shape, labels.shape, masks.shape))
    print("Processing time: {:.2f} sec".format(time.time() - start))

    if show_debug_images:
        boxes_init = boxes.copy()

    boxes[:, 0] /= image.shape[1]
    boxes[:, 2] /= image.shape[1]
    boxes[:, 1] /= image.shape[0]
    boxes[:, 3] /= image.shape[0]

    if show_debug_images:
        show_image_debug(draw.astype(np.uint8), boxes_init, scores, labels, masks, classes)

    return boxes, scores, labels, masks


def get_preds_as_string(id, input_image, boxes, scores, labels, masks, classes_google):
    thr_keep_in_predictions = 0.01
    thr_mask = 0.5
    shape0, shape1 = input_image.shape[0], input_image.shape[1]
    s1 = '{},{},{},'.format(id, shape1, shape0)

    for i in range(scores.shape[0]):
        score = scores[i]

        if score < thr_keep_in_predictions:
            continue

        box = boxes[i]
        label = classes_google[labels[i]]
        mask = masks[i]

        x1 = int(box[0] * shape1)
        y1 = int(box[1] * shape0)
        x2 = int(box[2] * shape1)
        y2 = int(box[3] * shape0)
        mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

        mask[mask > thr_mask] = 1
        mask[mask <= thr_mask] = 0
        mask_complete = np.zeros((shape0, shape1), dtype=np.uint8)
        mask_complete[y1:y2, x1:x2] = mask

        enc_mask = encode_binary_mask(mask_complete)
        str1 = str(label) + ' ' + str(score) + ' '
        str1 += str(enc_mask)[2:-1] + ' '
        s1 += '{} {:.8f} {} '.format(label, score, str(enc_mask)[2:-1])

    s1 += '\n'
    return s1


def get_maskrcnn_predictions(model_path, backbone, image_files, classes_description, output_csv, show_debug_image):
    model = models.load_model(model_path, backbone_name=backbone)
    classes = get_class_arr(classes_description, type='name')
    classes_google = get_class_arr(classes_description, type='google_name')
    print('Image files to process: {}'.format(len(image_files)))

    out = open(output_csv, 'w')
    out.write('ImageID,ImageWidth,ImageHeight,PredictionString\n')
    for i in range(len(image_files)):
        inp_file = image_files[i]
        id = os.path.basename(inp_file)
        img = read_single_image(inp_file)
        if img is None:
            print('Problem reading image: {}'.format(inp_file))
            continue
        boxes, scores, labels, masks = get_maskrcnn_single_predictions(model, img, classes, show_debug_image)
        s1 = get_preds_as_string(id, img, boxes, scores, labels, masks, classes_google)
        out.write(s1)

    out.close()


if __name__ == '__main__':
    backbone = 'resnet50'
    model_path = 'mask_rcnn_resnet50_oid_v1.0.h5'
    classes_description = 'data_segmentation/challenge-2019-classes-description-segmentable.csv'

    show_debug_images = True
    image_files = glob.glob('img/*.jpg')
    output_csv = 'output.csv'

    get_maskrcnn_predictions(model_path, backbone, image_files, classes_description, output_csv, show_debug_images)
