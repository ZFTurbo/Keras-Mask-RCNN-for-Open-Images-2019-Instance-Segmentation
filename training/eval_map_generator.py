"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from keras_maskrcnn.utils.overlap import compute_overlap
from keras_maskrcnn.utils.visualization import draw_masks

import numpy as np
import os
import time

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_masks      = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        # run network
        outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes  = outputs[-4]
        scores = outputs[-3]
        labels = outputs[-2]
        masks  = outputs[-1]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_masks      = masks[0, indices[scores_sort], :, :, image_labels]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            # draw_annotations(raw_image, generator.load_annotations(i)[0], label_to_name=generator.label_to_name)
            #draw_detections(raw_image, image_boxes, image_scores, image_labels, score_threshold=score_threshold, label_to_name=generator.label_to_name)
            draw_masks(raw_image, image_boxes.astype(int), image_masks, labels=image_labels)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
            all_masks[i][label]      = image_masks[image_detections[:, -1] == label, ...]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections, all_masks


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_masks       = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)
        annotations['masks'] = np.stack(annotations['masks'], axis=0)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()
            all_masks[i][label]       = annotations['masks'][annotations['labels'] == label, ..., 0].copy()

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_annotations, all_masks


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    binarize_threshold=0.5,
    save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator          : The generator that represents the dataset to evaluate.
        model              : The model to evaluate.
        iou_threshold      : The threshold used to consider when a detection is positive or negative.
        score_threshold    : The score confidence threshold to use for detections.
        max_detections     : The maximum number of detections to use per image.
        binarize_threshold : Threshold to binarize the masks with.
        save_path          : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, all_masks     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations, all_gt_masks = _get_annotations(generator)
    average_precisions = {}

    # import pickle
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_masks, open('all_masks.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))
    # pickle.dump(all_gt_masks, open('all_gt_masks.pkl', 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = []
        true_positives  = []
        scores          = []
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            masks                = all_masks[i][label]
            annotations          = all_annotations[i][label]
            gt_masks             = all_gt_masks[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d, mask in zip(detections, masks):
                box = d[:4].astype(int)
                scores.append(d[4])

                if annotations.shape[0] == 0:
                    false_positives.append(1)
                    true_positives.append(0)
                    continue

                if box[3] > gt_masks[0].shape[0]:
                    print('Box 3 error: {} Fix {} -> {}'.format(box, box[3], gt_masks[0].shape[0]))
                    box[3] = gt_masks[0].shape[0]
                if box[2] > gt_masks[0].shape[1]:
                    print('Box 2 error: {} Fix {} -> {}'.format(box, box[2], gt_masks[0].shape[1]))
                    box[2] = gt_masks[0].shape[1]
                if box[0] < 0:
                    print('Box 0 error: {} Fix {} -> {}'.format(box, box[0], 0))
                    box[0] = 0
                if box[1] < 0:
                    print('Box 1 error: {} Fix {} -> {}'.format(box, box[1], 0))
                    box[1] = 0

                # resize to fit the box
                mask = cv2.resize(mask, (box[2] - box[0], box[3] - box[1]))

                # binarize the mask
                mask = (mask > binarize_threshold).astype(np.uint8)

                # place mask in image frame
                mask_image = np.zeros_like(gt_masks[0])
                mask_image[box[1]:box[3], box[0]:box[2]] = mask
                mask = mask_image

                overlaps            = compute_overlap(np.expand_dims(mask, axis=0), gt_masks)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives.append(0)
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives.append(1)
                    true_positives.append(0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        false_positives = np.array(false_positives, dtype=np.uint8)
        true_positives = np.array(true_positives, dtype=np.uint8)
        scores = np.array(scores)

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


class Evaluate(keras.callbacks.Callback):
    def __init__(
        self,
        generator,
        iou_threshold=0.5,
        score_threshold=0.01,
        max_detections=300,
        save_map_path=None,
        binarize_threshold=0.5,
        tensorboard=None,
        weighted_average=False,
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator          : The generator that represents the dataset to evaluate.
            iou_threshold      : The threshold used to consider when a detection is positive or negative.
            score_threshold    : The score confidence threshold to use for detections.
            max_detections     : The maximum number of detections to use per image.
            binarize_threshold : The threshold used for binarizing the masks.
            save_path          : The path to save images with visualized detections to.
            tensorboard        : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average   : Compute the mAP using the weighted average of precisions among classes.
            verbose            : Set the verbosity level, by default this is set to 1.
        """
        self.generator        = generator
        self.iou_threshold    = iou_threshold
        self.score_threshold  = score_threshold
        self.max_detections   = max_detections
        self.save_map_path    = save_map_path
        self.tensorboard      = tensorboard
        self.weighted_average = weighted_average
        self.verbose          = verbose

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        # run evaluation
        start_time = time.time()
        average_precisions = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=None
        )

        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if self.weighted_average:
            mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = mean_ap
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

        if self.verbose == 1:
            print('Time: {:.2f} mAP: {:.4f}'.format(time.time() - start_time, mean_ap))

        if self.save_map_path is not None:
            out = open(self.save_map_path, 'a')
            out.write('Ep {}: mAP: {:.4f}\n'.format(epoch + 1, mean_ap))
            out.close()

        logs['mAP'] = mean_ap
