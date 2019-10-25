# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import argparse
import os
import sys
import cv2

import keras.preprocessing.image
import tensorflow as tf

import keras_retinanet.losses
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.model import freeze as freeze_model


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_maskrcnn.bin
    __package__ = "keras_maskrcnn.bin"


# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_maskrcnn import losses
from keras_maskrcnn import models
from training.eval_map_generator import Evaluate
from training.csv_generator import CSVGenerator
from albumentations import *


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    config.allow_soft_placement = True
    config.log_device_placement = False
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, args, freeze_backbone=False, class_specific_filter=True, anchor_params=None):
    modifier = freeze_model if freeze_backbone else None

    model = model_with_weights(
        backbone_retinanet(
            num_classes,
            nms=True,
            class_specific_filter=class_specific_filter,
            modifier=modifier,
            anchor_params=anchor_params
        ), weights=weights, skip_mismatch=True)
    training_model   = model
    prediction_model = model

    # compile model
    opt = keras.optimizers.adam(lr=1e-5, clipnorm=0.001)

    # compile model
    training_model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal(),
            'masks'         : losses.mask(),
        },
        optimizer=opt
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    callbacks = []

    # save the last prediction model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        os.makedirs(args.snapshot_path, exist_ok=True)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_fold_{fold}_last.h5'.format(backbone=args.backbone, fold=args.fold)
            ),
            verbose=1,
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    tensorboard_callback = None
    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    # Calculate mAP
    if args.evaluation and validation_generator:
        evaluation = Evaluate(validation_generator,
                              tensorboard=tensorboard_callback,
                              weighted_average=args.weighted_average,
                              save_map_path=args.snapshot_path + '/mask_rcnn_fold_{}.txt'.format(args.fold))
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save prediction model with mAP
    if args.snapshots:
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_fold_{fold}_{{mAP:.4f}}_ep_{{epoch:02d}}.h5'.format(backbone=args.backbone, fold=args.fold)
            ),
            verbose=1,
            save_best_only=False,
            monitor="mAP",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, prediction_model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.9,
        patience = 3,
        verbose  = 1,
        mode     = 'auto',
        epsilon  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    ))

    return callbacks


def create_generators(args):
    bbox_params = BboxParams(format='pascal_voc', min_area=1.0, min_visibility=0.1, label_fields=['labels'])

    transform_generator = Compose([
        HorizontalFlip(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.1),
        OneOf([
            MotionBlur(p=.1),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.1),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.1, border_mode=cv2.BORDER_CONSTANT),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.1),
        OneOf([
            RGBShift(p=1.0, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
            HueSaturationValue(p=1.0),
        ], p=0.1),
        ToGray(p=0.01),
        ImageCompression(p=0.05, quality_lower=50, quality_upper=99),
    ], bbox_params=bbox_params, p=1.0)

    train_generator = CSVGenerator(
        args.annotations,
        args.classes,
        dataset_path=args.dataset_location,
        transform_generator=transform_generator,
        batch_size=args.batch_size,
        config=args.config,
        image_min_side=800,
        image_max_side=1024,
        group_method=args.group_method,
        is_rle=False
    )

    if args.val_annotations:
        validation_generator = CSVGenerator(
            args.val_annotations,
            args.classes,
            dataset_path=args.dataset_location,
            batch_size=args.batch_size,
            config=args.config,
            image_min_side=800,
            image_max_side=1024,
            group_method=args.group_method,
            is_rle=False
        )
    else:
        validation_generator = None

    return train_generator, validation_generator


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    :param parsed_args: parser.parse_args()
    :return: parsed_args
    """

    return parsed_args


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet mask network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('dataset_location', help='Path to OID training images.')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--fold',             help='Fold number.', type=int, default=1)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--accum_iters',      help='Accum iters. If more than 1 used AdamAccum optimizer', type=int, default=1)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--no-class-specific-filter', help='Disables class specific filtering.', dest='class_specific_filter', action='store_false')
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--group_method',     help='How to form batches', default='random')

    return check_args(parser.parse_args(args))


def main(args=None):
    from keras import backend as K

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # create object that stores backbone information
    backbone = models.backbone(args.backbone)

    # optionally choose specific GPU
    if args.gpu:
        print('Use GPU: {}'.format(args.gpu))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(args)

    # create the model
    if args.snapshot is not None:
        print('Loading model {}, this may take a second...'.format(args.snapshot))
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        prediction_model = model
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        anchor_params = None

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.maskrcnn,
            num_classes=train_generator.num_classes(),
            weights=weights,
            args=args,
            freeze_backbone=args.freeze_backbone,
            class_specific_filter=args.class_specific_filter,
            anchor_params=anchor_params
        )

    # print model summary
    print(model.summary())

    print('Learning rate: {}'.format(K.get_value(model.optimizer.lr)))
    if args.lr > 0.0:
        K.set_value(model.optimizer.lr, args.lr)
        print('Updated learning rate: {}'.format(K.get_value(model.optimizer.lr)))

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    initial_epoch = 0
    if args.snapshot is not None:
        initial_epoch = int((args.snapshot.split('_')[-1]).split('.')[0])

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        max_queue_size=1,
        initial_epoch=initial_epoch,
    )

if __name__ == '__main__':
    models_path = './maskrcnn_training_models/'
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    DATASET_PATH = 'C:/Projects/2019_Google_Open_Images/input/'

    params = [
        # '--snapshot', models_path + 'mask_rcnn_resnet50_oid_v1.0.h5',
        # '--imagenet-weights',
        # '--freeze-backbone',
        '--weights', '../mask_rcnn_resnet50_oid_v1.0.h5',
        '--epochs', '1000',
        '--gpu', '2',
        '--steps', '5',
        '--snapshot-path', models_path,
        '--lr', '1e-5',
        '--backbone', 'resnet50',
        '--group_method', 'random',
        '--batch-size', '1',
        'csv',
        DATASET_PATH,
        DATASET_PATH + 'data_segmentation/challenge-2019-train-segmentation-masks.csv',
        DATASET_PATH + 'data_segmentation/challenge-2019-classes-description-segmentable.csv',
        '--val-annotations', DATASET_PATH + 'data_segmentation/challenge-2019-validation-segmentation-masks.csv',
    ]
    main(params)

