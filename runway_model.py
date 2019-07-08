import runway
from runway.data_types import text, image

import argparse
import glob
import json
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt

from openpifpaf.network import nets
from openpifpaf import decoder, show, transforms
import datasets


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.1, seed_threshold=0.5)
    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--output-directory',
                        help=('Output directory. When using this option, make '
                              'sure input images have distinct file names.'))
    parser.add_argument('--show', default=False, action='store_true',
                        help='show image of output overlay')
    parser.add_argument('--output-types', nargs='+', default=['skeleton', 'json'],
                        help='what to output: skeleton, keypoints, json')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='processing batch size')
    parser.add_argument('--long-edge', default=None, type=int,
                        help='apply preprocessing to batch images')
    parser.add_argument('--loader-workers', default=2, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
    parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
    group = parser.add_argument_group('logging')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    args = parser.parse_args([])

    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args


def bbox_from_keypoints(kps):
    m = kps[:, 2] > 0
    if not np.any(m):
        return [0, 0, 0, 0]

    x, y = np.min(kps[:, 0][m]), np.min(kps[:, 1][m])
    w, h = np.max(kps[:, 0][m]) - x, np.max(kps[:, 1][m]) - y
    return [x, y, w, h]


setup_options = {}


@runway.setup(options=setup_options)
def setup(opts):
    args = cli()

    # load model
    model, _ = nets.factory_from_args(args)
    model = model.to(args.device)
    processor = decoder.factory_from_args(args, model)

    return model, processor


@runway.command(name='predict',
                inputs={'image': image()},
                outputs={'keypoints': text(),
                         'image': image()
                         })
def generate(m, inputs):
    args = cli()
    model, processor = m
    image = inputs["image"]

    # data
    preprocess = None
    if args.long_edge:
        preprocess = transforms.Compose([
            transforms.Normalize(),
            transforms.RescaleAbsolute(args.long_edge),
            transforms.CenterPad(args.long_edge),
        ])
    data = datasets.PilImageList([image], preprocess=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers)

    # visualizers
    keypoint_painter = show.KeypointPainter(show_box=False)
    skeleton_painter = show.KeypointPainter(show_box=False, color_connections=True,
                                            markersize=1, linewidth=6)

    image_paths, image_tensors, processed_images_cpu = next(iter(data_loader))
    images = image_tensors.permute(0, 2, 3, 1)

    processed_images = processed_images_cpu.to(args.device, non_blocking=True)
    fields_batch = processor.fields(processed_images)
    pred_batch = processor.annotations_batch(fields_batch, debug_images=processed_images_cpu)

    # unbatch
    image_path, image, processed_image_cpu, pred = image_paths[0], images[0], processed_images_cpu[0], pred_batch[0]

    processor.set_cpu_image(image, processed_image_cpu)
    keypoint_sets, scores = processor.keypoint_sets_from_annotations(pred)

    kp_json = json.dumps([
        {
            'keypoints': np.around(kps, 1).reshape(-1).tolist(),
            'bbox': bbox_from_keypoints(kps),
        }
        for kps in keypoint_sets])

    kwargs = {
        'figsize': (args.figure_width, args.figure_width * image.shape[0] / image.shape[1]),
    }
    fig = plt.figure(**kwargs)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    fig.add_axes(ax)
    ax.imshow(image)
    skeleton_painter.keypoints(ax, keypoint_sets, scores=scores)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    output_image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)

    return {
        'keypoints': kp_json,
        'image': output_image
    }


if __name__ == '__main__':
    # run the model server using the default network interface and ports,
    # displayed here for convenience
    runway.run(host='0.0.0.0', port=8000)
