from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import torch.nn as nn
import base64
from io import BytesIO
from PIL import Image
from torchvision.transforms import ToPILImage
from SinglePairDataset import SinglePairDataset
from flask import send_file
from networks import ConditionGenerator, load_checkpoint, make_grid
from network_generator import SPADEGenerator
import re
import torch
import torch.nn as nn

from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
import argparse
import os
import time
from cp_dataset_test import CPDatasetTest, CPDataLoader

from networks import ConditionGenerator, load_checkpoint, make_grid
from network_generator import SPADEGenerator
from tensorboardX import SummaryWriter
from utils import *

import torchgeometry as tgm
from collections import OrderedDict

app = Flask(__name__)

# Folder paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_FOLDER = os.path.join(UPLOAD_FOLDER, 'model')
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'cloth')
OUTPUT_FOLDER = os.path.join(UPLOAD_FOLDER, 'output')  # Change output path
# Ensure upload and output folders exist
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(CLOTH_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', help='use amp')
    # Cuda availability
    parser.add_argument('--cuda', default=True, help='cuda or cpu')

    parser.add_argument('--test_name', type=str, default='server', help='test name')
    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--data_list", default="pair.txt")
    parser.add_argument("--output_dir", type=str, default="./uploads/output")
    parser.add_argument("--datasetting", default="unpaired")
    parser.add_argument("--fine_width", type=int, default=256)
    parser.add_argument("--fine_height", type=int, default=384)

    parser.add_argument('--tensorboard_dir', type=str, default='./data/zalando-hd-resize/tensorboard',
                        help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, default='./eval_models/weights/v0.1/tocg_final.pth',
                        help='tocg checkpoint')
    parser.add_argument('--gen_checkpoint', type=str, default='./eval_models/weights/v0.1/gen_model_final.pth', help='G checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    parser.add_argument('--gen_semantic_nc', type=int, default=7, help='# of input label classes without unknown class')

    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")

    # training
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'],
                        default='warp_grad')

    # Hyper-parameters
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--occlusion', action='store_true', help="Occlusion handling")

    # generator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance',
                        help='instance normalization or batch normalization')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--init_type', type=str, default='xavier',
                        help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='most',
                        # normal: 256, more: 512
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

    opt = parser.parse_args()
    return opt


# Fixed opt configuration function
def get_fixed_opt():
    class Opt:
        dataroot = UPLOAD_FOLDER
        fine_height = 384  # Example fixed height
        fine_width = 256   # Example fixed width
        semantic_nc = 13   # Number of channels for the segmentation map
        output_dir = OUTPUT_FOLDER
        tocg_checkpoint = './eval_models/weights/v0.1/tocg_final.pth'
        gen_checkpoint = './eval_models/weights/v0.1/gen_model_final.pth'
        cuda = True
        warp_feature = "T1"
        out_layer = "relu"
        clothmask_composition = 'warp_grad'
        upsample = 'bilinear'
        occlusion = True
        norm_G = 'spectralaliasinstance'
        ngf = 64
        init_type = 'xavier'
        init_variance = 0.02
        num_upsampling_layers = 'most'
        gen_semantic_nc = 7
        # Add other fixed options as needed for the dataset and model
    return Opt()


# Route to serve the HTML page
@app.route('/')
def index():
    return send_from_directory('client', 'index.html')

def update_pair(pair):
    # Filepath for 'pair.txt'
    pair_file_path = "./data/pair.txt"

    # Write the constructed pair to the file, overwriting previous content
    try:
        with open(pair_file_path, 'w') as pair_file:
            pair_file.write(pair)
        print(f"Constructed Pair written to pair.txt: {pair}")
        return jsonify({"message": "Pair updated successfully", "pair": pair}), 200
    except Exception as e:
        print(f"Failed to write pair to pair.txt. Reason: {e}")
        return jsonify({"error": "Failed to update pair.txt"}), 500

# Route to handle image uploads and virtual try-on processing
@app.route('/virtual_tryon', methods=['POST'])
def virtual_tryon():
    # Clear previous uploads
    clear_folder(MODEL_FOLDER)
    clear_folder(CLOTH_FOLDER)
    clear_folder(OUTPUT_FOLDER)


    # Save the uploaded files
    person_image = request.files['personImage']
    cloth_image = request.files['shirtImage']
    person_image_path = os.path.join(MODEL_FOLDER, person_image.filename)
    cloth_image_path = os.path.join(CLOTH_FOLDER, cloth_image.filename)
    person_image.save(person_image_path)
    cloth_image.save(cloth_image_path)



    # Construct the pair using filenames
    model_filename = person_image.filename
    cloth_filename = cloth_image.filename
    pair = f"{model_filename} {cloth_filename}"

    # Print the constructed pair
    print("Constructed Pair:", pair)

    update_pair(pair)

    # Specify the paths for the person and cloth images
    person_image_path = os.path.join('./uploads/model', person_image.filename)
    cloth_image_path = os.path.join('./uploads/cloth', cloth_image.filename)

    print("person path" + person_image_path)
    print("cloth path" + cloth_image_path)

    if not os.path.exists(person_image_path) or not os.path.exists(cloth_image_path):
        return jsonify({"error": "Person or cloth image not found in the specific folders."}), 400

    print(f"Person image path: {person_image_path}")
    print(f"Cloth image path: {cloth_image_path}")
    if not os.path.exists(person_image_path):
        print("Error: Person image file does not exist.")
        return jsonify({"error": "Person image file does not exist."}), 400
    if not os.path.exists(cloth_image_path):
        print("Error: Cloth image file does not exist.")
        return jsonify({"error": "Cloth image file does not exist."}), 400








    opt = get_opt()
    print(opt)
    print("Start to test %s!")

    # create test dataset & loader
    test_dataset = CPDatasetTest(opt)
    test_loader = CPDataLoader(opt, test_dataset)

    ## Model
    # tocg
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96,
                              norm_layer=nn.BatchNorm2d)
    # generator
    opt.semantic_nc = 7
    generator = SPADEGenerator(opt, 3 + 3 + 3)
    generator.print_network()

    # Load Checkpoint
    load_checkpoint(tocg, opt.tocg_checkpoint, opt)
    load_checkpoint_G(generator, opt.gen_checkpoint, opt)

    # Test
    test(opt, test_loader, tocg, generator)

    print("Finished testing!")

    # Use regex to create the modified pair
    modified_pair = re.sub(r'(\d+_\d+)\.jpg (\d+_\d+)\.jpg', r'\1_\2.png', pair)

    output_path = os.path.join(OUTPUT_FOLDER, modified_pair)
    output_path = os.path.abspath(output_path)
    print("output:" + output_path)

    if not os.path.exists(output_path):
        print(f"File not found at: {output_path}")
        return jsonify({"error": "Processed image not found"}), 500

    print(f"Sending file: {output_path}")
    # Generate the image (your existing processing logic goes here)
    print("Output file path:", output_path)

    # Check if the output image exists
    if not os.path.exists(output_path):
        print(f"File not found at: {output_path}")
        return jsonify({"error": "Processed image not found"}), 500

    # Send the relative path to the client
    relative_path = f"/uploads/output/{modified_pair}"
    print("relative path:", relative_path)
    return jsonify({"processed_image_path": relative_path}), 200

    #return send_file(output_path, mimetype='image/png')










# Helper function to clear the folder
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def load_checkpoint_G(model, checkpoint_path, opt):
    if not os.path.exists(checkpoint_path):
        print("Invalid path!")
        return
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict(
        [(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict.items()])
    new_state_dict._metadata = OrderedDict(
        [(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict._metadata.items()])
    model.load_state_dict(new_state_dict, strict=True)
    if opt.cuda:
        model.cuda()


# Serve static files (CSS, JS, images)
@app.route('/client/<path:filename>')
def serve_client_file(filename):
    return send_from_directory('client', filename)


# Serve uploaded files (e.g., generated output image)
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def remove_overlap(seg_out, warped_cm):
    assert len(warped_cm.shape) == 4

    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1,
                                                                                                  keepdim=True) * warped_cm
    return warped_cm


def test(opt, test_loader, tocg, generator):
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    if opt.cuda:
        gauss = gauss.cuda()

    # Model
    if opt.cuda:
        tocg.cuda()
    tocg.eval()
    generator.eval()

    if opt.output_dir is not None:
        output_dir = opt.output_dir
    else:
        output_dir = os.path.join('./output', opt.test_name,
                                  opt.datamode, opt.datasetting, 'generator', 'output')
    grid_dir = os.path.join('./output', opt.test_name,
                            opt.datamode, opt.datasetting, 'generator', 'grid')

    os.makedirs(grid_dir, exist_ok=True)

    os.makedirs(output_dir, exist_ok=True)

    num = 0
    iter_start_time = time.time()
    with torch.no_grad():
        for inputs in test_loader.data_loader:

            if opt.cuda:
                pose_map = inputs['pose'].cuda()
                pre_clothes_mask = inputs['cloth_mask'][opt.datasetting].cuda()
                label = inputs['parse']
                parse_agnostic = inputs['parse_agnostic']
                agnostic = inputs['agnostic'].cuda()
                clothes = inputs['cloth'][opt.datasetting].cuda()  # target cloth
                densepose = inputs['densepose'].cuda()
                im = inputs['image']
                input_label, input_parse_agnostic = label.cuda(), parse_agnostic.cuda()
                pre_clothes_mask = torch.FloatTensor(
                    (pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(float)).cuda()
            else:
                pose_map = inputs['pose']
                pre_clothes_mask = inputs['cloth_mask'][opt.datasetting]
                label = inputs['parse']
                parse_agnostic = inputs['parse_agnostic']
                agnostic = inputs['agnostic']
                clothes = inputs['cloth'][opt.datasetting]  # target cloth
                densepose = inputs['densepose']
                im = inputs['image']
                input_label, input_parse_agnostic = label, parse_agnostic
                pre_clothes_mask = torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(float))

            # down
            pose_map_down = F.interpolate(pose_map, size=(256, 192), mode='bilinear')
            pre_clothes_mask_down = F.interpolate(pre_clothes_mask, size=(256, 192), mode='nearest')
            input_label_down = F.interpolate(input_label, size=(256, 192), mode='bilinear')
            input_parse_agnostic_down = F.interpolate(input_parse_agnostic, size=(256, 192), mode='nearest')
            agnostic_down = F.interpolate(agnostic, size=(256, 192), mode='nearest')
            clothes_down = F.interpolate(clothes, size=(256, 192), mode='bilinear')
            densepose_down = F.interpolate(densepose, size=(256, 192), mode='bilinear')

            shape = pre_clothes_mask.shape

            # multi-task inputs
            input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
            input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

            # forward
            flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)

            # warped cloth mask one hot
            if opt.cuda:
                warped_cm_onehot = torch.FloatTensor(
                    (warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(float)).cuda()
            else:
                warped_cm_onehot = torch.FloatTensor(
                    (warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(float))

            if opt.clothmask_composition != 'no_composition':
                if opt.clothmask_composition == 'detach':
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                    fake_segmap = fake_segmap * cloth_mask

                if opt.clothmask_composition == 'warp_grad':
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                    fake_segmap = fake_segmap * cloth_mask

            # make generator input parse map
            fake_parse_gauss = gauss(
                F.interpolate(fake_segmap, size=(opt.fine_height, opt.fine_width), mode='bilinear'))
            fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

            if opt.cuda:
                old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
            else:
                old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_()
            old_parse.scatter_(1, fake_parse, 1.0)

            labels = {
                0: ['background', [0]],
                1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
                2: ['upper', [3]],
                3: ['hair', [1]],
                4: ['left_arm', [5]],
                5: ['right_arm', [6]],
                6: ['noise', [12]]
            }
            if opt.cuda:
                parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
            else:
                parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    parse[:, i] += old_parse[:, label]

            # warped cloth
            N, _, iH, iW = clothes.shape
            flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
            flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)],
                                  3)

            grid = make_grid(N, iH, iW, opt)
            warped_grid = grid + flow_norm
            warped_cloth = F.grid_sample(clothes, warped_grid, padding_mode='border')
            warped_clothmask = F.grid_sample(pre_clothes_mask, warped_grid, padding_mode='border')
            if opt.occlusion:
                warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
                warped_cloth = warped_cloth * warped_clothmask + torch.ones_like(warped_cloth) * (1 - warped_clothmask)

            output = generator(torch.cat((agnostic, densepose, warped_cloth), dim=1), parse)
            # visualize
            unpaired_names = []
            for i in range(shape[0]):
                grid = make_image_grid([(clothes[i].cpu() / 2 + 0.5), (pre_clothes_mask[i].cpu()).expand(3, -1, -1),
                                        visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i] + 1) / 2),
                                        (warped_cloth[i].cpu().detach() / 2 + 0.5),
                                        (warped_clothmask[i].cpu().detach()).expand(3, -1, -1),
                                        visualize_segmap(fake_parse_gauss.cpu(), batch=i),
                                        (pose_map[i].cpu() / 2 + 0.5), (warped_cloth[i].cpu() / 2 + 0.5),
                                        (agnostic[i].cpu() / 2 + 0.5),
                                        (im[i] / 2 + 0.5), (output[i].cpu() / 2 + 0.5)],
                                       nrow=4)
                unpaired_name = (inputs['c_name']['paired'][i].split('.')[0] + '_' +
                                 inputs['c_name'][opt.datasetting][i].split('.')[0] + '.png')
                save_image(grid, os.path.join(grid_dir, unpaired_name))
                unpaired_names.append(unpaired_name)

            # save output
            save_images(output, unpaired_names, output_dir)

            num += shape[0]
            print(num)

    print(f"Test time {time.time() - iter_start_time}")


if __name__ == '__main__':
    app.run(debug=True)
