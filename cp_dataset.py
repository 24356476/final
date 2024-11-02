import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import json
import os

import os.path as osp
import numpy as np


class CPDataset(data.Dataset):
    """
        Dataset for CP-VTON.
    """

    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode  # train or test or self-defined
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Add logs to check
        print(f"Fine Height: {self.fine_height}")
        print(f"Fine Width: {self.fine_width}")

        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = dict()
        self.c_names['paired'] = im_names
        self.c_names['unpaired'] = c_names

    def name(self):
        return "CPDataset"


    def get_agnostic(self, im, im_parse, pose_data):
        parse_array = np.array(im_parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        agnostic = im.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)





        # Save original image and parse mask
        #agnostic_save_dir = './agnostic_visualizations'
        #os.makedirs(agnostic_save_dir, exist_ok=True)
        #im.save(os.path.join(agnostic_save_dir, 'original_image.png'))
        #im_parse.save(os.path.join(agnostic_save_dir, 'original_parse.png'))





        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        #r = int(length_a / 25) + 1
        #r = int(length_a / 16) + 1
        r_neck = int(length_a / 25) + 1  # Adjust this divisor for the neck
        r_torso = int(length_a / 50) + 1  # Adjust this divisor specifically for the torso
        r_arms = int(length_a / 25) + 1  # Adjust this for the arms if needed

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r_torso * 3, pointy - r_torso * 6, pointx + r_torso * 3, pointy + r_torso * 6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r_torso * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r_torso * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r_torso * 12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')





        # Save torso-masked image
        #agnostic.save(os.path.join(agnostic_save_dir, 'torso_masked.png'))




        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx - r_neck * 5, pointy - r_neck * 9, pointx + r_neck * 5, pointy), 'gray', 'gray')






        # Save neck-masked image
        #agnostic.save(os.path.join(agnostic_save_dir, 'neck_masked.png'))






        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r_arms * 12)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r_arms * 5, pointy - r_arms * 6, pointx + r_arms * 5, pointy + r_arms * 6), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                    pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r_arms * 10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r_arms * 5, pointy - r_arms * 5, pointx + r_arms * 5, pointy + r_arms * 5), 'gray', 'gray')

        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'white')  # Corrected dimensions
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            pointx, pointy = pose_data[pose_ids[0]]
            mask_arm_draw.ellipse((pointx - r_arms * 5, pointy - r_arms * 6, pointx + r_arms * 5, pointy + r_arms * 6), 'black', 'black')
            for i in pose_ids[1:]:
                if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                        pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r_arms * 10)
                pointx, pointy = pose_data[i]
                if i != pose_ids[-1]:
                    mask_arm_draw.ellipse((pointx - r_arms * 5, pointy - r_arms * 5, pointx + r_arms * 5, pointy + r_arms * 5), 'black',
                                          'black')

            # Save arm-masked image
            #agnostic.save(os.path.join(agnostic_save_dir, 'arms_masked.png'))

            # Save final agnostic image after all masking
            #agnostic.save(os.path.join(agnostic_save_dir, 'final_agnostic.png'))

            # Resize parse_array to match mask_arm size if needed
            parse_array_resized = np.array(
                transforms.Resize((self.fine_height, self.fine_width))(im_parse))  # Ensure consistent dimensions

            #print(f"mask_arm shape: {mask_arm.size}, parse_array_resized shape: {parse_array_resized.shape}")
            # Perform the operation
            parse_arm = (np.array(mask_arm) / 255) * (parse_array_resized == parse_id).astype(np.float32)

            #print(f"Agnostic size: {agnostic.size}, Person image size: {im.size}, Parse arm size: {parse_arm.shape}")

            agnostic_resized = agnostic.resize((self.fine_width, self.fine_height), Image.NEAREST)
            im_resized = im.resize((self.fine_width, self.fine_height), Image.NEAREST)
            parse_arm_resized = Image.fromarray(np.uint8(parse_arm * 255), 'L').resize(im_resized.size)
            #print(f"parse_arm_resized shape: {parse_arm_resized.size}, im shape: {im_resized.size}")
            #print(f"Agnostic size: {agnostic_resized.size}")

            agnostic_resized.paste(im_resized, None, parse_arm_resized)
            # im size and parse arm size are wrong and maybe agnostic too
            # agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
            ################################################################################################################
            # Ensure these operations happen after arms are processed
            parse_head_resized = Image.fromarray(np.uint8(parse_head * 255), 'L').resize(im_resized.size, Image.NEAREST)
            parse_lower_resized = Image.fromarray(np.uint8(parse_lower * 255), 'L').resize(im_resized.size,
                                                                                           Image.NEAREST)
            agnostic_resized.paste(im_resized, None, parse_head_resized)
            agnostic_resized.paste(im_resized, None, parse_lower_resized)
            #agnostic_resized.paste(im_resized, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
            #agnostic_resized.paste(im_resized, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
            #print(
            #    f"parse_head_resized size: {parse_head_resized.size}, parse_lower_resized size: {parse_lower_resized.size}, im_resized size: {im_resized.size}")
            #print(f"Agnostic resized size: {agnostic_resized.size}")
            #print(f"Agnostic size: {agnostic.size}")

        return agnostic

    def __getitem__(self, index):
        im_name = self.im_names[index]
        im_name = 'image/' + im_name
        c_name = {}
        c = {}
        cm = {}
        for key in ['paired']:
            c_name[key] = self.c_names[key][index]
            c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
            c[key] = transforms.Resize((self.fine_height, self.fine_width), interpolation=2)(c[key])
            cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
            cm[key] = transforms.Resize((self.fine_height, self.fine_width), interpolation=0)(cm[key])

            c[key] = self.transform(c[key])  # [-1,1]
            cm_array = np.array(cm[key])
            cm_array = (cm_array >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_array)  # [0,1]
            cm[key].unsqueeze_(0)

        # person image
        im_pil_big = Image.open(osp.join(self.data_path, im_name))
        im_pil = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.BICUBIC)(im_pil_big)
        #print(f"Resized person image size: {im_pil.size}")

        im = self.transform(im_pil)



        # load parsing image
        parse_name = im_name.replace('image', 'image-parse-v3').replace('.jpg', '.png')
        im_parse_pil_big = Image.open(osp.join(self.data_path, parse_name))
        im_parse_pil = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.NEAREST)(
            im_parse_pil_big)
        #print(f"Resized parsing image size: {im_parse_pil.size}")
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        im_parse = self.transform(im_parse_pil.convert('RGB'))

        # parse map
        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

        parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        # load image-parse-agnostic
        image_parse_agnostic = Image.open(
            osp.join(self.data_path, parse_name.replace('image-parse-v3', 'image-parse-agnostic-v3.2')))
        image_parse_agnostic = transforms.Resize(self.fine_width, interpolation=0)(image_parse_agnostic)
        #print(f"Resized image-parse-agnostic size: {image_parse_agnostic.size}")
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        image_parse_agnostic = self.transform(image_parse_agnostic.convert('RGB'))

        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

                # parse cloth & parse cloth mask
        pcm = new_parse_map[3:4]
        pcm = torch.nn.functional.interpolate(pcm.unsqueeze(0), size=(self.fine_height, self.fine_width),
                                              mode='nearest').squeeze(0)

        #pcm = torch.nn.functional.interpolate(pcm.unsqueeze(0), size=(self.fine_height, self.fine_width), mode='nearest').squeeze(0)
        im = torch.nn.functional.interpolate(im.unsqueeze(0), size=(self.fine_height, self.fine_width),
                                             mode='bilinear').squeeze(0)

        #print(" coming from cp_dataset.py getitem in CPDataset class")
        #print(f"im shape: {im.shape}")
        #print(f"pcm shape: {pcm.shape}")
        im_c = im * pcm + (1 - pcm)

        # load pose points
        pose_name = im_name.replace('image', 'openpose_img').replace('.jpg', '_rendered.png')
        pose_map = Image.open(osp.join(self.data_path, pose_name))
        pose_map = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.BICUBIC)(pose_map)
        #print(f"Resized pose map size: {pose_map.size}")

        pose_map = self.transform(pose_map)  # [-1,1]

        # pose name
        pose_name = im_name.replace('image', 'openpose_json').replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

        # load densepose
        densepose_name = im_name.replace('image', 'image-densepose')
        densepose_map = Image.open(osp.join(self.data_path, densepose_name))
        densepose_map = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.BICUBIC)(
            densepose_map)
        #print(f"Resized densepose map size: {densepose_map.size}")

        densepose_map = self.transform(densepose_map)  # [-1,1]

        # agnostic
        agnostic = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.BICUBIC)(agnostic)
        #print(f"Resized agnostic image size: {agnostic.size}")

        agnostic = self.transform(agnostic)

        result = {
            'c_name': c_name,  # for visualization
            'im_name': im_name,  # for visualization or ground truth
            # intput 1 (clothfloww)
            'cloth': c,  # for input
            'cloth_mask': cm,  # for input
            # intput 2 (segnet)
            'parse_agnostic': new_parse_agnostic_map,
            'densepose': densepose_map,
            'pose': pose_map,  # for conditioning
            # generator input
            'agnostic': agnostic,
            # GT
            'parse_onehot': parse_onehot,  # Cross Entropy
            'parse': new_parse_map,  # GAN Loss real
            'pcm': pcm,  # L1 Loss & vis
            'parse_cloth': im_c,  # VGG Loss & vis
            # visualization & GT
            'image': im,  # for visualization
        }

        return result

    def __len__(self):
        return len(self.im_names)


class CPDatasetTest(data.Dataset):
    """
        Test Dataset for CP-VTON.
    """

    def __init__(self, opt):
        super(CPDatasetTest, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode  # train or test or self-defined
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = dict()
        self.c_names['paired'] = im_names
        self.c_names['unpaired'] = c_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        im_name = self.im_names[index]
        c_name = {}
        c = {}
        cm = {}
        for key in self.c_names:
            c_name[key] = self.c_names[key][index]
            c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
            c[key] = transforms.Resize(self.fine_width, interpolation=2)(c[key])
            cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
            cm[key] = transforms.Resize((self.fine_height, self.fine_width), interpolation=0)(cm[key])

            c[key] = self.transform(c[key])  # [-1,1]
            cm_array = np.array(cm[key])
            cm_array = (cm_array >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_array)  # [0,1]
            cm[key].unsqueeze_(0)

        # person image
        im_pil_big = Image.open(osp.join(self.data_path, 'image', im_name))
        im_pil = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.BICUBIC)(im_pil_big)
        im = self.transform(im_pil)



        # load parsing image
        parse_name = im_name.replace('image', 'image-parse-v3').replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse-v3', parse_name))
        im_parse = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.NEAREST)(
            im_parse)
        parse = torch.from_numpy(np.array(im_parse)[None]).long()
        im_parse = self.transform(im_parse.convert('RGB'))


        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

        parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        # load image-parse-agnostic
        image_parse_agnostic = Image.open(osp.join(self.data_path, 'image-parse-agnostic-v3.2', parse_name))
        image_parse_agnostic = transforms.Resize(self.fine_width, interpolation=0)(image_parse_agnostic)
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        image_parse_agnostic = self.transform(image_parse_agnostic.convert('RGB'))

        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

                # parse cloth & parse cloth mask
        pcm = new_parse_map[3:4]
        pcm = torch.nn.functional.interpolate(pcm.unsqueeze(0), size=(self.fine_height, self.fine_width),
                                              mode='nearest').squeeze(0)

        #pcm = torch.nn.functional.interpolate(pcm.unsqueeze(0), size=(self.fine_height, self.fine_width), mode='nearest').squeeze(0)
        im = torch.nn.functional.interpolate(im.unsqueeze(0), size=(self.fine_height, self.fine_width),
                                             mode='bilinear').squeeze(0)

        #print(" coming from cp_dataset.py getitem in datasettest class")
        #print(f"im shape: {im.shape}")
        #print(f"pcm shape: {pcm.shape}")
        im_c = im * pcm + (1 - pcm)

        # load pose points
        pose_name = im_name.replace('.jpg', '_rendered.png')
        pose_map = Image.open(osp.join(self.data_path, 'openpose_img', pose_name))
        pose_map = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.BICUBIC)(pose_map)
        pose_map = self.transform(pose_map)  # [-1,1]

        # load densepose
        densepose_name = im_name.replace('image', 'image-densepose')
        densepose_map = Image.open(osp.join(self.data_path, 'image-densepose', densepose_name))
        densepose_map = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.BICUBIC)(
            densepose_map)
        #print(f"Resized densepose map size: {densepose_map.size}")

        densepose_map = self.transform(densepose_map)  # [-1,1]

        result = {
            'c_name': c_name,  # for visualization
            'im_name': im_name,  # for visualization or ground truth
            # intput 1 (clothfloww)
            'cloth': c,  # for input
            'cloth_mask': cm,  # for input
            # intput 2 (segnet)
            'parse_agnostic': new_parse_agnostic_map,
            'densepose': densepose_map,
            'pose': pose_map,  # for conditioning
            # GT
            'parse_onehot': parse_onehot,  # Cross Entropy
            'parse': new_parse_map,  # GAN Loss real
            'pcm': pcm,  # L1 Loss & vis
            'parse_cloth': im_c,  # VGG Loss & vis
            # visualization
            'image': im,  # for visualization
        }

        return result

    def __len__(self):
        return len(self.im_names)


class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch