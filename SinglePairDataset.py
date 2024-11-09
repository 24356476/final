import os
from PIL import Image
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage
import torch
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import transforms
import json
import numpy as np
class SinglePairDataset(Dataset):
    """
        Single Pair Dataset for handling one specific model and cloth image pair.
    """

    def __init__(self, model_image_path, cloth_image_path, opt):
        super(SinglePairDataset, self).__init__()
        self.opt = opt
        self.opt.data_root = ""
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc

        # File paths
        self.model_image_path = model_image_path
        self.cloth_image_path = cloth_image_path

        # Load pose data from JSON file
        self.pose_data = self.load_pose_data(model_image_path)

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.fine_height, self.fine_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Debug logging
        print(f"Initialized SinglePairDataset with model image: {model_image_path}, cloth image: {cloth_image_path}, and pose data provided.")




    def load_pose_data(self, model_image_path):
        # Use the base directory for the test dataset
        base_dir = os.path.join(self.opt.data_root, "data/test/openpose_json")
        image_name = os.path.basename(model_image_path).replace('.jpg', '_keypoints.json')
        json_path = os.path.join(base_dir, image_name)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Pose data not found at path: {json_path}")

        with open(json_path, 'r') as f:
            pose_data = json.load(f)['people'][0]['pose_keypoints_2d']
        return np.array(pose_data).reshape((-1, 3))[:, :2]
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

        # Calculate lengths for masking
        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        r_neck = int(length_a / 30) + 1
        r_torso = int(length_a / 50) + 1
        r_arms = int(length_a / 25) + 1

        # Mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r_torso * 3, pointy - r_torso * 6, pointx + r_torso * 3, pointy + r_torso * 6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r_torso * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r_torso * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r_torso * 12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # Mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx - r_neck * 5, pointy - r_neck * 9, pointx + r_neck * 5, pointy), 'gray', 'gray')

        # Mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r_arms * 12)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r_arms * 5, pointy - r_arms * 6, pointx + r_arms * 5, pointy + r_arms * 6),
                                  'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                    pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r_arms * 10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r_arms * 5, pointy - r_arms * 5, pointx + r_arms * 5, pointy + r_arms * 5),
                                  'gray', 'gray')

        # Add detailed arm masking
        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'white')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            pointx, pointy = pose_data[pose_ids[0]]
            mask_arm_draw.ellipse((pointx - r_arms * 5, pointy - r_arms * 6, pointx + r_arms * 5, pointy + r_arms * 6),
                                  'black', 'black')
            for i in pose_ids[1:]:
                if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                        pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r_arms * 10)
                pointx, pointy = pose_data[i]
                if i != pose_ids[-1]:
                    mask_arm_draw.ellipse(
                        (pointx - r_arms * 5, pointy - r_arms * 5, pointx + r_arms * 5, pointy + r_arms * 5), 'black',
                        'black')

            # Apply the resized mask arm
            parse_array_resized = np.array(transforms.Resize((self.fine_height, self.fine_width))(im_parse))
            parse_arm = (np.array(mask_arm) / 255) * (parse_array_resized == parse_id).astype(np.float32)

            agnostic_resized = agnostic.resize((self.fine_width, self.fine_height), Image.NEAREST)
            im_resized = im.resize((self.fine_width, self.fine_height), Image.NEAREST)
            parse_arm_resized = Image.fromarray(np.uint8(parse_arm * 255), 'L').resize(im_resized.size)

            agnostic_resized.paste(im_resized, None, parse_arm_resized)

        # Apply head and lower body masking
        parse_head_resized = Image.fromarray(np.uint8(parse_head * 255), 'L').resize(im.size, Image.NEAREST)
        parse_lower_resized = Image.fromarray(np.uint8(parse_lower * 255), 'L').resize(im.size, Image.NEAREST)
        agnostic_resized.paste(im, None, parse_head_resized)
        agnostic_resized.paste(im, None, parse_lower_resized)

        return agnostic

    def __getitem__(self, index):
        # Debugging inputs
        print(f"Processing index: {index}")
        print(f"Model image path: {self.model_image_path}")
        print(f"Cloth image path: {self.cloth_image_path}")

        # Model Image
        im_pil_big = Image.open(self.model_image_path).convert('RGB')
        im_pil = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.BICUBIC)(im_pil_big)
        im = self.transform(im_pil)

        # Cloth Image
        c = Image.open(self.cloth_image_path).convert('RGB')
        c = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.BICUBIC)(c)
        c = self.transform(c)

        # Cloth Mask
        cloth_mask_name = os.path.basename(self.cloth_image_path)
        cloth_mask_path = os.path.join(self.opt.data_root, "data/test/cloth-mask", cloth_mask_name)
        cm = Image.open(cloth_mask_path).convert('L')
        cm = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.NEAREST)(cm)
        cm_array = (np.array(cm) >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array).unsqueeze(0)

        # Parsing Image
        parse_name = os.path.basename(self.model_image_path).replace('.jpg', '.png')
        parse_path = os.path.join(self.opt.data_root, "data/test/image-parse-v3", parse_name)
        im_parse_pil_big = Image.open(parse_path).convert('L')
        im_parse_pil = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.NEAREST)(
            im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()

        # Labels and Parsing Maps
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
        # Create parsing maps
        parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse = torch.clamp(parse, max=19)  # Clamp values to the valid range [0, 19]
        print("Unique parse values after clamping:", torch.unique(parse))

        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        # Parse-Agnostic
        image_parse_agnostic_path = os.path.join(self.opt.data_root, "data/test/image-parse-agnostic-v3.2", parse_name)
        image_parse_agnostic = Image.open(image_parse_agnostic_path).convert('RGB')
        image_parse_agnostic = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.NEAREST)(
            image_parse_agnostic)
        parse_agnostic = self.transform(image_parse_agnostic)

        # DensePose
        densepose_path = os.path.join(self.opt.data_root, "data/test/image-densepose",
                                      os.path.basename(self.model_image_path))
        densepose_map = Image.open(densepose_path).convert('RGB')
        densepose_map = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.BICUBIC)(
            densepose_map)
        densepose_map = self.transform(densepose_map)

        # Pose Map and Keypoints
        pose_name = os.path.basename(self.model_image_path).replace('.jpg', '_rendered.png')
        pose_map_path = os.path.join(self.opt.data_root, "data/test/openpose_img", pose_name)
        pose_map = Image.open(pose_map_path).convert('RGB')
        pose_map = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.BICUBIC)(pose_map)
        pose_map = self.transform(pose_map)

        pose_keypoints_name = os.path.basename(self.model_image_path).replace('.jpg', '_keypoints.json')
        pose_keypoints_path = os.path.join(self.opt.data_root, "data/test/openpose_json", pose_keypoints_name)
        with open(pose_keypoints_path, 'r') as f:
            pose_data = np.array(json.load(f)['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:, :2]

        # Agnostic
        agnostic = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = transforms.Resize((self.fine_height, self.fine_width), interpolation=Image.NEAREST)(agnostic)
        agnostic = self.transform(agnostic)

        # Add logs to check input image loading
        print(f"Processing index: {index}")
        print(f"Model image path: {self.model_image_path}")
        print(f"Cloth image path: {self.cloth_image_path}")

        # Check intermediate outputs
        print("Generated agnostic:")
        print(f"Agnostic shape: {agnostic.shape if agnostic is not None else 'None'}")
        print("Other outputs:")
        print(f"Cloth shape: {c.shape if c is not None else 'None'}")




        # Final result dictionary
        result = {
            'cloth': c,
            'cloth_mask': cm,
            'parse_agnostic': new_parse_map,
            'densepose': densepose_map,
            'pose': torch.tensor(pose_data),
            'parse_onehot': parse_onehot,
            'parse': new_parse_map,
            'pcm': new_parse_map[3:4],
            'parse_cloth': im * new_parse_map[3:4] + (1 - new_parse_map[3:4]),
            'image': im,
            'agnostic': agnostic,
            'output_image': self.generate_output_image(im_pil, c)
        }

        # Debug final result
        print("Final result from SinglePairDataset:")
        for key, value in result.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    print(f"{key}[{subkey}]: {subvalue.shape if isinstance(subvalue, torch.Tensor) else subvalue}")
            else:
                print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

        return result

        #return result
        # At the end of __getitem__, after processing
        #output_path = os.path.join(self.opt.output_dir, f"output_{index}.png")
        #save_image(result['parse_cloth'], output_path)  # Save the generated image
        #result['output_path'] = output_path  # Add the path to result

    def name(self):
        return "SinglePairDataset"

    def __len__(self):
        # Only one pair, so the dataset length is 1
        return 1

    from torchvision.transforms import ToPILImage

    from torchvision.transforms import ToPILImage

    def generate_output_image(self, model_image, cloth_image_tensor):
        """
        Generate the virtual try-on output image by combining the model and cloth images.

        Args:
            model_image (PIL.Image.Image): The model image (already a PIL image).
            cloth_image_tensor (torch.Tensor): The cloth image (Tensor format).

        Returns:
            PIL.Image: The generated output image.
        """
        # Convert cloth tensor to PIL image
        to_pil = transforms.ToPILImage()
        cloth_image = to_pil(cloth_image_tensor)

        # Blend the two images (example logic; replace with your actual generation logic)
        combined_image = Image.blend(model_image, cloth_image, alpha=0.5)

        return combined_image



