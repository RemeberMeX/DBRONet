import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from network import *
from DerainDataset import normalize
import time 

parser = argparse.ArgumentParser(description="network")
parser.add_argument("--logdir", type=str, default="logs/Rain100L", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="./input", help='path to training data')
parser.add_argument("--save_path", type=str, default="./output/Rain100L", help='path to save results')
parser.add_argument("--save_path1", type=str, default="./output/Rain100L-1", help='path to save results')
parser.add_argument("--save_path2", type=str, default="./output/Rain100L-2", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def main():
    os.makedirs(opt.save_path, exist_ok=True)
    os.makedirs(opt.save_path1, exist_ok=True)
    os.makedirs(opt.save_path2, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = Network(nin=64,use_GPU=opt.use_GPU)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth')))
    model.eval()

    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        img_path = os.path.join(opt.data_path, img_name)

        y = cv2.imread(img_path)
        b, g, r = cv2.split(y)
        y = cv2.merge([r, g, b])

        y = normalize(np.float32(y))
        y = np.expand_dims(y.transpose(2, 0, 1), 0)
        y = Variable(torch.Tensor(y))

        if opt.use_GPU:
            y = y.cuda()

        with torch.no_grad():
            if opt.use_GPU:
                torch.cuda.synchronize()
            start_time = time.time()

            out,r1,r2 = model(y)
            out = torch.clamp(out, 0., 1.)
            r1 = torch.clamp(r1, 0., 1.)
            r2 = torch.clamp(r2, 0., 1.)

            if opt.use_GPU:
                torch.cuda.synchronize()
            end_time = time.time()
            dur_time = end_time - start_time
            time_test += dur_time

            print(img_name, ': ', dur_time)

        if opt.use_GPU:
            save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
            r1_out=np.uint8(255 * r1.data.cpu().numpy().squeeze())
            r2_out=np.uint8(255 * r2.data.cpu().numpy().squeeze())
        else:
            save_out = np.uint8(255 * out.data.numpy().squeeze())
            r1_out=np.uint8(255 * r1.data.cpu().numpy().squeeze())
            r2_out=np.uint8(255 * r2.data.cpu().numpy().squeeze())

        save_out = save_out.transpose(1, 2, 0)
        b, g, r = cv2.split(save_out)
        save_out = cv2.merge([r, g, b])

        r1_out = r1_out.transpose(1, 2, 0)
        b, g, r = cv2.split(r1_out)
        r1_out = cv2.merge([r, g, b])

        r2_out = r2_out.transpose(1, 2, 0)
        b, g, r = cv2.split(r2_out)
        r2_out = cv2.merge([r, g, b])

        cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)
        cv2.imwrite(os.path.join(opt.save_path1, img_name), r1_out)
        cv2.imwrite(os.path.join(opt.save_path2, img_name), r2_out)

        count += 1

    print('Avg. time:', time_test/count)

if __name__ == "__main__":
    main()
