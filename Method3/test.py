import os
import argparse
import torch
from torchvision import transforms, utils
from skimage import io, transform
from PIL import Image
import cv2
import numpy as np
from network import ReCoNet
from utilities import *

device = 'cuda'
video_capture = cv2.VideoCapture("../source.mp4")
model = ReCoNet().to(device)

output = []

model.load_state_dict(torch.load("./autoportrait/reconet_epoch_9.pth"))

img_arr = []

while(True):
    ret, frame = video_capture.read()

    if frame is None:
        break;

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(transform.resize(frame, (360, 640))).to(device).permute(2, 0, 1).float()
    features, styled_frame = model(frame.unsqueeze(0))
    styled_frame = transforms.ToPILImage()(styled_frame[0].detach().cpu())
    styled_frame = np.array(styled_frame)
    styled_frame = styled_frame[:, :,::-1]
    cv2.imshow('frame', styled_frame)
    img_arr.append(styled_frame)
    output.append(styled_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# video_capture.release()
# generate_video() 

size = (640,360)
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_arr)):
    out.write(img_arr[i])
out.release()