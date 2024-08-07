import torch
from mtcnn import MTCNN
import cv2
import numpy as np


import PIL.Image as Image
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from torchvision import transforms as trans
import os
# import libnvjpeg
# import pickle
#todo 按照原始顺序存储

# img_root_dir = r'D:\BaiduNetdiskDownload\vggface2\vggface2_test\vggface2_test\test'
# save_path = r'D:\BaiduNetdiskDownload\vggface2\vggface2_test\vggface2_test\test1'
img_root_dir = '/media/HDD1/wangtao/lunwen5_code/VGGFace/A_VGGFace/'
save_path = '/media/HDD1/wangtao/lunwen5_code/VGGFace/A/'



# embed_path = '/home/taotao/Downloads/celeb-aligned-256/embed.pkl'

device = torch.device('cuda:0')
# device = torch.device('cpu')
mtcnn = MTCNN()

model = Backbone(50, 0.6, 'ir_se').to(device)
model.eval()
model.load_state_dict(torch.load('model_ir_se50.pth'))

# threshold = 1.54
test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# decoder = libnvjpeg.py_NVJpegDecoder()

embed_map = {}

for root, dirs, files in os.walk(img_root_dir):
    for name in files:
        if name.endswith('jpg') or name.endswith('png'):

            try:
                p = os.path.join(root, name)
                img = cv2.imread(p)
                # if img.shape[0]>256 and img.shape[1]
                faces = mtcnn.align_multi(Image.fromarray(img[:, :, ::-1]), min_face_size=64, crop_size=(128, 128))
                if len(faces) == 0:
                    continue
                for face in faces:
                    # scaled_img = face.resize((112, 112), Image.ANTIALIAS)
                    # with torch.no_grad():
                    #     embed = model(test_transform(scaled_img).unsqueeze(0).cuda()).squeeze().cpu().numpy()
                    new_path = name
                    print(new_path)
                    face.save(os.path.join(save_path, new_path))
                # embed_map[new_path] = embed.detach().cpu()
            except Exception as e:
                print(name+'----------------------------------------')
                continue

# with open(embed_path, 'wb') as f:
#     pickle.dump(embed_map, f)
#
# img = cv2.imread('/home/taotao/Pictures/47d947b4d9cf3e2f62c0c8023a1c0dea.jpg')[:,:,::-1]
# # bboxes, faces = mtcnn.align_multi(Image.fromarray(img), limit=10, min_face_size=30)
# bboxes, faces = mtcnn.align(Image.fromarray(img))
# input = test_transform(faces[0]).unsqueeze(0)
# embed = model(input.cuda())
# print(embed.shape)
# print(bboxes)
# face = np.array(faces[0])[:,:,::-1]
# cv2.imshow('', face)
# cv2.waitKey(0)
