from skimage import measure
import cv2
import os

path1 = "D:/pycpython/image/A/Rain100"
path2 = "D:/pycpython/image/A/Rain100HA21"
ssim_1 = 0
psnr_1 = 0

'''
#rain12
for i in range(12):
    if i< 9:
        target_file = "00%d_GT.png" % (i + 1)
        input_file = "00%d_in.png" % (i + 1)
    elif i < 99:
        target_file = "0%d_GT.png" % (i + 1)
        input_file = "0%d_in.png" % (i + 1)
    else:
        target_file = "%s_GT.png" % (i + 1)
        input_file = "%s_in.png" % (i + 1)
    norain = cv2.imread(os.path.join(path1, input_file))
    rain = cv2.imread(os.path.join(path2, target_file))
    psnr = measure.compare_psnr(norain, rain, 255)
    ssim = measure.compare_ssim(norain, rain, multichannel=True, data_range=255, win_size=11)
    ssim_1 = ssim_1 + ssim
    psnr_1 = psnr_1 + psnr
    print('number:%d,ssim=%.4f,psnr=%.4f' % (i + 1, ssim, psnr))
print("avessim=%.4f,avepsnr=%.4f" % (ssim_1 / 12, psnr_1 / 12))
'''
'''
#Rain1400 
for i in range(100):
    target_file = "%d.jpg" % (i + 901)
    for j in range(14):
        input_file = "%d_%d.jpg" % (i+901, j+1)
        norain = cv2.imread(os.path.join(path1, input_file))
        rain = cv2.imread(os.path.join(path2, target_file))
        psnr = measure.compare_psnr(norain, rain, 255)
        ssim = measure.compare_ssim(norain, rain, multichannel=True, data_range=255, win_size=11)
        ssim_1 = ssim_1 + ssim
        psnr_1 = psnr_1 + psnr
        print('number:%d-%d,ssim=%.4f,psnr=%.4f' % (i + 1,j+1, ssim, psnr))
print("avessim=%.4f,avepsnr=%.4f" % (ssim_1 / 1400, psnr_1 / 1400))
'''
#Rain100H and Rain100L
for i in range(100):
    if i < 9:
        norain_path = "norain-00%d.png" % (i + 1)
        rain_path = "rain-00%d.png" % (i + 1)
    elif i < 99:
        norain_path = "norain-0%d.png" % (i + 1)
        rain_path = "rain-0%d.png" % (i + 1)
    else:
        norain_path = "norain-%d.png" % (i + 1)
        rain_path = "rain-%d.png" % (i + 1)
    norain = cv2.imread(os.path.join(path1, norain_path))
    rain = cv2.imread(os.path.join(path2, rain_path))

    psnr = measure.compare_psnr(norain, rain, 255)
    ssim = measure.compare_ssim(norain, rain, multichannel=True, data_range=255, win_size=11)
    ssim_1 = ssim_1 + ssim
    psnr_1 = psnr_1 + psnr
    print('number:%d,ssim=%.4f,psnr=%.4f' % (i + 1, ssim, psnr))
print("avessim=%.4f,avepsnr=%.4f" % (ssim_1 / 100, psnr_1 / 100))
