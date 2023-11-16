# import rasterio
import os
# from rasterio.enums import Resampling
import numpy as np
from PIL import Image
# temp = rasterio.open('ua/320img/30-2012-0800-6285-LA93-0M50-E080.jp2').read()
# with rasterio.open('ua/320img/30-2012-0800-6285-LA93-0M50-E080.tif', moverlape='w', driver='GTiff', height=temp.shape[1], width=temp.shape[2],
#                     count=temp.shape[0], dtype=temp.dtype) as dst:
#     dst.write(temp)

# alldatapath = [
#     r'ua/320img/train1',r'ua/320img/train2',r'ua/320img/test',r'ua/320img/val',
#     r'ua/320arf/train1',r'ua/320arf/train2',r'ua/320arf/test',r'ua/320arf/val',
#     r'ua/320slope/train1',r'ua/320slope/train2',r'ua/320slope/test',r'ua/320slope/val',
#     r'ua/320label/train1',r'ua/320label/train2',r'ua/320label/test',r'ua/320label/val',
#     ]

# allsavepath = [
#     r'ua_4k/320img/train1',r'ua_4k/320img/train2',r'ua_4k/320img/test',r'ua_4k/320img/val',
#     r'ua_4k/320arf/train1',r'ua_4k/320arf/train2',r'ua_4k/320arf/test',r'ua_4k/320arf/val',
#     r'ua_4k/320slope/train1',r'ua_4k/320slope/train2',r'ua_4k/320slope/test',r'ua_4k/320slope/val',
#     r'ua_4k/320label/train1',r'ua_4k/320label/train2',r'ua_4k/320label/test',r'ua_4k/320label/val',
    # ]

alldatapath = [
    r'ori/Image_RGB/train',r'ori/Image_RGB/test',r'ori/Image_RGB/val',
    # r'ori/Image__8bit_NirRGB/train',r'ori/Image__8bit_NirRGB/test',r'ori/Image__8bit_NirRGB/val',
    r'ori/Annotation__index/train',r'ori/Annotation__index/test',r'ori/Annotation__index/val',
    # r'ori/Annotation__color/train',r'ori/Annotation__color/test',r'ori/Annotation__color/val',
    ]

allsavepath = [
    r'fbp_3072/Image_RGB/train',r'fbp_3072/Image_RGB/test',r'fbp_3072/Image_RGB/val',
    # r'fbp_3072/Image__8bit_NirRGB/train',r'fbp_3072/Image__8bit_NirRGB/test',r'fbp_3072/Image__8bit_NirRGB/val',
    r'fbp_3072/Annotation__index/train',r'fbp_3072/Annotation__index/test',r'fbp_3072/Annotation__index/val',
    # r'fbp_3072/Annotation__color/train',r'fbp_3072/Annotation__color/test',r'fbp_3072/Annotation__color/val',
    ]

# alldatapath = [
    # r'ori/Image_RGB/train',r'ori/Image_RGB/test',r'ori/Image_RGB/val',
    # r'ori/Annotation__index/train',r'ori/Annotation__index/test',r'ori/Annotation__index/val',
    # r'ori/Annotation__color/train',r'ori/Annotation__color/test',r'ori/Annotation__color/val',
    # ]

# allsavepath = [
    # r'fbp_3072/Image_RGB/train',r'fbp_3072/Image_RGB/test',r'fbp_3072/Image_RGB/val',
    # r'fbp_3072/Annotation__index/train',r'fbp_3072/Annotation__index/test',r'fbp_3072/Annotation__index/val',
    # r'fbp_3072/Annotation__color/train',r'fbp_3072/Annotation__color/test',r'fbp_3072/Annotation__color/val',
    # ]

HH = 6800
WW = 7200
# sizeImgH = 3400
# sizeImgW = 3600
sizeImgH = 3072
sizeImgW = 3072
nnr = np.uint8(np.floor(HH/sizeImgH))
nnc = np.uint8(np.floor(WW/sizeImgW))

# overlapH = 0
# overlapW = 0
overlapH = np.uint8(np.ceil(sizeImgH - (HH-sizeImgH)/nnr))
overlapW = np.uint8(np.ceil(sizeImgW - (WW-sizeImgW)/nnc))
# overlapH = 1208
disH = sizeImgH - overlapH
# overlapW = 1008
disW = sizeImgW - overlapW

r = np.uint8(np.ceil(HH/disH))
c = np.uint8(np.ceil(WW/disW))

if (c-1)*disW+overlapW >= WW:
    c = c - 1
if (r-1)*disH+overlapH >= HH:
    r = r - 1

for nn in range(len(alldatapath)):
    datapath = alldatapath[nn]
    savepath = allsavepath[nn]
    datadir = os.listdir(datapath)
    for ii in range(len(datadir)):
        # if '1293546' not in datadir[ii]:
        #     continue
        path = os.path.join(datapath,datadir[ii])
        # with rasterio.open(path) as src:
        #     img = src.read()
        # img = np.transpose(np.array(Image.open(path)), (2,0,1))
        if 'Annotation__index' in savepath:
            img = np.expand_dims(np.array(Image.open(path)), 0)
        else:
            img = np.transpose(np.array(Image.open(path)), (2,0,1))
        [C,H,W] = img.shape

        for i in range(r):
            if i%10 == 0:
                print([ii, i*100/r])
            for j in range(c):
                temp = np.zeros([C,sizeImgH,sizeImgW],dtype=np.uint8)
                for cc in range(C):
                    if i==r-1:
                        if j==c-1:
                            temp[cc,:,:] = img[cc, H-sizeImgH:H, W-sizeImgW:W] 
                        else:
                            temp[cc,:,:] = img[cc, H-sizeImgH:H, j*disW:(j+1)*disW+overlapW] 
                    else:
                        if j==c-1:
                            temp[cc,:,:] = img[cc, i*disH:(i+1)*disH+overlapH, W-sizeImgW:W] 
                        else:
                            temp[cc,:,:] = img[cc, i*disH:(i+1)*disH+overlapH, j*disW:(j+1)*disW+overlapW]

                if 'Annotation__index' in savepath:
                    newpath = os.path.join(savepath,datadir[ii][:-12]+'_'+str(i+1)+'_'+str(j+1)+'_24label.png')
                    temp = Image.fromarray(temp[0])
                elif 'Annotation__color' in savepath:
                    newpath = os.path.join(savepath,datadir[ii][:-12]+'_'+str(i+1)+'_'+str(j+1)+'_24label.png')
                    temp = Image.fromarray(np.transpose(temp, (1,2,0)))
                else:
                    newpath = os.path.join(savepath,datadir[ii][:-4]+'_'+str(i+1)+'_'+str(j+1)+'.tif')
                    temp = Image.fromarray(np.transpose(temp, (1,2,0)))
                # temp = Image.fromarray(np.transpose(temp, (1,2,0)))
                os.makedirs(os.path.dirname(newpath), exist_ok=True)
                temp.save(newpath)
                # with rasterio.open(newpath, moverlape='w', driver='GTiff', height=temp.shape[1], width=temp.shape[2],
                #                     count=temp.shape[0], dtype=temp.dtype) as dst:
                #     dst.write(temp)
                
