# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):  # 自动对比度，img为PIL格式图片
    # 使用PIL库中的ImageOps.autocontrast函数来对输入的图像进行自动对比度增强，并将增强后的图像作为函数的返回值。
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    # 用于对输入的图像进行亮度增强，并返回增强后的图像。该函数使用PIL库中的ImageEnhance.Brightness类实现亮度增强，其增强程度由参数 v 控制。
    v = _float_parameter(v, max_v) + bias
    # v 表示亮度增强的程度，它是一个介于0到1之间的浮点数，实际的增强程度由 _float_parameter 函数生成；max_v 表示亮度增强程度的上限，它是一个浮点数，用于控制亮度增强的幅度，
    # 如果 v 大于 max_v，则取 max_v 作为增强程度；bias 表示偏置量，它是一个浮点数，用于调整亮度增强的基准值。
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    # 用于对输入的图像进行颜色增强，并返回增强后的图像。该函数使用PIL库中的ImageEnhance.Color类实现颜色增强，其增强程度由参数 v 控制。
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)
    # 参数v控制图像的色彩平衡，即色彩的鲜艳程度，范围为(0,正无穷)。factor 为 0 时生成灰度图像，为 1 时还是原始图像，该值越大，图像颜色越饱和。


def Contrast(img, v, max_v, bias=0):
    # 用于对输入的图像进行对比度增强，并返回增强后的图像。该函数使用PIL库中的ImageEnhance.Contrast类实现对比度增强，其增强程度由参数 v 控制。
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)
    


def Cutout(img, v, max_v, bias=0):
    # 用于对输入的图像进行随机擦除，并返回擦除后的图像。该函数会首先计算擦除区域的大小，然后在图像中随机选取一个位置，并将选中位置周围的像素点替换成固定的颜色值（例如黑色或者0值），从而实现擦除操作。
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    # 这段代码定义了一个名为 CutoutAbs 的函数，用于对输入的图像进行绝对位置的随机擦除，并返回擦除后的图像。该函数会首先随机选取一个擦除区域的左上角坐标（x0,y0），然后根据输入的擦除大小，
    # 计算出擦除区域的右下角坐标(x1,y1)，并将该区域内的像素值替换成固定的颜色值（例如黑色或者0值），从而实现擦除操作。
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    # 使用 PIL.ImageOps.equalize 函数对输入图像执行直方图均衡化操作。直方图均衡化是一种常用的图像增强方法，
    # 通过增加图像对比度来使得图像更加清晰。该函数会调整图像中每个像素的亮度值，使得输出图像中每个像素值的出现概率更加均衡，从而提高图像的质量。
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    # 将输入图像直接返回，即不做任何变换。这种函数通常用于在数据增强过程中，为了使得数据增强过程更加灵活，增加一些随机性。当随机变换的概率为零时，就使用 Identity 函数来保持原始图像不变。
    return img


def Invert(img, **kwarg):
    # 使用 PIL.ImageOps.invert 函数对输入图像进行反转（即图像中每个像素的值取反）。该变换通常用于增强图像的对比度和细节，使得原本暗淡的部分更加突出，从而更加清晰地呈现图像的特征。
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    # 使用 PIL.ImageOps.posterize 函数对输入图像进行色调分离操作，将图像中每个像素的色调分为 $2^v$ 个等级（其中 $v$ 是一个整数参数）。该变换可以使得图像变得更加简单，同时可以降低图像中的噪声和细节，从而使得图像更加清晰和易于处理。
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    # 使用 PIL 库中的 rotate 函数对输入图像进行旋转操作。该函数会将图像按照给定的角度 $v$ 进行顺时针或逆时针旋转（取决于一个 $50%$ 的概率），从而产生新的图像。旋转操作可以使得图像的内容重新排布，从而提供一种不同于原始图像的视角，也可以用于对图像进行增强和修复，从而使得图像更加鲜明和生动。
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    # 使用 PIL 库中的 ImageEnhance 模块中的 Sharpness 函数对输入图像进行锐化操作。该函数会增强图像的边缘和细节，使得图像更加清晰和锐利。具体来说，该函数会对图像进行一定的卷积操作，从而增强图像的高频分量。函数中的参数 $v$ 控制了锐化的程度，
    # 取值范围为 $[0, \text{max}_v]$，其中 $\text{max}_v$ 是参数的最大值。参数 $bias$ 是一个偏移量，用于调整参数 $v$ 的基准值。
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    # ShearX用于在x方向上对图像进行剪切，使图像在x方向上发生倾斜变化。该函数通过对输入图像进行仿射变换实现ShearX。其中，v是变换强度，max_v是变换强度的上限，bias是变换偏移。函数先从变换强度v和变换强度上限max_v中随机选择一个值，并加上变换偏移bias，
    # 得到最终的变换强度。然后，函数根据随机数决定变换方向（正方向或负方向），并利用PIL库提供的transform函数，将输入图像进行仿射变换，实现ShearX变换。
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    # 对图片进行反相处理（solarization），即将像素值小于阈值v的像素值取反，大于等于阈值v的像素值保持不变。阈值v在0到max_v之间随机选取，并加上一个偏差bias。
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    # 对输入的图像进行太阳化加操作。具体来说，该函数会将输入图像转换为NumPy数组，对其所有像素值加上一个随机数v，然后将数组中的值裁剪到0和255之间。
    # 接下来，将处理后的数组重新转换为图像，并对其进行Solarize操作，将所有像素值低于给定阈值threshold的像素值翻转。最后返回处理后的图像。
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    # 对图像进行水平方向的平移。详细描述：将输入的图像在水平方向上移动一个随机的距离。如果随机数小于0.5，则向左移动，否则向右移动。移动的距离为输入参数v与图像宽度（img.size[0]）的乘积。
    # 如果bias不为0，则移动的距离还要加上bias与图像宽度的乘积，返回一个经过平移后的图像。
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    # 将一个参数 v 从 [0, PARAMETER_MAX] 的整数范围映射到 [0, max_v] 的浮点数范围。其中 PARAMETER_MAX 是一个预先定义的常量，表示参数的最大值。
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    # 将0到PARAMETER_MAX之间的整数值v转换为0到max_v之间的整数值。具体来说，它首先将v除以PARAMETER_MAX，然后将结果乘以max_v并取整数。这个函数通常用于从超参数空间中采样一个值，并将其转换为在具体数据增强函数中使用的合适值。
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    # 对输入图像进行一系列随机增强
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


def my_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.2, 0),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0),
            (TranslateX, 0.45, 0),
            (TranslateY, 0.45, 0)]
    return augs

def augment_list():  
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l

class RandAugmentPC(object):  # 随机数据增强器，通过在给定的一组增强操作中随机选择一些来随机改变图像
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img

# RandAugmentPC通过随机选择一组增强算子，并根据指定的参数在图像上应用，同时使用CutoutAbs函数进行遮挡，使得图像增强后更具有鲁棒性。而RandAugmentMC使用了FixMatch论文中提出的增强算子组合，
# 并根据随机生成的参数在图像上应用，同样使用CutoutAbs函数进行遮挡。两者的主要区别在于增强算子的不同和参数生成方式的不同，因此也会对增强效果产生影响。

class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img

class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

        
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            img = op(img, val) 
        cutout_val = random.random() * 0.5 
        img = Cutout(img, cutout_val) #for fixmatch
        return img
