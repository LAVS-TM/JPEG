# %%
import numpy as np
import cv2

from numpy.core.numeric import binary_repr
import skimage as ski
from skimage import data
from skimage import color
from skimage import io
import skimage
from skimage.color.colorconv import rgb2gray
from skimage.util import *
from skimage.transform import resize
from scipy.fft import dct
from scipy.fft import idct
from tables import *
from math import sqrt
from matplotlib import pyplot as plt
from PIL import Image

test_dct = [[52, 55, 61, 66, 70, 61, 64, 73],
            [63, 59, 55, 90, 109, 85, 69, 72],
            [62, 59, 68, 113, 144, 104, 66, 73],
            [63, 58, 71, 122, 154, 106, 70, 69],
            [67, 61, 68, 104, 126, 88, 68, 70],
            [79, 65, 60, 70, 77, 68, 58, 75],
            [85, 71, 64, 59, 55, 61, 65, 83],
            [87, 79, 69, 68, 65, 76, 78, 94]]

quantization_matrix = [[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 58, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]]

chrominance_quant_matrix = [[17, 18, 24, 47, 99, 99, 99, 99],
                            [18, 21, 26, 66, 99, 99, 99, 99],
                            [24, 26, 56, 99, 99, 99, 99, 99],
                            [47, 66, 99, 99, 99, 99, 99, 99],
                            [99, 99, 99, 99, 99, 99, 99, 99],
                            [99, 99, 99, 99, 99, 99, 99, 99],
                            [99, 99, 99, 99, 99, 99, 99, 99],
                            [99, 99, 99, 99, 99, 99, 99, 99]]

test_reverse_dct = [[-416, -33, -60, 32, 48, -40, 0, 0],
                    [0, -24, -56, 19, 26, 0, 0, 0],
                    [-42, 13, 80, -24, -40, 0, 0, 0],
                    [-42, 17, 44, -29, 0, 0, 0, 0],
                    [18, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]

__zag = np.array([
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 41, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
])


test_16zero_zigzag = [1, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def __resize(image):

    width = len(image[0])
    height = len(image)

    if (width % 8 == 0 and height % 8 == 0):
        return image

    new_width = width if width % 8 == 0 else 8 - width % 8 + width
    new_height = height if height % 8 == 0 else 8 - height % 8 + height

    resized_image = np.pad(image, pad_width=(
        (0, new_height - height), (0, new_width - width)), mode='reflect')

    return resized_image


# pb d'arrondis -> on a utilisé scikit plutôt que matlab donc arrondi différent
def __dct(block):
    #block = block.astype(uint)
    block = np.subtract(block, 128)
    block = np.float32(block)
    dct_block = cv2.dct(block)
    #dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    for i in range(len(dct_block)):
        for j in range(len(dct_block[i])):
            v = dct_block[i][j] 
            if abs(v) == 0.5:
                v *= 2
            elif abs(v) % 1 == 0.5:
                if v < 0:
                    v += 0.5
                else:
                    v -= 0.5
            else:
                v = round(v)
            dct_block[i][j] = v
    return dct_block.astype(int)


def quantization(dct_block, realQuant):  # wrong rounding
    for i in range(len(dct_block)):
        for j in range(len(dct_block[i])):
            v = dct_block[i][j] / realQuant[i][j]
            if abs(v) == 0.5:
                v *= 2
            elif abs(v) % 1 == 0.5:
                if v < 0:
                    v += 0.5
                else:
                    v -= 0.5            
            else:
                v = round(v)
            dct_block[i][j] = v
    return dct_block


def zigzag(quantized):
    zigzaged = np.concatenate([np.diagonal(quantized[::-1, :], i)[::(2*(i % 2)-1)]
                              for i in range(1-quantized.shape[0], quantized.shape[0])])
    zigzaged = np.trim_zeros(zigzaged, trim='b')
    if len(zigzaged) == 0:
        zigzaged = np.append(zigzaged, 0)
    return zigzaged


def huffman(zigzaged, lastDC):

    DC = zigzaged[0] - lastDC
    AC = zigzaged[1:]

    huffmaned = []
    for i in range(len(RangeTable)):
        if RangeTable[i][0] <= abs(DC) <= RangeTable[i][1]:
            if DC > 0:
                pos = DC
            else:
                pos = DC + RangeTable[i][1]
            prefix = DCTable[i][0]
            length = DCTable[i][1]
            if length != len(prefix):
                huffmaned.append(prefix + format(pos, "0" +
                                 str(length - len(prefix)) + "b"))
            else:
                huffmaned.append(prefix)
            break

    i = 0
    while (i != len(AC)):
        cpt = 0
        while AC[i] == 0 and cpt != 15:
            cpt += 1
            i += 1
        for j in range(len(RangeTable)):
            if RangeTable[j][0] <= abs(AC[i]) <= RangeTable[j][1]:
                category = j
                if AC[i] > 0:
                    pos = AC[i]
                else:
                    pos = AC[i] + RangeTable[j][1]
                prefix = ACTable[cpt][category][0]
                length = ACTable[cpt][category][1]
                if length != len(prefix):
                    huffmaned.append(
                        prefix + format(pos, "0" + str(length - len(prefix)) + "b"))
                else:
                    huffmaned.append(prefix)
                break
        i += 1

    # EOB
    huffmaned.append(ACTable[0][0][0])
    return huffmaned


def reverse_huffman(huffmaned):

    DC = huffmaned[0]
    AC = huffmaned[1:]

    zigzag = []

    if 0 <= int(DC, 2) <= 7:
        if len(DC) == 5:
            category = 3
        elif len(DC) == 4:
            category = 1
        else:
            category = 0

        e = DCTable[category]
        length = e[1] - len(e[0])
        min = e[0]
        max = e[0]
        for _ in range(length):
            min += "0"
            max += "1"
        ranges = RangeTable[category]
        size = ranges[1] - ranges[0]
        pos = int(DC, 2) - int(min, 2)
        if pos > size:
            value = ranges[0] + pos - size - 1
        else:
            value = -ranges[1] + pos
        zigzag.append(value)
    else:
        for i, e in enumerate(DCTable):
            if i in [0, 1, 3]:
                continue
            length = e[1] - len(e[0])
            min = e[0]
            max = e[0]
            for _ in range(length):
                min += "0"
                max += "1"

            if min <= DC <= max:
                category = i
                ranges = RangeTable[category]
                size = ranges[1] - ranges[0]
                pos = int(DC, 2) - int(min, 2)
                if pos > size:
                    value = ranges[0] + pos - size - 1
                else:
                    value = -ranges[1] + pos
                zigzag.append(value)
                break

    i = 0
    for e in AC[:-1]:
        for i in range(len(ACTable)):
            for j in range(len(ACTable[i])):
                length = ACTable[i][j][1] - len(ACTable[i][j][0])
                min, max = ACTable[i][j][0], ACTable[i][j][0]
                for _ in range(length):
                    min += "0"
                    max += "1"
                if min <= e <= max:
                    run = i
                    category = j
                    for _ in range(run):
                        zigzag.append(0)
                    ranges = RangeTable[category]
                    size = ranges[1] - ranges[0]
                    pos = int(e, 2) - int(min, 2)
                    if pos > size:
                        value = ranges[0] + pos - size - 1
                    else:
                        value = -ranges[1] + pos
                    zigzag.append(value)
                    break
    return zigzag


def reverse_zigzag(zigzagPath):
    while(len(zigzagPath) != 64):
        zigzagPath.append(0)

    zigzag = np.array(zigzagPath).reshape((len(zigzagPath)//64, 64))
    quantized = np.zeros(zigzag.shape)
    for i in range(len(__zag)):
        quantized[:, i] = zigzag[:, __zag[i]]
    quantized = quantized.reshape(8, 8)
    return quantized


def reverse_quantization(quantized, realQuant):
    dct = np.multiply(quantized, realQuant)
    return dct


def reverse_dct(dct_block):
    dct_block = idct(idct(dct_block, axis=0, norm='ortho'),
               axis=1, norm='ortho')
    for i in range(len(dct_block)):
        for j in range(len(dct_block[i])):
            v = dct_block[i][j] 
            if abs(v) == 0.5:
                v *= 2
            elif abs(v) % 1 == 0.5:
                if v < 0:
                    v += 0.5
                else:
                    v -= 0.5
            else:
                v = round(v)
            dct_block[i][j] = v
    dct_block = dct_block.astype(np.uint8)
    block = np.add(dct_block, 128)
    return block


def compress(image, macro_blocks, filePath, realQuant):
    with open(filePath, 'w') as file:
        file.write(str(len(image)) + " " + str(len(image[0])) + '\n')
        for i in range(len(macro_blocks)):
            lastDC = 0
            for j in range(len(macro_blocks[i])):
                dcted = __dct(macro_blocks[i][j])
                quantized = quantization(dcted, realQuant)
                zigzaged = zigzag(quantized)
                huffmaned = huffman(zigzaged, lastDC)
                lastDC = zigzaged[0]
                for k in range(len(huffmaned) - 1):
                    file.write(huffmaned[k] + " ")
                file.write(huffmaned[-1] + "\n")
        file.close()


def decompress(filePath, realQuant):
    with open(filePath, 'r') as file:
        lines = file.read().splitlines()
        first_line = lines[0].split(" ")

        width, height = int(first_line[0]), int(first_line[1])

        nb_blocks_line = (width if width %
                          8 == 0 else 8 - width % 8 + width) // 8
        nb_blocks_col = (height if height %
                         8 == 0 else 8 - height % 8 + height) // 8

        new_macro_blocks = np.ndarray(
            (nb_blocks_line, nb_blocks_col, 8, 8), dtype=int)

        count_width = 0
        count_height = 0
        line_of_block = np.ndarray((nb_blocks_col, 8, 8), dtype=int)

        lastDC = 0
        for i in range(1, len(lines)):
            huffmaned = lines[i].split(" ")
            r_zigzaged = reverse_huffman(huffmaned)
            r_zigzaged[0] += lastDC
            lastDC = r_zigzaged[0]
            r_quantized = reverse_zigzag(r_zigzaged)
            r_dct = reverse_quantization(r_quantized, realQuant)
            r_block = reverse_dct(r_dct)

            line_of_block[count_width] = r_block
            count_width += 1
            if count_width == nb_blocks_col:  # line or col?
                new_macro_blocks[count_height] = line_of_block
                count_height += 1
                count_width = 0
                lastDC = 0

        new_macro_blocks[count_height - 1] = line_of_block

    result = new_macro_blocks.swapaxes(2, 1).reshape(
        new_macro_blocks.shape[0] * new_macro_blocks.shape[2], -1)
    result = result[0:width, 0:height]
    return result


def jpeg(image, realQuant):
    filePath = "compressed"

    if len(image.shape) == 2:
        print("Gray image")
    
    # RGB



    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    #print("Red:\n", red)
    print("Blue:\n", blue)
    # Gentil nico

    # convertir blue, green et red en graaaaaay ?
    
    # blue
    img_resized = __resize(blue)
    macro_blocks = view_as_blocks(img_resized, block_shape=(8, 8))

    compress(blue, macro_blocks, filePath, realQuant)
    resultBlue = decompress(filePath, realQuant)

    # green
    img_resized = __resize(green)
    macro_blocks = view_as_blocks(img_resized, block_shape=(8, 8))

    compress(green, macro_blocks, filePath, realQuant)
    resultGreen = decompress(filePath, realQuant)
    
    # red    
    img_resized = __resize(red)
    macro_blocks = view_as_blocks(img_resized, block_shape=(8, 8))

    compress(red, macro_blocks, filePath, realQuant)
    resultRed = decompress(filePath, realQuant)

    #resultRed = rgb2gray(resultRed)
    #return red, resultRed

    result = np.zeros((len(image), len(image[0]), 3), dtype='uint8')
    result[..., 0] = resultRed
    result[..., 1] = resultGreen
    result[..., 2] = resultBlue
    
    # YUV
    # image = rgb2yuv(image)
    # ... do compr/decompr...
    # result = yuv2rgv(result)
    #
    # /!\ q pour chrominance, mais ne change pas la chrominance_quant_matrix! (2:41:45)
    # sous-échantillonnage 4:4:4 --> compr/décompr de base pour les 3 canaux
    # 4:2:2 --> avant de compresser, on enlève une colonne sur 2 (pour les 2 chrominances)
    #               et après décompression, on fait juste le plus proche voisin
    # 4:2:0 --> 

    fig, axes = plt.subplots(2, 3, figsize=(8, 4))
    ax = axes.ravel()
    ax[0].imshow(red)
    ax[1].imshow(green)
    ax[2].imshow(blue)
    ax[3].imshow(resultRed)
    ax[4].imshow(resultGreen)
    ax[5].imshow(resultBlue)

    fig.tight_layout()
    plt.show()

    return result


def main(path):
    #image = np.arange(100).reshape(10, 10)
    #image = np.arange(16*32).reshape(16,32)
    image = io.imread(path)
    print(image.shape)

    # RGB

    #image = rgb2gray(image)
    #image = image.astype(int)
    # print(image)
    # print("Image originale:", image)
    #image = np.asarray(test_dct)

    #img_resized = __resize(image)

    #macro_blocks = view_as_blocks(img_resized, block_shape=(8, 8))
    #filePath = "compressed"

    # to change compression quality
    q = 50  # à changer nous même, entre 0 et 100!
    if q < 50:
        alpha = 5000 / q
    else:
        alpha = 200 - 2*q
    realQuant = np.divide(np.add(np.multiply(quantization_matrix, alpha), 50), 100)
    realQuant = realQuant.astype(int)

    #compress(image, macro_blocks, filePath, realQuant)
    #result = decompress(filePath, realQuant)
    result = jpeg(image, realQuant)
    print("Result:", result)

    #sub = np.subtract(image, result)



    '''
    #plt.imshow(result, interpolation='nearest', cmap=plt.cm.gray)
    plt.show()
    '''

    # Decompression c bon je craque ????

    # matR = recup matrice RED
    # matG = recup matrice GREEN
    # matB = recup matrice BLUE

    # jpeg(matR)
    # jpeg(matG)
    # jpeg(matB)


main("samples/randompic_color.png")
# %%
