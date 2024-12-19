import cv2
import  numpy as np
from cv2.typing import MatLike



"""_summary_
    This Function obtains the RGB and grayscale Channels of a thermal image, then converts the rgb portion to grayscale
    and finally merges both grayscale images into weighted image one.
"""    
def imgDivide(img: MatLike) -> MatLike:
    height  = int(img.shape[0] / 2)            # gets the size of image
    width   = int(img.shape[1])
    
    imgA = img[:height, :] 
    imgB = img[height:,:]
    
    # Convierte los datos a doble precisión y los divide por 255 ya que la imagen se encuentra en escala de grises
    imgA = (imgA.astype(np.float64)) / 255
    imgB = (imgB.astype(np.float64)) / 255
    #img = imgA*imgB
    img = imgB
    
    img = np.asarray(img)
    
    # Inviertes el orden de las filas
    #img = np.flip(img, 0)

    # Luego inviertes el orden de las columnas
    #img = np.flip(img, 1)
    # img     = cv2.addWeighted(imgA,0.5,imgB,0.5,0) # both images are fusioned  

    # clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(3,3)) # creates the kernel to filter apply
    # img = clahe.apply(img)
    return img

def imgPrepare(img : MatLike) -> MatLike:
    __tmp = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # converts the image to gray scale
    __tmp = (__tmp.astype(np.float64)) / 255
    return np.asarray(__tmp)

def imgNormalize(img : MatLike) -> MatLike:
    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8) 
