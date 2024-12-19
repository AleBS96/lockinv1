from Function.Methods import Fourier as Fou
import pyqtgraph as pg
import functools, math, cv2
import shutil, os ,  time, numpy as np

from PySide6 import QtWidgets as Wigdets
from PySide6 import QtGui as Gui
from PySide6 import QtMultimedia as Media
from PySide6 import QtCore as Core
from pathlib import Path
from cv2.typing import MatLike
import matplotlib.pyplot as plt


__absFile = lambda p :  str(Path(p).name)   # Returns the file name of the file passed
getTime = lambda f, fr : divmod(f / fr, 60)

def setProgress(self):
    if self.ui.progressBar.value() == 100: self.ui.progressBar.setValue(0)

def media_position(self):
    self.ui.videoSlider.setValue(self.mediaPlayer.position())

def move_video(self):
    self.mediaPlayer.setPosition(self.ui.videoSlider.value())

def forwardvideo(self):
    self.mediaPlayer.setPosition(self.mediaPlayer.duration())

def backvideo(self):
    self.mediaPlayer.setPosition(0)

def stopvideo(self):
    if self.mediaPlayer.playbackState() == Media.QMediaPlayer.PlayingState or self.mediaPlayer.playbackState()  == Media.QMediaPlayer.PausedState:
        self.mediaPlayer.stop()

def playvideo(self):
    if self.mediaPlayer.playbackState() == Media.QMediaPlayer.PlayingState:
        self.mediaPlayer.pause()
    else:
        self.mediaPlayer.play()
    self.ui.statusbar.showMessage('Playing Video......')

def calculateFrames(Final: Wigdets.QSpinBox, Init: Wigdets.QSpinBox, Frames: Wigdets.QSpinBox):
    t = Final.value() - Init.value()
    Frames.setValue(t)
    return t

"""
    This function check at the temp folder exist and its VAcio
    if the folder doesn`t exist its created.
"""
def checkTempFolder(self):
    self.__TempPath = Path(self.__absPath(self.__FilePath)) / 'temp'
    shutil.rmtree(self.__TempPath, ignore_errors=True)
    os.makedirs(self.__TempPath)
           
"""
    This function presents a dialog windows to open the video file to be process
"""
def openFile(self):
    self.__FilePath , _ = Wigdets.QFileDialog.getOpenFileName(self,"Abrir archivo", str("Videos (*.avi *.mp4)"))
    self.ui.statusbar.showMessage('Openning the Video......')
    self.mediaPlayer.setSource(Core.QUrl(self.__FilePath))                          # set the file like source of video
    self.ui.videoSlider.setMaximum(self.mediaPlayer.duration())     # Set the Video player slider to maximum value of video's duration                              # Enable the video player controls
    checkTempFolder(self)

    self.video = cv2.VideoCapture(self.__FilePath)
    t = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    self.ui.initFrameSpinBox.setMaximum(t)
    self.ui.finalFrameSpinBox.setMaximum(t)
    self.ui.finalFrameSpinBox.setValue(t)
    self.ui.framesSpinBox.setMaximum(t)
    self.ui.framesByLockInPeriod.setMaximum(t)
    t = int(self.video.get(cv2.CAP_PROP_FPS))
    self.ui.frameRate.setValue(t)
    self.ui.framesHeight.setValue(int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    self.ui.frameWidth.setValue(int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)))

    # Se calcula el tiempo de duracion del video
    __MinutesGet, __SecondsGet = getTime(self.ui.finalFrameSpinBox.value(), self.ui.frameRate.value())
    self.ui.duration.setText(Core.QCoreApplication.translate("MainWindow", str(f'{round(__MinutesGet)}:{round(__SecondsGet)}'), None))

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

def imgUmbralize(img: MatLike, min: int, max: int ) -> MatLike:
    # normalizamos la imagen
    img = imgNormalize(img)
    # Aplicar suavizado Gaussiano
    img = cv2.GaussianBlur(img, (5, 5), 0)
    a, img = cv2.threshold(img, min, max,cv2.THRESH_BINARY) # + cv2.THRESH_OTSU
    return img

def imgContours(img:MatLike, kernel)->MatLike:
    # Aplicamos transformacion morfologica de closing
    morph = cv2.morphologyEx(img , cv2.MORPH_GRADIENT , kernel)
    # Encontrar los bordes
    img = cv2.Canny(morph,150,180)
    # Encontrar los contornos
    contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        # Calcular el centroide del contorno para colocar el texto
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Calcular el área del contorno
            area = cv2.contourArea(contour)
            
            # Calcular el perímetro del contorno
            perimeter = cv2.arcLength(contour, True)

            cv2.drawContours(morph, contours[0], -1, (0, 255, 0), 1)
            cv2.putText(morph, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)


    
    return morph

    """
    # Dibujar contornos y numerarlos
    
        

            

            
            # Usar la relación entre el área y el cuadrado del perímetro para identificar círculos
            # Esta es una simplificación de la fórmula para el índice de redondez de un objeto
            # Los objetos que son más cercanos a ser círculos tendrán valores más cercanos a 12.57 (4*pi)
            roundness = 4 * np.pi * area / (perimeter * perimeter)
            
            if roundness >= 1:
                
                

    # Dibujar los contornos circulares en la imagen original
    #return cv2.drawContours(closing, contours, 0, (255, 255, 255), 1)
    """

def ExportImgAmplitudeFourier(img: Fou):
    # Mostrar el cuadro de diálogo para seleccionar el directorio
    options = Wigdets.QFileDialog.Options()
    options |= Wigdets.QFileDialog.DontUseNativeDialog
    file_name, _ = Wigdets.QFileDialog.getSaveFileName(None,"Guardar imagen", "","Imagenes (*.png *.xpm *.jpg)", options=options)

    if file_name:
        # Agregar la extensión .png si no está presente
        if not file_name.endswith('.png'):
            file_name += '.png'
            # Guardar la imagen en el directorio seleccionado
        cv2.imwrite(file_name, img.Thermogram_Amplitude)

def ExportDataAmplitudeFourier(img:Fou):
    # Mostrar el cuadro de diálogo para seleccionar el directorio
    options = Wigdets.QFileDialog.Options()
    options |= Wigdets.QFileDialog.DontUseNativeDialog
    file_name, _ = Wigdets.QFileDialog.getSaveFileName(None,"Guardar matriz", "","CSV (*.csv)", options=options)

    if file_name:
        # Agregar la extensión .csv si no está presente
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        # Guardar la matriz en el archivo seleccionado
        np.savetxt(file_name, img.Thermogram_Amplitude, delimiter=",")

def ExportImgPhaseFourier(img: Fou):
    # Mostrar el cuadro de diálogo para seleccionar el directorio
    options = Wigdets.QFileDialog.Options()
    options |= Wigdets.QFileDialog.DontUseNativeDialog
    file_name, _ = Wigdets.QFileDialog.getSaveFileName(None,"Guardar imagen", "","Imagenes (*.png *.xpm *.jpg)", options=options)

    if file_name:
        # Agregar la extensión .png si no está presente
        if not file_name.endswith('.png'):
            file_name += '.png'
            # Guardar la imagen en el directorio seleccionado
        cv2.imwrite(file_name, imgNormalize(img.Thermogram_Phase))

def ExportDataPhaseFourier(img: Fou):
        # Mostrar el cuadro de diálogo para seleccionar el directorio
    options = Wigdets.QFileDialog.Options()
    options |= Wigdets.QFileDialog.DontUseNativeDialog
    file_name, _ = Wigdets.QFileDialog.getSaveFileName(None,"Guardar matriz", "","CSV (*.csv)", options=options)

    if file_name:
        # Agregar la extensión .csv si no está presente
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        # Guardar la matriz en el archivo seleccionado
        np.savetxt(file_name, img.Thermogram_Phase, delimiter=",")
    
"""
def referenceX(self):
    # Generamos el vector de puntos de la señal de referencia
    return np.linspace(0, self.fourier_method.Frames, self.fourier_method.Frames)

def referenceSen(self):
    return np.sin(2 * np.pi * self.fourier_method.Modulation * referenceX())

def referenceCos(self):
    return np.cos(2 * np.pi * self.fourier_method.Modulation * referenceX())

def initPlotReference(self: pg.PlotWidget):
    # Se configuran el color de fondo y parametros bàsicos de las graficas de las señales de referencia
    self.setBackground("w")
    self.addLegend()
    self.setTitle('Reference Signals', size = '20px')
    self.setLabel('left','Amplitude')
    self.setLabel('bottom','Frames',size = '12px')

"""