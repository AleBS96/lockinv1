import cv2
from collections import deque
import Function.Basics as Basics
import sys
import threading
import numpy as np
from cv2.typing import MatLike
import matplotlib.pyplot as plt
import time


class IRCamera (cv2.VideoCapture):
    
    __device_number             : int           
    __frame                     = deque()       #Queue of frames to process
    __frame_to_view             = MatLike
    __fourier                   = Basics.Fou()  #Camera lock in method
    __lock                      = threading.Lock()
    __fourier_flag              : bool
    __video                     : cv2.VideoCapture

    def __init__(self,device_num) -> None:
        self.__frame_to_view = np.empty(shape=(0,0))
        self.__device_number = device_num
        self.__fourier_flag = 0
        return super().__init__(self.__device_number)

    
    
    def set_video(self,path):
        self.__video = cv2.VideoCapture(path)

    def new_method(self):
        return cv2.Mat()
    
    @property
    def Frame(self) -> deque: 
        return self.__frame
    
    @property
    def Fourier(self) -> Basics.Fou: 
        return self.__fourier

    @Frame.getter
    def Frame(self) -> deque:
        self.__frame.append(self.read()) 
        return self.__frame.popleft()

    def Wait_Init_Frame(self):
        current_frame = 0
        print("Waiting for frame 3000")
        while current_frame < self.Fourier.InitFrame:
            self.Progress_Bar(current_frame/self.Fourier.InitFrame)
            ret, img = self.Frame
            self.__frame_to_view = img
            current_frame += 1


    #  Execute the Fourier Method
    def Run_Fourier(self):
        self.__fourier_flag = 0
        self.Crear_Hilo_Video()
        #Shows in console the Lock-in paramters that was set by user
        self.Print_Parameters()
        #Wait until the initial frame has been captured
        self.Wait_Init_Frame()
        #Starts the Lock-in
        print("\nLock-in started")
        while self.__fourier.Porcentage < 100:
            ret, img = self.Frame
            if ret:
                self.__frame_to_view = img
                self.__fourier.Thermogram = img
                self.__fourier_flag = 1
                self.Progress_Bar(self.__fourier.Porcentage/100)
            else:
                break
        
        #Save the amplitude and phase thermograms as two .png images
        Thermogram_Amplitude_N = Basics.imgNormalize(self.__fourier.Thermogram_Amplitude)
        Thermogram_Phase_N = Basics.imgNormalize(self.__fourier.Thermogram_Phase)
        cv2.imwrite("./Amplitude1.png", Thermogram_Amplitude_N)
        cv2.imwrite("./Phase1.png",Thermogram_Phase_N)


    def Print_Parameters(self):
        print("*********PARAMETERS*********")
        print("Initial Frame: " + str(self.__fourier.InitFrame))
        print("Final Frame: " + str(self.__fourier.FinalFrame))
        print("Modulation Frequency: " + str(self.__fourier.Modulation))
        print("Frame Rate: " + str(self.__fourier.FrameRate))
        print("****************************")


    def Progress_Bar(self,percentage):
        bar_length = 40
        filled_length = int(bar_length * percentage)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\r[{bar}] {percentage:.0%}')                 # Imprimir la barra de progreso en una sola línea
        sys.stdout.flush()

    def Crear_Hilo_Video(self):
        # Crear un hilo para ejecutar la función de mostrar el video
        hilo_video = threading.Thread(target=self.Mostrar_Video)
        hilo_video.start()

    def Mostrar_Video(self):
        while True:
        # Usa el lock para acceder de manera segura a la variable compartida
            with self.__lock:
                if self.__frame_to_view.size != 0:
                    if self.__fourier_flag == 1:
                        amp = Basics.imgNormalize(self.__fourier.Thermogram_Amplitude)                        
                        #amp = Gui.QImage(amp.data, amp.shape[1], amp.shape[0], Gui.QImage.Format_Grayscale8)
                        cv2.imshow('Amplitude', amp)
                        pha = Basics.imgNormalize(self.__fourier.Thermogram_Phase) 
                        #pha = Gui.QImage(pha.data, pha.shape[1], pha.shape[0], Gui.QImage.Format_Grayscale8)                    
                        cv2.imshow('Phase', pha)
                    
                    # Muestra el frame en una ventana llamada 'Video'
                    cv2.imshow('Video', self.__frame_to_view)

        # Si se presiona la tecla 'q', se cierra la ventana
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Cierra las ventanas
        cv2.destroyAllWindows()

if __name__ == "__main__":
  
    camera = IRCamera(0)
    camera.Fourier.InitFrame = 50
    camera.Fourier.FinalFrame = 3000
    camera.Fourier.Modulation = 60
    camera.Fourier.FrameRate = camera.get(cv2.CAP_PROP_FPS)
    if not camera.isOpened():
        print("Camara no encontrada")
        exit()
    else:
        camera.Run_Fourier()

  # Libera los recursos y cierra las ventanas
    camera.release()