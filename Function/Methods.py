import app.Function.Basics as Basics
import numpy as np, cmath
from cv2.typing import MatLike

class Fourier:

    __Frame_Rate        = int                   # Represents the frame Rate of the video that will be used by lock-in method    (fs)
    __Frames            = int                   # Represents the value of the frames will be process  
    __init_frame        = int                   # Represents the number of the initial's frame of the video that will be process
    __final_frame       = int                   # Represents the number of the final's frame of the video that will be process
    __modulation        = float                 # Represents the modulation's frequency of the reference signal                 (fe)
    __K                 = float                 # Represents the K's factor, defined by this Method like                        [N(fe/fs)+1]
    __FramesByPeriod    = int                   # Represents the N's factor, it is knowed like frames by lock-in periods        (N)
    __currentFrame      = int                   # Represents the current frame in the range of frames to be process
    __porcentage        = int                   # Represents the progress's porcentage of the process compute   
    __last_image_Frame  = []                    # Represents the last image loader by openCV
    __Image_full        = bool                  # Indicates if the thermal image has 2 images or just has a full size one
    __Reset             = bool                  # Indicates that the process should be reset on the next video file read operation  
    __W                 = complex               # Represents the W's Factor used in this Method
    __WNValue           = complex               # Represents the W's Factor when variables K and n are used in every frame processed (current WN value)
    __listener          = []                    # Se inicializa el controlador de receptores de eventos

    __Statistic         = {'Min':float,'Max':float, 'Mean':float}
    __currentStatistic  = {'Global':__Statistic, 'X':__Statistic, 'Y':__Statistic}


    __curr_image_frame  = np.array               # Represents the current image loaded and processed

    
    Wn          = lambda _, W,  n , K: pow( W, (n - 1) * (K - 1))
    KFactor     = lambda _, N,fe,fs: N*( fe / fs )+1
    FbP         = lambda _, modulation , framerate : (1/modulation) / (1/framerate)     # Frames by Lock-in Period
    WFactor     = lambda _, N : cmath.exp(-1j*2*cmath.pi/N)
    cp          = lambda _, F , N : int(F / N) + 1                                      # Current lock-in period value
    rangeFrame  = lambda _, f, i : f - i                                                # Number of total frames to be process
    vop         = lambda _, frames, current:((current * 100)) / frames              # Value  of  progress
   

    def register(self, listener):               # Mètodo que registra a los receptores de eventos de esta clase
        self.__listener.append(listener)
    
    def unregister(self, listener):             # Mètodo que elimina el registro de los receptores de eventos
        self.__listener.remove(listener)

    @property
    def DigitalPeriods(self)->int:
        return  self.Frames / self.N

    # Properties of the Current frame in process
    @property                                   # Gets the number of frame that is processing
    def CurrentFrame(self)->int:
        return self.__currentFrame

    @CurrentFrame.setter                        # Sets the number of frame that is processing
    def CurrentFrame(self,value:int):
        self.__currentFrame = value
        self.__porcentage   = self.vop(self.Frames,value)
        for watcher in self.__listener:
            watcher.Porcentage_Changed(self.Porcentage)
        
    @property                                   # Gets the value of porcentage of processing progress 
    def Porcentage(self)-> int:
        return self.__porcentage
    
    @Porcentage.getter
    def Porcentage(self)-> int:
        return self.__porcentage 

    # Frame Rate properties
    @property
    def FrameRate(self)->int:                   # Gets the frame Rate value
        return self.__Frame_Rate
    
    @FrameRate.setter
    def FrameRate(self, value:int):             # Sets the Frame Rate Value,  you must update the same values in Modulation property
        self.__Frame_Rate       = value
        self.__FramesByPeriod   = self.FbP(self.Modulation,self.FrameRate)
        self.__K                = self.KFactor(self.N, self.Modulation, self.FrameRate)
        self.__W                = self.WFactor(self.N)

    # Total Frames to be Processed propeties
    @property
    def Frames(self)->int:                      # Gets the value (range) of total frames to be processed by this Method
        return self.__Frames

    @Frames.setter                              # Sets the value (range) of total frames to be proessed by this Method
    def Frames(self,value):
        self.__Frames = value

    # Properties related with the Init's frame of the video that will be process
    @property                                   # Gets the number of initial's frame of the video to be processed
    def InitFrame(self)-> int:
        return self.__init_frame
  
    @InitFrame.setter                           # Sets the number of initial's frame of the video to be processed
    def InitFrame(self, value:int):
        self.__init_frame = value
        self.__Frames = self.rangeFrame(self.FinalFrame,self.InitFrame)

    # Properties related with the final's frame of the video that will be processed
    @property                                   # Gets the number of final's frame of the video that will be process
    def FinalFrame(self)->int:
        return self.__final_frame

    @FinalFrame.setter                          # Sets the number of final's frame of the video that will be process
    def FinalFrame(self, value:int):
        self.__final_frame = value
        self.__Frames = self.rangeFrame(self.FinalFrame,self.InitFrame)

    # Properties relates with the modulation's frequency of the lock-in signal
    @property                                   # Gets the modulation's frequency of the reference's signal 
    def Modulation(self)->float:
        return self.__modulation

    @Modulation.setter                          # Sets the modulation's frequency of the reference's signal
    def Modulation(self, value: float):
        self.__modulation       = value
        self.__FramesByPeriod   = self.FbP(self.Modulation,self.FrameRate)
        self.__K                = self.KFactor(self.N,self.Modulation,self.FrameRate)
        self.__W                = self.WFactor(self.N)
    # Properties related with the k's Factor used byy this Method 
    @property                                   # gets the value of K's Factor used by this Method
    def K(self)-> int:
       return self.__K
    
    # Properties related with the N's Factor used by this Method
    @property
    def N(self)-> int:
        return self.__FramesByPeriod

    # properties related with the W's Factor used by this Method
    @property
    def W(self)->complex:
        return self.__W

    # Properties related with the WN's Factor in the current frame that is processed
    @property
    def WN(self) -> complex:
        return self.__WNValue

    # properties related with the current lock-in period in process
    @property
    def currentPeriod(self)->int:
        return self.cp(self.CurrentFrame,self.N)

    # Properties related with the images in process by this Method
    @property                                   # Gets the actually image in process by this Method
    def CurrentImage(self)->np.array:
        return self.__curr_image_frame
    
    @CurrentImage.setter                        # Load the image to be process
    def CurrentImage(self , image:MatLike):
        if self.isImageFull:
            image = Basics.imgPrepare(image)    
        else:
            image = Basics.imgDivide(image)
        
        self.__WNValue = self.Wn(self.W, self.CurrentFrame , self.K)
        self.__curr_image_frame = image * self.__WNValue

    @property                                   # Gets the last image processed by this Method
    def Thermogram(self)-> MatLike:
        return self.__last_image_Frame
    
    @Thermogram.setter                          # Save the last image processed by this Method
    def Thermogram(self, img:MatLike):
        self.CurrentImage = img
        
        if self.CurrentFrame == 1:
            self.__last_image_Frame = self.CurrentImage
        else:
            self.__last_image_Frame = self.CurrentImage + self.__last_image_Frame
        
        for watcher in self.__listener:
            watcher.CurrentImage_Processed(self.__last_image_Frame)

        self.CurrentFrame += 1

    @property                                   # Gets the variable that indicate if the image to be process is fullsize  
    def isImageFull(self)-> bool:
        return self.__Image_full

    @isImageFull.setter                         # Indicates that the image to be process is fullsize
    def isImageFull(self,value:bool):
        self.__Image_full = value

    @property    
    def Thermogram_Phase(self):
        return np.angle(self.Thermogram)

    @property
    def Thermogram_Amplitude(self):
        return np.abs(self.Thermogram)

    @property
    def FrameStatistics(self)-> dict:
        return self.__currentStatistic
    
    @property
    def reset(self) -> bool:
        return self.__Reset
    
    @reset.setter
    def reset(self, value:bool):
        self.__Reset = value
        if self.CurrentFrame >1:
            self.CurrentFrame = 1
        else:
            print('no se puede resetear')
            
    @FrameStatistics.setter
    def FrameStatistics(self, key: str, value : __Statistic):
        self.__currentStatistic[key] = value


    def __init__(self, Frames: int, FrameRate:int): 
        self.Frames = Frames
        self.FrameRate = FrameRate
        self.isImageFull = False
        self.__currentFrame = 1
        self.kernel  = np.ones((3,3),np.uint8)
        self.__currentStatistic = {'Global':0 , 'X':0, 'Y':0}


    def __init__(self):
        self.isImageFull = False
        self.__init_frame = 0
        self.__final_frame = 3000
        self.__Frame_Rate = 1
        self.__modulation = 0.2
        self.__FramesByPeriod= 60
        self.__currentFrame = 1
        self.__porcentage = 0
        self.kernel  = np.ones((3,3),np.uint8)
        self.__currentStatistic = {'Global':0 , 'X':0, 'Y':0}

    def reset_fourier(self):
        self.__currentFrame = 1
        self.__porcentage = 0
