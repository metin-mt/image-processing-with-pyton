  # -*- coding: utf-8 -*-
"""
Created on Thu Jan 2 20:15:12 2018

@author: MONSTER
"""
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from skimage import io
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage import feature,filter,io,color        
from skimage.segmentation import clear_border
import os
import matplotlib
import matplotlib.patches as mpatches
from skimage.transform import resize
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.widgets as widgets
import cv2
from PIL import Image
import random
from guruntu import Ui_Dialog
from skimage import io

from skimage.transform import resize

from skimage.measure import structural_similarity as ssim


class MainWindow(QtGui.QMainWindow, Ui_Dialog):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)
        print "__init__"
        
        #---- Temel İşlemler-------------------------------------------------------------------------------------------------------------------------------------------------------
        self.btnResimYukle.clicked.connect(self.btnYukle)
        self.qsBoyutYukseklik.valueChanged .connect(self.BoyutYukseklik)
        self.qsBoyutGenislik.valueChanged .connect(self.BoyutGenislik)
        self.btnSifirla.clicked.connect(self.sifirla)
        self.btnGri.clicked.connect(self.gri)
        self.btnIkili.clicked.connect(self.ikili)
        self.btnPikselGoster.clicked.connect(self.goster)
        self.hsBulanklastir.valueChanged .connect(self.bulanik)
        self.dlDundur.valueChanged .connect(self.aci)
        self.btnKaydet.clicked.connect(self.kaydet)
        self.btnHistogram.clicked.connect(self.histogram)
        
        #---- Filtreleme -------------------------------------------------------------------------------------------------------------------------------------------------------
        self.btnResimYukle2.clicked.connect(self.btnYukle2)
        self.btnCanny.clicked.connect(self.canny)
        self.btnSobel.clicked.connect(self.sobel)
        self.btnPrewit.clicked.connect(self.prewit)
        self.btnTreshold.clicked.connect(self.threshold)

       #---- Labeling -------------------------------------------------------------------------------------------------------------------------------------------------------  
        self.btnLabelingYukle.clicked.connect(self.LabelingYukle)
        self.btnLabeling.clicked.connect(self.Labeling)
        self.cbLabelingBulunan.currentIndexChanged.connect(self.LabelingBulunanSecilen)  
       # self.rbtnKare.setChecked(True)
        self.btnLabelingKarsilastir.clicked.connect(self.LabelingKarsilastir)  
 
 #---- Histogram Eşitleme-------------------------------------------------------------------------------------------------------------------------------------------------------
        self.btnHistogramYukle.clicked.connect(self.HistogramYukle)   
        self.btnHistogramKarsilastir.clicked.connect(self.HistogramKarsilastir) 

 #---- Gürültü Oluştur-------------------------------------------------------------------------------------------------------------------------------------------------------
       
        self.btnGurultuYukle.clicked.connect(self.GurultuYukle)
        self.btnGurultuOlustur.clicked.connect(self.GurultuOlustur)
        self.hsGurultuOlustur.valueChanged.connect(self.GurultuOlustur10luk)
        self.btnGurultuBenzerlikHesapla.clicked.connect(self.GurultuBenzerlikHesapla)   

#---- Mantık İşlemleri -------------------------------------------------------------------------------------------------------------------------------------------------------
        self.btnMantikYukle1Resim.clicked.connect(self.MantikYukle1Resim)
        self.btnMantikYukle2Resim.clicked.connect(self.MantikYukle2Resim)
        self.btnMantikGoster.clicked.connect(self.MantikGoster)
        self.cbMantik.currentIndexChanged.connect(self.Mantik1)
 
 #----Erosion-Dilation -------------------------------------------------------------------------------------------------------------------------------------------------------
        self.btnErosionYukle.clicked.connect(self.ErosionYukle)        
        self.btnDilation.clicked.connect(self.Dilation) 
        self.btnErosion.clicked.connect(self.Erosion)        
 #----Template Matching-------------------------------------------------------------------------------------------------------------------------------------------------------
        self.btnTemplateYukle.clicked.connect(self.TemplateYukle)        
        self.btnTemplate.clicked.connect(self.Template)
        self.cbGuruntuSec.currentIndexChanged.connect(self.GuruntuSec)    
   
#----Yüz Tanima-------------------------------------------------------------------------------------------------------------------------------------------------------
        self.btnYuzTanimaYukle1.clicked.connect(self.YuzTanimaYukle1)
        self.btnYuzTanimaKisiBul.clicked.connect(self.YuzTanimaKisiBul)
        self.btnYuzTanimaYukle2.clicked.connect(self.YuzTanimaYukle2)
        self.btnKisiBul.clicked.connect(self.KisiBul)
        self.cbYuzTanimaBulunanYuzler.currentIndexChanged.connect(self.YuzTanimaBulunanYuzler)
        self.btnYuzIsimKaydet.clicked.connect(self.YuzIsimKaydet)     
        self.btnYuzEkleme.clicked.connect(self.YuzEkleme)
#---- Temel İşlemler-------------------------------------------------------------------------------------------------------------------------------------------------------
    degerh=0  
    degerg=0
    sonuc_fileName=""
    fileName=""
    fileName_sifirla=""
    budeger=0
    acdeger=0
    art=0
    
    def show_image(self, img_name,width,height):       
        pixMap = QtGui.QPixmap(img_name)   
        pixMap=pixMap.scaled(width,height)            
        pixItem = QtGui.QGraphicsPixmapItem(pixMap)
        scene2 = QGraphicsScene()
        scene2.addItem(pixItem)    
        return scene2     
        
    def mse(imageA, imageB):
	 # the 'Mean Squared Error' between the two images is the
	 # sum of the squared difference between the two images;
	 # NOTE: the two images must have the same dimension
	 err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 3)
	 err /= float(imageA.shape[0] * imageA.shape[1])
	 # return the MSE, the lower the error, the more "similar"
	 # the two images are
	 return err
    def btnYukle(self):
        self.fileName = unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.*)"))
        w,h=self.gvOrjinal.width()-5,self.gvOrjinal.height()-5
        self.gvOrjinal.setScene(self.show_image(self.fileName,w,h))
        if(len(self.fileName)>0):
            self.dlDundur.setValue(0)
            self.hsBulanklastir.setValue(0)
            self.sonuc_fileName=self.fileName
            self.fileName_sifirla=self.fileName
            img = cv2.imread(self.fileName)
            self.teBoyutYukseklik.setText(str(img.shape[0]))
            self.teBoyutGenislik.setText(str(img.shape[1]))
            
            if(int(img.shape[1])>880):
                self.w=875
            else:
                self.w=int(img.shape[1])
            if(int(img.shape[0])>440):
                self.h=435
            else:
                self.h=int(img.shape[0])
            
            self.qsBoyutYukseklik.setValue(self.h)
            self.qsBoyutGenislik.setValue(self.w)
        img = cv2.imread(self.fileName)
        img2 = cv2.imread(self.sonuc_fileName)
        liste1=[]
        liste2=[]
        
        for i in range(100):
            x=random.randint(0,int(img.shape[0])-1)
            y=random.randint(0,int(img.shape[1])-1)
            liste1.append( img[x,y])
            liste2.append( img2[x,y])   
        
       
           
    def BoyutYukseklik(self):
        self.degerh=int(self.qsBoyutYukseklik.value())-5
        self.teBoyutYukseklik.setText(str(self.degerh))
       
        
    def BoyutGenislik(self):
        self.degerg=int(self.qsBoyutGenislik.value())-5
        self.teBoyutGenislik.setText(str(self.degerg))
        
        
    def kaydet(self):
        img = cv2.imread(self.sonuc_fileName)
        avging = cv2.blur(img,(int(self.budeger)/2+1,int(self.budeger)/2+1))
        cv2.imwrite('./sonuclar/yeni_resim.png',avging)
        img = Image.open('./sonuclar/yeni_resim.png')
        rotated=img.rotate(self.acdeger)
        rotated.save('./sonuclar/yeni_resim.png')
        img = cv2.imread('./sonuclar/yeni_resim.png')
        dim=(int(self.teBoyutGenislik.toPlainText()),int(self.teBoyutYukseklik.toPlainText()))
        resized = cv2.resize(img, dim)
        cv2.imwrite('./yeni_resim.png', resized)
        
        self.gvOrjinal.setGeometry(QtCore.QRect(450, 20, 0,0))
        self.gvSonuc.setGeometry(QtCore.QRect(450, 500, 0,0))
        
       
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Masaustune Basariyla Kayit Edilmistir.")
        
    def aci(self):
        deger=self.dlDundur.value()
        img = cv2.imread(self.sonuc_fileName)
        avging = cv2.blur(img,(int(self.budeger)/2+1,int(self.budeger)/2+1))
        cv2.imwrite('./sonuclar/rotate.png',avging)
        self.label_29.setText(str(deger))
        img = Image.open('./sonuclar/rotate.png')
        rotated=img.rotate(deger)
        rotated.save("./sonuclar/rotate.png")
        w,h=self.gvSonuc.width()-5,self.gvSonuc.height()-5
        self.gvSonuc.setScene(self.show_image('./sonuclar/rotate.png',w,h))
        self.acdeger=deger
        
    def bulanik(self):
        deger=self.hsBulanklastir.value()
        img = cv2.imread(self.sonuc_fileName,1)
        avging = cv2.blur(img,(int(deger)/2+1,int(deger)/2+1))
        cv2.imwrite('./sonuclar/bulanik.png',avging)
        w,h=self.gvSonuc.width()-5,self.gvSonuc.height()-5
        self.gvSonuc.setScene(self.show_image('./sonuclar/bulanik.png',w,h))
        self.budeger=deger
        
    def goster(self):
        img2 = cv2.imread(self.sonuc_fileName)
        img21 = cv2.imread(self.fileName)
        if((int(self.tePixelYukseklik.toPlainText())< img2.shape[1] )&( int(self.tePixelGenislik.toPlainText())< img2.shape[0])):            
            self.label_43.setText("Sonuc Goruntu : "+str(img2[int(self.tePixelYukseklik.toPlainText())-1,int(self.tePixelGenislik.toPlainText())-1])+
            "\n"+"Orginal Goruntu : "+str(img21[int(self.tePixelYukseklik.toPlainText())-1,int(self.tePixelGenislik.toPlainText())-1]))
            
        else:
            print "mesaj ver"
            
    def ikili(self):
        img = cv2.imread(self.fileName,cv2.IMREAD_GRAYSCALE)
        ret,ikili=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)  
        cv2.imwrite('./sonuclar/ikili.png', cv2.bitwise_not(ikili))
        self.sonuc_fileName='./sonuclar/ikili.png'
        w,h=self.gvSonuc.width()-5,self.gvSonuc.height()-5
        self.gvSonuc.setScene(self.show_image(self.sonuc_fileName,w,h))
        
        
    def gri(self):
        img = cv2.imread(self.fileName)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./sonuclar/gri.png', gray_image)
        self.sonuc_fileName='./sonuclar/gri.png'
        w,h=self.gvSonuc.width()-5,self.gvSonuc.height()-5
        self.gvSonuc.setScene(self.show_image(self.sonuc_fileName,w,h))
        
    def sifirla(self):
        img = cv2.imread(self.fileName_sifirla)
        if(int(img.shape[1])>880):
            self.w=875
        else:
            self.w=int(img.shape[1])
        if(int(img.shape[0])>440):
           self.h=435
        else:
            self.h=int(img.shape[0])
        
        self.qsBoyutYukseklik.setValue(self.h)
        self.qsBoyutGenislik.setValue(self.w)
        w,h=self.gvOrjinal.width()-5,self.gvOrjinal.height()-5
        self.gvOrjinal.setScene(self.show_image(self.fileName,w,h))
        w,h=self.gvSonuc.width()-5,self.gvSonuc.height()-5
        self.gvSonuc.setScene(self.show_image(self.sonuc_fileName,w,h))
        self.sonuc_fileName=self.fileName_sifirla
        
    def histogram(self):
        imgo = cv2.imread(self.fileName)
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([imgo],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.savefig('./sonuclar/histogramo.png', format='png')
        plt.show()
        imgs = cv2.imread(self.sonuc_fileName)
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([imgs],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.savefig('./sonuclar/histograms.png', format='png')
        plt.show()
        w,h=self.gvOrjinalHistogram.width()-5,self.gvOrjinalHistogram.height()-5
        self.gvOrjinalHistogram.setScene(self.show_image('./sonuclar/histogramo.png',w,h))
        
        w,h=self.gvSonucHistogram.width()-5,self.gvSonucHistogram.height()-5
        self.gvSonucHistogram.setScene(self.show_image('./sonuclar/histograms.png',w,h))
        
#---- Temel İşlemler-------------------------------------------------------------------------------------------------------------------------------------------------------

#---- Filtreleme -------------------------------------------------------------------------------------------------------------------------------------------------------

    def btnYukle2(self):
        self.fileName2 = unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.*)"))
        w,h=self.gvFiltreYuklenen.width()-5,self.gvFiltreYuklenen.height()-5
        self.gvFiltreYuklenen.setScene(self.show_image(self.fileName2,w,h))
                    
    def canny(self):
        img = cv2.imread(self.fileName2,0)
        canny = cv2.Canny(img,100,200)
        cv2.imwrite('./sonuclar/canny.png', canny)
        w, h = self.gvFiltreIslem.width() - 5, self.gvFiltreIslem.height() - 5
        self.gvFiltreIslem.setScene(self.show_image('./sonuclar/canny.png', w, h))
        
    def sobel(self):
        img = cv2.imread(self.fileName2,0)
        sobe = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        cv2.imwrite('./sonuclar/sobel.png', sobe)
        w, h = self.gvFiltreIslem.width() - 5, self.gvFiltreIslem.height() - 5
        self.gvFiltreIslem.setScene(self.show_image('./sonuclar/sobel.png', w, h))
    def prewit(self):
        from scipy.ndimage import prewitt
        img = cv2.imread(self.fileName2,0)
        prewit = prewitt(img)
        cv2.imwrite('./sonuclar/prewitt.png', prewit)
        w, h = self.gvFiltreIslem.width() - 5, self.gvFiltreIslem.height() - 5
        self.gvFiltreIslem.setScene(self.show_image('./sonuclar/prewitt.png', w, h))

    def threshold(self):
        image = io.imread(self.fileName2,0)
        width, height = image.shape[:2] 
        for x in range(width):
            for y in range(height):
                r,g,b=image[x,y][:3]
                gray=int(0.2989 * r + 0.5870 * g + 0.1140 * b)
                if gray<128:
                    image[x,y]=(0,0,0)
                else:
                    image[x,y]=(255,255,255)
        io.imsave('./sonuclar/threshold.png',image)
        w, h = self.gvFiltreIslem.width() - 5, self.gvFiltreIslem.height() - 5
        self.gvFiltreIslem.setScene(self.show_image('./sonuclar/threshold.png', w, h))


#---- Filtreleme -------------------------------------------------------------------------------------------------------------------------------------------------------
    
#---- Labeling -------------------------------------------------------------------------------------------------------------------------------------------------------
    def LabelingYukle(self):
        self.fileName3 = unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.*)"))
        w,h=self.gvLabelingYukle.width()-5,self.gvLabelingYukle.height()-5
        self.gvLabelingYukle.setScene(self.show_image(self.fileName3,w,h))
        
    def Labeling(self):    
        fname=self.fileName3
        image = color.rgb2gray(io.imread(fname))
        cleared=clear_border(image)
        label_image = label(cleared)
        liste=[]
        for i,region in enumerate(regionprops(label_image)):
            minr, minc, maxr, maxc = region.bbox
            liste1=minr, minc, maxr, maxc
            liste.append(liste1)        
            bolge=image[minr:maxr,minc:maxc]
            io.imsave('./labeling/Goruntu_'+str(i)+'.png',bolge)
        img = cv2.imread('./sekiller.png', cv2.IMREAD_COLOR)
        for i in  range(len(liste)):
            cv2.rectangle(img,(liste[i][1], liste[i][0]), (liste[i][3], liste[i][2]), (137,231,111), 2)
        cv2.imwrite('./sonuclar/labeling.png', img)
        w,h=self.gvLabelingYukle.width()-5,self.gvLabelingYukle.height()-5
        self.gvLabelingYukle.setScene(self.show_image('./sonuclar/labeling.png',w,h))
        klasor=os.listdir('./labeling/')
        for item in klasor:
            self.cbLabelingBulunan.addItem(item)
            
    def LabelingBulunanSecilen(self):
        w,h=self.gvLabelingIslem.width()-5,self.gvLabelingIslem.height()-5
        self.gvLabelingIslem.setScene(self.show_image("./labeling/"+self.cbLabelingBulunan.currentText(),w,h))
        
    def LabelingKarsilastir(self):
        img=cv2.imread("./labeling/"+self.cbLabelingBulunan.currentText(),1)
        yukseklik=img.shape[0]
        genislik=img.shape[1]
        sekil=""
        if(self.rbtnDikdortgen.isChecked()):
            radio=self.rbtnDikdortgen.text()
        else:
            radio=self.rbtnKare.text()
        for i in range(yukseklik):
            for j in range(genislik):
                if(img[i,j][0]==255 & img[i,j][1]==255 & img[i,j][2]==255):
                    
                    sekil="Bu goruntu dikdortgen"
                else:
                    sekil="Bu goruntu "+radio+" degildir"
                    break
        if(sekil=="Bu goruntu dikdortgen"):
            if(yukseklik==genislik):
                if(self.rbtnDikdortgen.isChecked()):
                    sekil="Bu goruntu dikdortgen  "
                else:
                    sekil="Bu goruntu Kare"
            else:
                if(self.rbtnDikdortgen.isChecked()):
                    sekil="Bu goruntu dikdortgen"
                else:
                    sekil="Bu goruntu Kare degil"
        self.lblKarsilastir.setText(sekil)

#---- Labeling -------------------------------------------------------------------------------------------------------------------------------------------------------

#---- Histogram Eşitleme -----------------------------------------------------------------------------------------------------------------------------------------
    def HistogramYukle(self):
        self.fileName4 = unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.*)"))
      
        w,h=self.gvHistogramYukle.width()-5,self.gvHistogramYukle.height()-5
        self.gvHistogramYukle.setScene(self.show_image(self.fileName4,w,h))
        
        img = cv2.imread(self.fileName4)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        cv2.imwrite('./sonuclar/histogram_esitleme.png', img_output)
        w,h=self.gvHistogramSonuc.width()-5,self.gvHistogramSonuc.height()-5
        self.gvHistogramSonuc.setScene(self.show_image('./sonuclar/histogram_esitleme.png',w,h))
        
    def HistogramKarsilastir(self):
        img = cv2.imread(self.fileName4,0)
        img2 = cv2.imread('./sonuclar/histogram_esitleme.png',0)
        print img.shape,img2.shape
        benzerlik_degeri = ssim(img, img2)
        
        img=resize(img, (img.shape[0], img.shape[1]))
    
        img2=resize(img2, (img2.shape[0], img2.shape[1])) 
        
        err = np.sum((img.astype("float") - img2.astype("float")) ** 2)
        err = err /float(img.shape[0] * img.shape[1])
        self.lblHistogramKarsilastir.setText("SSIM : " +str(benzerlik_degeri)+" MSE : "+str(err))
#---- Histogram Eşitleme -------------------------------------------------------------------------------------------------------------------------------------------------------
        
#---- Gürültü Oluştur-------------------------------------------------------------------------------------------------------------------------------------------------------
       
       
    def GurultuYukle(self):
        self.fileName5 = unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.*)"))
        w,h=self.gvGurultuYukle.width()-5,self.gvGurultuYukle.height()-5
        self.gvGurultuYukle.setScene(self.show_image(self.fileName5,w,h))
   
    def GurultuOlustur10luk(self):
        self.lblGurultu.setText(str(self.hsGurultuOlustur.value()))
             
    def GurultuOlustur(self):
        m = cv2.imread(self.fileName5)
        h,w,bpp = np.shape(m)
        yzde=((h*w)/100)*int(self.hsGurultuOlustur.value())
        for py in range(0,yzde):
            m[random.randint(0, h-1)][random.randint(0, w-1)] = [0,0,0]
        cv2.imwrite('./sonuclar/gurultu.png',m)
        w,h=self.gvGurultuSonuc.width()-5,self.gvGurultuSonuc.height()-5
        self.gvGurultuSonuc.setScene(self.show_image('./sonuclar/gurultu.png',w,h))
        self.fileNamearanan5='./sonuclar/gurultu.png'
        
    def GurultuBenzerlikHesapla(self): 
        img = cv2.imread(self.fileName5,0)
        img2 = cv2.imread(self.fileNamearanan5,0)
        benzerlik_degeri = ssim(img, img2)
        
        img=resize(img, (img.shape[0], img.shape[1]))
    
        img2=resize(img2, (img2.shape[0], img2.shape[1])) 
        
        err = np.sum((img.astype("float") - img2.astype("float")) ** 2)
        err = err /float(img.shape[0] * img.shape[1])
        self.lblSsim.setText("SSIM : " +str(benzerlik_degeri))
        self.lblMse.setText("MSE : "+str(err))

 #---- Gürültü Oluştur-------------------------------------------------------------------------------------------------------------------------------------------------------

#---- Mantık İşlemleri -------------------------------------------------------------------------------------------------------------------------------------------------------

    def MantikYukle1Resim(self):
        self.fileName6 = unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.*)"))
        w,h=self.gvMantik1Yukle.width()-5,self.gvMantik1Yukle.height()-5
        self.gvMantik1Yukle.setScene(self.show_image(self.fileName6,w,h))
    def MantikYukle2Resim(self):
        self.fileName6_2 = unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.*)"))
        w,h=self.gvMantik2Yukle.width()-5,self.gvMantik2Yukle.height()-5
        self.gvMantik2Yukle.setScene(self.show_image(self.fileName6_2,w,h))

    def MantikGoster(self):
        img1 = cv2.imread(self.fileName6)
        img2 = cv2.imread(self.fileName6_2)

        if(int(self.cbMantik.currentIndex())==0):
            and_out = cv2.bitwise_and(img1, img2)
            cv2.imwrite('./sonuclar/sonuc.png', and_out)

        elif(int(self.cbMantik.currentIndex())==1):
            or_out=cv2.bitwise_or(img1, img2)
            cv2.imwrite('./sonuclar/sonuc.png', or_out)

        elif(int(self.cbMantik.currentIndex())==2):
            xor_out = cv2.bitwise_xor(img1, img2)
            cv2.imwrite('./sonuclar/sonuc.png', xor_out)

        elif(int(self.cbMantik.currentIndex())==3):
            not_out = cv2.bitwise_not(img1)
            cv2.imwrite('./sonuclar/sonuc.png', not_out)
            
        elif(int(self.cbMantik.currentIndex())==4):
            not_out = cv2.bitwise_not(img2)
            cv2.imwrite('./sonuclar/sonuc.png', not_out)
            
        w, h = self.gvMantikSonuc.width() - 5, self.gvMantikSonuc.height() - 5
        self.gvMantikSonuc.setScene(self.show_image('./sonuclar/sonuc.png', w, h))

    def Mantik1(self):
        print int(self.cbMantik.currentIndex())
    

#---- Mantık İşlemleri -------------------------------------------------------------------------------------------------------------------------------------------------------
     
#----Erosion-Dilation -------------------------------------------------------------------------------------------------------------------------------------------------------
    def ErosionYukle(self):
        self.fileName7 = unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.*)"))
        w,h=self.gvErosionYukle.width()-5,self.gvErosionYukle.height()-5
        self.gvErosionYukle.setScene(self.show_image(self.fileName7,w,h))
        self.sonucfileName7=self.fileName7 
        self.kernel = np.ones((5,5),np.uint8)
    def Erosion(self):
        img = cv2.imread(self.sonucfileName7,0)        
        erosion = cv2.erode(img,self.kernel,iterations = 1)
        cv2.imwrite('./sonuclar/erosion.png', erosion)
        self.sonucfileName7='./sonuclar/erosion.png'
        w,h=self.gvErosionSonuc.width()-5,self.gvErosionSonuc.height()-5
        self.gvErosionSonuc.setScene(self.show_image(self.sonucfileName7,w,h))
        
    def Dilation(self):
        img = cv2.imread(self.sonucfileName7,0)        
        dilation = cv2.dilate(img,self.kernel,iterations = 1)
        cv2.imwrite('./sonuclar/dilation.png', dilation)
        self.sonucfileName7='./sonuclar/dilation.png'
        w,h=self.gvErosionSonuc.width()-5,self.gvErosionSonuc.height()-5
        self.gvErosionSonuc.setScene(self.show_image(self.sonucfileName7,w,h))
#----Erosion-Dilation -------------------------------------------------------------------------------------------------------------------------------------------------------

 
#----Template Matching-------------------------------------------------------------------------------------------------------------------------------------------------------
    def templateMatching(self):
        img = cv2.imread(self.fileName8,0)
        img2 = img.copy()
        Template = cv2.imread("./templet/"+self.cbGuruntuSec.currentText(),0)
        w, h = Template.shape[::-1]
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        i=0
        print len(methods)
        for meth in methods:
            img = img2.copy()
            method = eval(meth)
            print i
            # Apply template Matching
            res = cv2.matchTemplate(img,Template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            print top_left, bottom_right
            cv2.rectangle(img,top_left, bottom_right, 128, 2)
            plt.subplot(121),plt.imshow(res,cmap = 'gray')
            i=i+1
            plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img,cmap = 'gray')
            plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            plt.savefig('./TemplateMatching/plot'+str(i)+'.png')
            plt.show()
            
            img1 = Image.open('./TemplateMatching/plot'+str(i)+'.png')
            img12 = img1.crop((85, 150, 400, 500))
            img12.save('./TemplateMatching/temp/'+str(meth)+'.png')
            
            
            ima=cv2.imread(self.fileName8)
            cv2.rectangle(ima,top_left, bottom_right, (0,0,255), 2)
            cv2.imwrite('./TemplateMatching/bolge/'+str(meth)+'.png',ima)
            
            img11 = Image.open(self.fileName8)
            img12 = img11.crop((top_left[0],top_left[1], bottom_right[0] ,bottom_right[1]))
            img12.save('./TemplateMatching/sonuc/'+str(meth)+'.png')
            
        w,h=self.gvCcoeff1.width()-5,self.gvCcoeff1.height()-5
        self.gvCcoeff1.setScene(self.show_image("./TemplateMatching/bolge/cv2.TM_CCOEFF.png",w,h))
        self.gvSqdiff1.setScene(self.show_image("./TemplateMatching/bolge/cv2.TM_SQDIFF.png",w,h))
        self.gvSqdiffNormed1.setScene(self.show_image("./TemplateMatching/bolge/cv2.TM_SQDIFF_NORMED.png",w,h))
        self.gvCcoeffNormed1.setScene(self.show_image("./TemplateMatching/bolge/cv2.TM_CCOEFF_NORMED.png",w,h))
        self.gvCcorrNormed1.setScene(self.show_image("./TemplateMatching/bolge/cv2.TM_CCORR_NORMED.png",w,h))
        self.gvCcorr1.setScene(self.show_image("./TemplateMatching/bolge/cv2.TM_CCORR.png",w,h))
        
        self.gvCcoeff3.setScene(self.show_image("./TemplateMatching/sonuc/cv2.TM_CCOEFF.png",w,h))
        self.gvSqdiff3.setScene(self.show_image("./TemplateMatching/sonuc/cv2.TM_SQDIFF.png",w,h))
        self.gvSqdiffNormed3.setScene(self.show_image("./TemplateMatching/sonuc/cv2.TM_SQDIFF_NORMED.png",w,h))
        self.gvCcoeffNormed3.setScene(self.show_image("./TemplateMatching/sonuc/cv2.TM_CCOEFF_NORMED.png",w,h))
        self.gvCcorrNormed3.setScene(self.show_image("./TemplateMatching/sonuc/cv2.TM_CCORR_NORMED.png",w,h))
        self.gvCcorr3.setScene(self.show_image("./TemplateMatching/sonuc/cv2.TM_CCORR.png",w,h))
        
        
        self.gvCcoeff2.setScene(self.show_image("./TemplateMatching/temp/cv2.TM_CCOEFF.png",w,h))
        self.gvSqdiff2.setScene(self.show_image("./TemplateMatching/temp/cv2.TM_SQDIFF.png",w,h))        
        self.gvSqdiffNormed2.setScene(self.show_image("./TemplateMatching/temp/cv2.TM_SQDIFF_NORMED.png",w,h))        
        self.gvCcoeffNormed2.setScene(self.show_image("./TemplateMatching/temp/cv2.TM_CCOEFF_NORMED.png",w,h))        
        self.gvCcorrNormed2.setScene(self.show_image("./TemplateMatching/temp/cv2.TM_CCORR_NORMED.png",w,h))        
        self.gvCcorr2.setScene(self.show_image("./TemplateMatching/temp/cv2.TM_CCORR.png",w,h))
        listem=[]
        img = cv2.imread("./templet/"+self.cbGuruntuSec.currentText(),0)
        klasor=os.listdir('./TemplateMatching/sonuc/')
        for item in klasor:
            print ('./TemplateMatching/sonuc/'+item)
            img2 = cv2.imread('./TemplateMatching/sonuc/'+item,0)
           
            benzerlik_degeri = ssim(img, img2)
            
          
            
            err = np.sum((img.astype("float") - img2.astype("float")) ** 2)
            err = err /float(img.shape[0] * img.shape[1])
            listem.append([benzerlik_degeri,err])

        self.lblccoeff.setText("SSIM:" +str(listem[0][0])+" MSE:"+str(listem[0][1]))
        self.lblccoeffNormed.setText("SSIM:" +str(listem[1][0])+" MSE:"+str(listem[1][1]))
        self.lblccorr.setText("SSIM:" +str(listem[2][0])+" MSE:"+str(listem[2][1]))
        self.lblccorrNormed.setText("SSIM:" +str(listem[3][0])+" MSE:"+str(listem[3][1]))
        self.lblsqdiff.setText("SSIM:" +str(listem[4][0])+" MSE:"+str(listem[4][1]))
        self.lblsqdiffNormed.setText("SSIM:" +str(listem[5][0])+" MSE:"+str(listem[5][1]))
       
    def Template(self):
        self.templateMatching()
        klasor=os.listdir('./TemplateMatching/')
        liste=[]
        for item in klasor:
            liste.append(item)
        
    def TemplateYukle(self):
        self.fileName8 = unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.*)"))
        
        fname=self.fileName8
        image = color.rgb2gray(io.imread(fname))
        cleared=clear_border(image)
        label_image = label(cleared)
        liste=[]
        for i,region in enumerate(regionprops(label_image)):
            minr, minc, maxr, maxc = region.bbox
            liste1=minr, minc, maxr, maxc
            liste.append(liste1)        
            bolge=image[minr:maxr,minc:maxc]
            io.imsave('./templet/Goruntu_'+str(i)+'.png',bolge)
        img = cv2.imread('./sekiller.png', cv2.IMREAD_COLOR)
        for i in  range(len(liste)):
            cv2.rectangle(img,(liste[i][1], liste[i][0]), (liste[i][3], liste[i][2]), (137,231,111), 2)
        cv2.imwrite('./sonuclar/labeling.png', img)
        klasor=os.listdir('./templet/')
        w,h=self.gvTemlateYukle.width()-5,self.gvTemlateYukle.height()-5
        self.gvTemlateYukle.setScene(self.show_image('./sonuclar/labeling.png',w,h))
        for item in klasor:
            self.cbGuruntuSec.addItem(item)
        
    def GuruntuSec(self):
        w,h=self.gvTemplateBulunan.width()-5,self.gvTemplateBulunan.height()-5
        self.gvTemplateBulunan.setScene(self.show_image("./templet/"+self.cbGuruntuSec.currentText(),w,h))
#----Template Matching-------------------------------------------------------------------------------------------------------------------------------------------------------


#----Yüz Tanima-------------------------------------------------------------------------------------------------------------------------------------------------------
     
        
    def YuzTanimaYukle2(self):
        self.fileNametemp = unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.*)"))
        w,h=self.gvYuzTanimaYukle2.width()-5,self.gvYuzTanimaYukle2.height()-5
        self.gvYuzTanimaYukle2.setScene(self.show_image(self.fileNametemp,w,h))
        
    def YuzTanimaYukle1(self):
        self.fileNametemp2 = unicode(QtGui.QFileDialog.getOpenFileName(self, u"Düzenlenecek dosyayı seçin", ".", u"Resim dosyaları (*.*)"))
        w,h=self.gvYuzTanimaYukle1.width()-5,self.gvYuzTanimaYukle1.height()-5
        self.gvYuzTanimaYukle1.setScene(self.show_image(self.fileNametemp2,w,h))

    def YuzIsimKaydet(self):
        klasor=os.listdir('./dataset/')

        if self.leKaydedilecekIsim.text() in(klasor):
            self.lblUyari.setText("Ayni ismde kayıt var ")
        else:
            os.mkdir('./dataset/'+self.leKaydedilecekIsim.text())
            image =cv2.imread('./bulunan/'+self.cbYuzTanimaBulunanYuzler.currentText())
            cv2.imwrite('./dataset/'+self.leKaydedilecekIsim.text()+'/'+str(len(klasor))+'.png', image)
            os.remove('./bulunan/'+self.cbYuzTanimaBulunanYuzler.currentText())
            self.leKaydedilecekIsim.setText("")
            self.cbYuzTanimaBulunanYuzler.clear()
            klasor=os.listdir('./bulunan/')
            for item in klasor:
                self.cbYuzTanimaBulunanYuzler.addItem(item)
                
    def YuzTanimaBulunanYuzler(self):
        klasor=os.listdir('./bulunan/')
        if(len(klasor)<1):
            self.gvYuzTanimaBulunanYuzler.setVisible(False)
        w,h=self.gvYuzTanimaBulunanYuzler.width()-5,self.gvYuzTanimaBulunanYuzler.height()-5
        self.gvYuzTanimaBulunanYuzler.setScene(self.show_image('./bulunan/'+self.cbYuzTanimaBulunanYuzler.currentText(),w,h))
   
    def KisiBul(self):
        self.findYuz(self.fileNametemp,'./bulunan/')
        w,h=self.gvYuzTanimaYukle2.width()-5,self.gvYuzTanimaYukle2.height()-5
        self.gvYuzTanimaYukle2.setScene(self.show_image('./sonuclar/sonuc.png',w,h))
        klasor=os.listdir('./bulunan/')
        for item in klasor:
            self.cbYuzTanimaBulunanYuzler.addItem(item)

    def findYuz(self,filename,nereye):
        save_dir=nereye
        faceCascade = cv2.CascadeClassifier('./haarcascade.xml')    
        image = cv2.imread(filename)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30)
        )
    
        im = Image.open(filename)
        # Draw a rectangle around the faces
        sayac=0
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            sayac+=1 # yuzleri kaydetmek icin
            save_file_name=("bulunanyuz"+str(sayac))+'.png'
            imcrop = im.crop((x,y, x+w,y+h))
            self.isimler.append([x,y,""])
            imcrop.save(save_dir+save_file_name)
        cv2.imwrite('./sonuclar/sonuc.png', image)
        
    def show_image(self, img_name,width,height):       
        pixMap = QtGui.QPixmap(img_name)   
        pixMap=pixMap.scaled(width,height)            
        pixItem = QtGui.QGraphicsPixmapItem(pixMap)
        scene2 = QGraphicsScene()
        scene2.addItem(pixItem)    
        return scene2     
    isimler=[]
    
    def YuzEkleme(self):
        os.remove('./bulunan/'+self.cbYuzTanimaBulunanYuzler.currentText())
        self.leKaydedilecekIsim.setText("")
        self.cbYuzTanimaBulunanYuzler.clear()
        klasor=os.listdir('./bulunan/')
        for item in klasor:
            self.cbYuzTanimaBulunanYuzler.addItem(item)
        
    def YuzTanimaKisiBul(self):
        self.findYuz(self.fileNametemp2,'./bulunan/')
        w,h=self.gvYuzTanimaYukle1.width()-5,self.gvYuzTanimaYukle1.height()-5
        self.gvYuzTanimaYukle1.setScene(self.show_image('./sonuclar/sonuc.png',w,h))
        klasor3=os.listdir('./bulunan/')
        self.findYuz(self.fileNametemp2,'./bulunan/')
        kisi=len(klasor3)
        
        klasor1=os.listdir('./dataset/')
        kim=""
        i=0
        dosya = open('./yoklama.txt','a')
        print len(klasor3)
        for item in klasor3:
            deger=1.0
            print item
            for item1 in klasor1:
                klasor2=os.listdir('./dataset/'+item1+"/")
                for item2 in klasor2:
                    imageA=io.imread('./bulunan/'+item,0)
                    imageB=io.imread('./dataset/'+item1+"/"+item2,0)
                    imageA=resize(imageA, (100, 100))
                    imageB=resize(imageB, (100, 100))
                    benzerlik_degeri = self.mse(imageA, imageB)
                    if(deger>benzerlik_degeri):
                        deger=benzerlik_degeri
                        kim=item1
            print kim
            self.isimler[i][2]=kim
            i=i+1
            os.remove('./bulunan/'+item)
            dosya.write(kim+",")
        dosya.write('\n')
        dosya.close()
        resim=cv2.imread('./sonuc.png')
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i in range(kisi):
            cv2.putText(resim, self.isimler[i][2], (self.isimler[i][0],self.isimler[i][1]), font , 1, (255,255,255), 3, cv2.LINE_8)
        cv2.imwrite('./sonuc.png', resim)
        w,h=self.gv_10.width()-5,self.gv_10.height()-5
        self.gv_10.setScene(self.show_image('./sonuc.png',w,h))
        
        print self.isimler
  
#----Yüz Tanima-------------------------------------------------------------------------------------------------------------------------------------------------------

