'''
version 0.1
鼠标轨迹追踪初代版本
模型输出：本次坐标点=上一次坐标点
'''
from cmath import pi
from glob import glob
from tkinter import *
from venv import create
import numpy as np
from collections import deque

#参数初始化
len =100#最大采样点
#队列
pointx=deque()
pointy=deque()
mesx=deque()
mesy=deque()
prex=deque()
prey=deque()
count=0#当前采样点个数
sigmaR=5#测量噪声方差
sigmaQ=0.1#过程噪声方差
width=2#曲线的宽度
pk_pre=sigmaQ#先验P
last_pk=sigmaQ#上一次的P
pk=sigmaQ#当前的P
kk=1#卡尔曼增益

#函数作用：输出卡尔曼滤波计算值
#                   模型坐标            测量噪声        实际坐标   采样点个数
def kalmanfilterCal(prex,prey,event,nosiePx,nosiePy,pointx,pointy,count):
    global pk
    global kk
    global pk_pre
    global last_pk
    last_pk=pk

    if count==1:
        prex.append(pointx[0])
        prey.append(pointy[0])
        pk_pre = last_pk+sigmaQ
        pk=(1-kk)*pk_pre
    else:    
        #先验估计    
        x_hat=prex[count-2]
        y_hat=prey[count-2]
        
        pk_pre = last_pk+sigmaQ
        kk=pk_pre/(pk_pre+sigmaR)
        # print(x_hat)
        #将输出点加入队列
        prex.append(np.round(x_hat+kk*(event.x+nosiePx-x_hat),2))
        prey.append(np.round(y_hat+kk*(event.y+nosiePy-y_hat),2))
        pk=(1-kk)*pk_pre

#采样点未达到len时候，进行更新    
def update1(event,pointx,pointy,mesx,mesy,prex,prey,count):

    nosiePx =np.round(np.random.normal(0, sigmaR), 2)
    nosiePy =np.round(np.random.normal(0, sigmaR), 2)  
    pointx.append(event.x)#实际坐标
    pointy.append(event.y)
    mesy.append(event.y+nosiePy)#加入噪声模拟测量坐标
    mesx.append(event.x+nosiePx)
    kalmanfilterCal(prex,prey,event,nosiePx,nosiePy,pointx,pointy,count)


def popout(pointx,pointy,mesx,mesy,prex,prey):
    pointx.popleft()
    pointy.popleft()
    mesx.popleft()
    mesy.popleft()
    prex.popleft()
    prey.popleft()


#采样点达到len个以后进行更新
def update2(event,pointx,pointy,mesx,mesy,prex,prey,count):
    #测量噪声满足正态分布
    nosiePx =np.round(np.random.normal(0, sigmaR), 2)
    nosiePy =np.round(np.random.normal(0, sigmaR), 2) 

    #去除队列最左边的采样（相当于删除过时的采样点）
    popout(pointx,pointy,mesx,mesy,prex,prey)

    #将最新采样点和测量点加入队列
    pointx.append(event.x)
    pointy.append(event.y)
    mesx.append(event.x+nosiePx)
    mesy.append(event.y+nosiePy)
    #计算卡尔曼滤波输出
    kalmanfilterCal(prex,prey,event,nosiePx,nosiePy,pointx,pointy,count)

def draw_point(event,pointx,pointy,mesx,mesy,prex,prey,count):
    global cv
    cv.create_line(pointx[0],pointy[0],pointx[1],pointy[1],fill='black',width =width)
    cv.create_line(prex[0],prey[0],prex[1],prey[1],fill='black',width =width)
    cv.create_oval(mesx[0], mesy[0],mesx[0]+1,mesy[0]+1,fill='black',width = width,outline='black')
    update2(event,pointx,pointy,mesx,mesy,prex,prey,count)
    cv.create_line(pointx[count-2], pointy[count-2],pointx[count-1],pointy[count-1],fill='blue',width = width)
    cv.create_oval(mesx[count-2], mesy[count-2],mesx[count-2]+1,mesy[count-2]+1,fill='white',width =width,outline='white') 
    cv.create_line(prex[count-2], prey[count-2],prex[count-1],prey[count-1],fill='red',width = width)

def MouseMove(event):
    global pointy
    global pointx
    global count
    global mesx
    global mesy
    global prex
    global prey

    if count == 0 :
        count+=1
        update1(event,pointx,pointy,mesx,mesy,prex,prey,count)

    #采样点没有达到len个
    # if count != len:
    #     update1(event,pointx,pointy,mesx,mesy,prex,prey,count)
    if count!=len and count!=0:
        update1(event,pointx,pointy,mesx,mesy,prex,prey,count)
        #画真实轨迹
        cv.create_line(pointx[count-1], pointy[count-1],pointx[count],pointy[count],fill='blue',width = width)
        #画测量点的散点图
        cv.create_oval(mesx[count], mesy[count],mesx[count]+1,mesy[count]+1,fill='white',width =width,outline='white') 
        #画卡尔曼输出的点
        cv.create_line(prex[count-1], prey[count-1],prex[count],prey[count],fill='red',width = width)
        count+=1


    #采样点达到 len个
    if(count == len):
        # 保持点数为len个
        draw_point(event,pointx,pointy,mesx,mesy,prex,prey,count)

last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)
window = Tk()
cv = Canvas(window,bg = 'black',width=1000,height=800)
window.geometry("1000x800")
window.title("Kalman Filter-Version0.1")
cv.pack()
window.bind("<Motion>",MouseMove)
window.mainloop()
