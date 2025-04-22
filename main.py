import time
import os

import numpy as np
import cv2


# 相机已固定，拍摄单个齿轮，齿轮在图中的大小、位置、光照、角度固定
class GearDefectRec(object):
    def __init__(self):
        self.imgSize = None  # 合格零件图片大小
        self.circumcircleArea = None  # 合格零件外接圆面积
        self.houghParam1 = None  # 霍夫圆检测Canny阈值
        self.houghParam2 = None  # 霍夫圆检测累加器阈值
        self.houghMinr = None  # 霍夫圆检测最小半径
        self.houghMaxr = None  # 霍夫圆检测最大半径
        self.houghCircleArea = None  # 合格零件轮廓凸包霍夫圆面积
        self.entitiesPercent = None  # 合格零件齿轮中空占外接圆面积比值
        self.medianKSize = 5  # 中值滤波核大小
        self.qualifiledGearArea = 181338  # 合格齿轮实体面积（为缺齿，多齿做参照）

        self.paramsSetted = False  # 参数已经设置好了
        self.showProcess = False  # 是否展示中间检测过程图片

    # 设置合格零件参数
    def setParams(self, imgSize = (2804, 1582),
                  circumcircleArea = 306201.3095,
                  houghParam1 = 120,
                  houghParam2 = 30,
                  houghMinr = 310,
                  houghMaxr = 330,
                  houghCircleArea = 82651.6553,
                  medianKSize = 7,
                  qualifiledGearArea = 181338,
                  showProcess = False):
        self.imgSize = imgSize
        self.circumcircleArea = circumcircleArea
        self.houghParam1 = houghParam1
        self.houghParam2 = houghParam2
        self.houghMinr = houghMinr
        self.houghMaxr = houghMaxr
        self.houghCircleArea = houghCircleArea
        self.entitiesPercent = self.houghCircleArea/self.circumcircleArea
        self.medianKSize = medianKSize
        self.qualifiledGearArea = qualifiledGearArea
        self.paramsSetted = True
        self.showProcess = showProcess


    # 检测新零件
    def defectDetect(self, img, areaMinThresh = 0.85, areaMaxThresh = 1.15, completeMinThresh = 0.98, completeMaxThresh = 1.02,
                     defectWhite = 300, teethThresh = 500, middleThresh=500) -> int:
        if self.paramsSetted:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, self.imgSize)
            img = cv2.medianBlur(img, ksize=self.medianKSize)
            _, imgThresh = cv2.threshold(img, 230,255,cv2.THRESH_BINARY_INV)
            contours, heriarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print('length of contours: ', len(contours))
            # 轮廓一个为正常
            if len(contours) == 1:
                # 计算外接圆
                circumCircle = cv2.minEnclosingCircle(contours[0])  # 返回格式：((x,y),r)
                circumCircleArea = np.pi * circumCircle[1]**2
                circumCircle = np.array(circumCircle[0]+(circumCircle[1], ), dtype=np.int32)
                # 计算霍夫圆
                houghCircle = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 500, None,
                                               self.houghParam1, self.houghParam2, self.houghMinr, self.houghMaxr)
                try:
                    houghCircle = houghCircle[0]
                    # for circle in houghCircle:
                    #     print(circle)
                    #     print(np.pi * circle[2]**2)
                    #     cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), 127, 3)
                    # cv2.imshow('img', img)
                    # cv2.waitKey(0)
                except:
                    print('霍夫圆检测异常')
                    return 1

                # 计算中空部分和外接圆部分面积比值
                houghCircle = houghCircle[0]
                houghCircleArea = np.pi * houghCircle[2]**2
                houghCircle = np.array(houghCircle, dtype=np.int32)  # 霍夫圆变成整数类型，后续画掩膜

                areaRatio = houghCircleArea/circumCircleArea  # 反应中空圆占实体圆的面积大小
                ratioRatio = areaRatio/self.entitiesPercent  # 当前齿轮中空比例与正常齿轮中空比例的大小
                print('areaRatio(中空比例): ', areaRatio)
                print('ratioRatio(中空比例与正常零件中空比例之比：', ratioRatio)


                # 提取齿轮实体部分
                mask = np.zeros(img.shape, dtype=np.uint8)  # 实体掩膜
                maskCircle = np.zeros(img.shape, dtype=np.uint8)  # 实体外接圆掩膜
                maskHough = np.zeros(img.shape, dtype=np.uint8)  # 中空的掩膜应该是全空的

                cv2.drawContours(mask, contours,-1, 255,-1)  # 画轮廓，可以通过直接对轮廓的绘制，去掉轮齿缝隙中的白色，内部缺陷部分不好处理
                cv2.circle(maskCircle, circumCircle[:2], circumCircle[2], 255,-1)  # 画圆无法较好的突出mask中的轮齿，方便后续做缺陷检测
                cv2.circle(maskHough, houghCircle[:2], houghCircle[2], 255,-1)  # 提取齿轮中空部分

                cv2.circle(mask, houghCircle[:2], houghCircle[2], 0, -1)
                cv2.circle(maskCircle, houghCircle[:2], houghCircle[2], 0, -1)

                maskGood = cv2.imread(r'./qualifiledGearMask.png', 0)  # 完整的齿轮掩膜
                maskMoreLessTeeth = cv2.bitwise_xor(mask, maskGood)  # 缺齿多齿掩膜
                k = np.ones((10,10), dtype=np.uint8)*255
                maskMoreLessTeeth = cv2.morphologyEx(maskMoreLessTeeth, cv2.MORPH_OPEN, k, None, (-1,-1), 1)  # 开运算一下边缘误差
                print('conut more less teeth: ', np.count_nonzero(maskMoreLessTeeth))

                imgGear = cv2.bitwise_and(img, img, mask = mask)
                imgGear[imgGear>248] = 0
                _, imgGear = cv2.threshold(imgGear, 1 , 255, cv2.THRESH_BINARY)

                # 将mask和Gear的白色比例对比
                goodGearArea = np.count_nonzero(maskGood)  # 正常齿轮实体面积
                thisGearArea = np.count_nonzero(imgGear)  # 当前齿轮实体面积
                completeRatio = thisGearArea/goodGearArea  # 面积比值
                print('area complete ratio(当前齿轮面积和正常齿轮面积之比): ', completeRatio)

                maskCrack = cv2.bitwise_xor(mask, imgGear)  # 裂缝缺陷掩膜图像
                crackArea = np.count_nonzero(maskCrack)  # 裂缝面积
                print('crackArea: ', crackArea)

                maskHough = cv2.bitwise_and(imgThresh, imgThresh, mask = maskHough)
                maskHough = cv2.morphologyEx(maskHough, cv2.MORPH_OPEN, k, None, (-1,-1), 1)
                moreMiddleCount = np.count_nonzero(maskHough)
                print('moreMiddleCount(中空部分突出：', moreMiddleCount)

                if self.showProcess:
                    cv2.namedWindow('maskCircle', cv2.WINDOW_NORMAL)  # 外接圆掩膜
                    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)  # 齿轮掩膜
                    cv2.namedWindow('maskMoreLessTeeth', cv2.WINDOW_NORMAL)  # 多齿缺齿掩膜
                    cv2.namedWindow('maskCrack', cv2.WINDOW_NORMAL)  # 齿轮
                    cv2.namedWindow('gear', cv2.WINDOW_NORMAL)  # 齿轮
                    cv2.namedWindow('maskHough', cv2.WINDOW_NORMAL)  # 中空部分掩膜

                    cv2.imshow('mask', mask)
                    cv2.imshow('gear', cv2.addWeighted(imgGear, 1, maskHough, 1, 0))
                    cv2.imshow('maskCircle', maskCircle)
                    cv2.imshow('maskMoreLessTeeth', maskMoreLessTeeth)
                    cv2.imshow('maskCrack', maskCrack)
                    cv2.imshow('maskHough', maskHough)
                    cv2.waitKey(0)


                if ratioRatio < areaMinThresh or ratioRatio > areaMaxThresh:
                    print('零件中空比例异常')
                    return 1
                elif completeRatio < completeMinThresh or completeRatio > completeMaxThresh:
                    print('存在缺陷，不符合完整比例')
                    return 1
                elif crackArea > defectWhite:
                    print('存在缺陷, 中部缺失')
                    return 1
                elif np.count_nonzero(maskMoreLessTeeth) > teethThresh:
                    print('存在缺陷，轮齿异常')
                    return 1
                elif moreMiddleCount > middleThresh:
                    print('中空部分突出')
                    return 1
                return 0
            else:
                print('零件异常！', len(contours))
                return 1
        else:
            print('set params first!')
            return -1

if __name__ == '__main__':

    img = cv2.imread(r'./imgs/defect5.png')

    detector = GearDefectRec()  # 创建检测器实例
    # 人工干预设置相关参数，以确保检测精度
    detector.setParams(imgSize = (2804, 1582),
                      circumcircleArea = 306201.3095,
                      houghParam1 = 120,
                      houghParam2 = 30,
                      houghMinr = 310,
                      houghMaxr = 330,
                      houghCircleArea = 82651.6553,
                      medianKSize = 7,
                       showProcess=1)
    t1 = time.time()
    result = detector.defectDetect(img, areaMinThresh = 0.85, areaMaxThresh = 1.15, completeMinThresh = 0.98, completeMaxThresh = 1.02,
                     defectWhite = 300, teethThresh=500, middleThresh=500)
    print('process time: ', 1000*(time.time()-t1), 'ms')
    print(result)

