# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:10:57 2018

@author: mayan
"""

import cv2
import os
import numpy as np
import time


def empty(x):
    pass

def create_directory(sign_name):
    if not os.path.exists("./Data/Train/" +str(sign_name)):
        os.makedirs("./Data/Train/" + str(sign_name))
    if not os.path.exists("./Data/Validation/" + str(sign_name)):
        os.makedirs("./Data/Validation/" + str(sign_name))
    if not os.path.exists("./Data/Test"):
        os.makedirs("./Data/Test")



def get_parameters():
    cap=cv2.VideoCapture(0)    
    cv2.namedWindow("Calibrate")
    #imgnos=0
    #test_name=1
    #train_name=1
    print("Use the trackbars to adjust parameters until you see your hand clearly and there is no white noise in the background\n")
    
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, empty)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, empty)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, empty)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, empty)
    ret, frame = cap.read()
    print("Press enter when you are satisfied with the segmentation\n")
    print("Press ESC to exit at any point\n")

    # kernel=np.ones((3,3),np.uint8)
    while (ret):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

        #lower = np.array([l_h, l_s, l_v])
        #upper = np.array([u_h, u_s, u_v])
        roi = img[102:298, 427:623]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        result = cv2.GaussianBlur(hsv, (5, 5), 100)
        result = cv2.GaussianBlur(result, (5, 5), 100)

        #result = cv2.medianBlur(result, 5)
        mask = cv2.inRange(hsv, np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v]))
        #result = cv2.inRange(hsv, lower, upper)
        result=mask

        result = cv2.resize(result, (200, 200))
        # result= cv2.dilate(result,kernel,iterations = 1)
        cv2.imshow("test", frame)
        # cv2.imshow("mask", mask)
        cv2.imshow("result", result)
        k=cv2.waitKey(1)
        if (k == 13):
            cv2.destroyAllWindows()
            return l_h, l_s, l_v, u_h, u_s, u_v
        elif (k==27):
            print("Exiting....\n")
            cv2.destroyAllWindows()
            return



def capture_images(ges_name,l_h,l_s,l_v,u_h,u_s,u_v):
    global test_set_image_name
    create_directory(ges_name)   
    cap=cv2.VideoCapture(0)
    img_counter = 1
    t_counter = 1
    training_set_image_name = 1
    validation_set_image_name = 1
    l_h = l_h
    l_s = l_s
    l_v = l_v
    u_h = u_h
    u_s = u_s
    u_v = u_v
    ret, frame = cap.read()
    kernel=np.ones((3,3),np.uint8)
    print("Press c to save sign\n")
    print("And make the sign for a few seconds\n")
    print("Press ESC to exit\n")
    while(ret):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)


            img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

            lower = np.array([l_h, l_s, l_v])
            upper = np.array([u_h, u_s, u_v])
            roi = img[102:298, 427:623]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            result = cv2.GaussianBlur(hsv,(5,5),100)
            result = cv2.GaussianBlur(result,(5,5),100)

            #result = cv2.medianBlur(result,5)
            result = cv2.inRange(hsv, lower, upper)
            result=cv2.resize(result,(200,200))

            cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (100, 100, 255))

            #result= cv2.dilate(result,kernel,iterations = 1)
            cv2.imshow("test", frame)
            #cv2.imshow("mask", mask)
            cv2.imshow("result", result)
            if cv2.waitKey(1) == ord('c'):
                if t_counter <= 800:
                    img_name = "./Data/Train/" + str(ges_name) + "/{}.png".format(training_set_image_name)
                    #save_img = cv2.resize(result, (image_x, image_y))
                    cv2.imwrite(img_name, result)
                    print("{} written!".format(img_name))
                    training_set_image_name += 1

                if t_counter > 800 and t_counter <= 900:
                    img_name = "./Data/Validation/" + str(ges_name) + "/{}.png".format(validation_set_image_name)
                    #save_img = cv2.resize(mask, (image_x, image_y))
                    cv2.imwrite(img_name, result)
                    print("{} written!".format(img_name))
                    validation_set_image_name += 1

                if t_counter > 900 and t_counter <= 1000:
                    img_name = "./Data/Test" + "/{}.png".format(test_set_image_name)
                    #save_img = cv2.resize(mask, (image_x, image_y))
                    cv2.imwrite(img_name, result)
                    print("{} written!".format(img_name))
                    test_set_image_name += 1

                t_counter += 1

                if t_counter >1000:
                    print("Finished calibrating " + str(ges_name) + "\n")
                    cv2.destroyAllWindows()
                    print('Do u want to continue with other alphabet\n')
                    #print('\npress y for yes and n for no')
                    choice=str(input("press y for yes and n for no\n"))
                    if ((choice=='y') or  (choice=='Y')):
                        ges_name=input("Enter other gesture name\n")
                        capture_images(ges_name, l_h, l_s, l_v, u_h, u_s, u_v)

                    elif((choice=='n') or (choice=='N')):
                        return

                img_counter+=1

            if cv2.waitKey(1)==27:
                print("Exiting.....\n")
                cv2.destroyAllWindows()
                break
                return


def main():
    ges_name = input("Enter gesture name: ")
    l_h,l_s,l_v,u_h,u_s,u_v=get_parameters()
    global test_set_image_name
    test_set_image_name = 1
    capture_images(ges_name,l_h,l_s,l_v,u_h,u_s,u_v)
    
    
    
    
if __name__=="__main__":    
    main()
    
    
    
    
    
    
    
    
    
    