import cv2
img = cv2.imread('images/employee.png') # the name of file for images  
classnames = [] # list or arriy 
classfile = "files/thing.names"

with open(classfile, 'rt') as f :    #   if we calling F the rt answing
    classnames = f.read() .rstrip('\n').split('\n')
#print(classnames)
    p= 'files/frozen_inference_graph.pb' 
    v='files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    net = cv2.dnn_DetectionModel(p,v) #اDetection and examination 
    net.setInputSize(320,230) # width and height 
    net.setInputScale(1.0 / 127.5)  #اsize 
    net.setInputMean((127.5,127.5,127.5))  
    net.setInputSwapRB(True) # color system  
    
    classIds , confs , bbox = net.detect(img, confThreshold=0.5)   # Detection accuracy 
    #print(classIds,bbox)
    for classId , confidence , box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img,box,color=(0,255,0),thickness=3 )
        cv2.putText(img , classnames[classId-1],
                        (box[0]+10, box[1]+20),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255) , thickness=2)
cv2.imshow('program', img)
cv2.waitKey(0)
