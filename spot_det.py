import cv2
import pickle

#we're storing the bounding boxes in pickle
width,height=107,48
try:
    with open("carparkposi", 'rb') as f:
        poslist=pickle.load(f)
except:
    poslist=[]

def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        #will append the xy position into the list once the lEft button clicked
        poslist.append((x,y))
    #if we want to delet on event click when mistakely draw some other points
    if events == cv2.EVENT_RBUTTONDOWN:
        #we need to know the iteration number so enumarate
        for i, pos in enumerate(poslist):
            x1,y1=pos
            #checking current position is between x1 y1 x1+width y1+height 
            if x1 <x < x1 + width and y1 < y < y1 + height:
                poslist.pop(i)
    with open("carparkposi", "wb") as f:
        pickle.dump(poslist, f)

while True:
    img=cv2.imread("carParkImg.png")

    #if we want to display 
    for pos in poslist:
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+ height), (255,0,240), 2)
    
    #width : 158-57=101 height: 239-158=81
    #cv2.rectangle(img, (57, 192), (158, 239), (255,0,240), 2)
    
    cv2.imshow("image", img)
    #detect the mouse click
    cv2.setMouseCallback("image", mouseClick)
    cv2.waitKey(1)

