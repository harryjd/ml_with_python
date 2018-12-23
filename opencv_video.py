import cv2

cap = cv2.VideoCapture('images/7.mp4')

isOpened = cap.isOpened()
if isOpened:
    fps = cap.get(cv2.CAP_PROP_FPS)
    f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps, f_width, f_heigth)
iround = 0
while(isOpened):
    if iround >= 2000:
        break
    else :
        iround = iround + 1
        (flag, frame) = cap.read()
        if iround>=1001 and iround % 20 ==0:
            filename = 'images/notcar_img' + str(iround) + '.jpg'
            if flag== True:
                cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
print('End.')