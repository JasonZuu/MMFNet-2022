import cv2
from DBface import common
from utils.pull_faces import get_DBface

DBface_params_path="DBface/dbface.pth"
dbface = get_DBface(DBface_params_path)
img = cv2.imread("kuaishou.jpg")
objs = common.detect(dbface, img, threshold=0.18)
for box in objs:
    common.drawbbox(img, box)
cv2.imwrite("out.jpg", img)