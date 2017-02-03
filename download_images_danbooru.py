import urllib2
import urllib
import json
import numpy as np
import cv2

maxsize = 512

# tags = ["asu_tora","puuakachan","mankun","hammer_%28sunset_beach%29",""]

for tag in tags:
    stringreturn = urllib2.urlopen("http://danbooru.donmai.us/posts.json?tags="+tag+"&limit=100").read()
    jsonreturn = json.loads(stringreturn)
    for post in jsonreturn:
        imgurl = "http://danbooru.donmai.us"+post["large_file_url"]
        print imgurl
        if ("png" in imgurl) or ("jpg" in imgurl):

            resp = urllib.urlopen(imgurl)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            if height > width:
                scalefactor = (maxsize*1.0) / width
                res = cv2.resize(image,(int(width * scalefactor), int(height*scalefactor)), interpolation = cv2.INTER_CUBIC)
                cropped = res[0:maxsize,0:maxsize]
            if width > height:
                scalefactor = (maxsize*1.0) / height
                res = cv2.resize(image,(int(width * scalefactor), int(height*scalefactor)), interpolation = cv2.INTER_CUBIC)
                center_x = width*scalefactor*0.5
                print center_x
                cropped = res[0:maxsize,center_x - maxsize/2:center_x + maxsize/2]

            # img_edge = cv2.adaptiveThreshold(cropped, 255,
            #                                  cv2.ADAPTIVE_THRESH_MEAN_C,
            #                                  cv2.THRESH_BINARY,
            #                                  blockSize=9,
            #                                  C=2)

            cv2.imwrite("imgs/"+str(post["id"])+".jpg",cropped)
            # cv2.imwrite("imgs/"+str(post["id"])+"-edge.jpg",img_edge)
