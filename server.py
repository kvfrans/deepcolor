from bottle import route, run, template, static_file, get, post, request
import urllib2
import cv2
import numpy as np
import re
import base64

import main
from main import *

# c = Color(512, 1)
# c.loadmodel(False)

@route('/<filename:path>')
def send_static(filename):
    return static_file(filename, root='web/')


@route('/upload_canvas', method='POST')
def do_uploadc():
    # lines = request.files.get('lines')
    # colors = request.files.get('colors')
    line_data = request.forms.get("lines")
    line_data = re.sub('^data:image/.+;base64,', '', line_data)
    line_s = base64.b64decode(line_data)
    line_img = np.fromstring(line_s, dtype=np.uint8)
    line_img = cv2.imdecode(line_img, -1)

    color_data = request.forms.get("colors")
    color_data = re.sub('^data:image/.+;base64,', '', color_data)
    color_s = base64.b64decode(color_data)
    color_img = np.fromstring(color_s, dtype=np.uint8)
    color_img = cv2.imdecode(color_img, -1)

    print "Got it"
    # for c in range(0,3):
    color_img = color_img * (line_img[:,:] / 255.0)

    cv2.imwrite("uploaded/lines.jpg", line_img)
    cv2.imwrite("uploaded/colors.jpg", color_img)

    # lines_img = cv2.imdecode(np.fromstring(lines.file.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
    # lines_img = np.array(cv2.resize(lines_img, (512,512)))
    # # lines_img = cv2.adaptiveThreshold(lines_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    # lines_img = np.array([lines_img]) / 255.0
    # lines_img = np.expand_dims(lines_img, 3)
    #
    # colors_img = cv2.imdecode(np.fromstring(colors.file.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
    # colors_img = np.array(cv2.resize(colors_img, (512,512)))
    # colors_img = cv2.blur(colors_img, (100, 100))
    # colors_img = np.array([colors_img]) / 255.0
    #
    # cv2.imwrite("uploaded/lines.jpg", lines_img[0]*255)
    # cv2.imwrite("uploaded/colors.jpg", colors_img[0]*255)
    #
    # generated = c.sess.run(c.generated_images, feed_dict={c.line_images: lines_img, c.color_images: colors_img})
    #
    # cv2.imwrite("uploaded/gen.jpg", generated[0]*255)

    return static_file("uploaded/gen.jpg",
                       root=".",
                       mimetype='image/jpg')

@route('/upload', method='POST')
def do_upload():
    lines = request.files.get('lines')
    colors = request.files.get('colors')

    lines_img = cv2.imdecode(np.fromstring(lines.file.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
    lines_img = np.array(cv2.resize(lines_img, (512,512)))
    # lines_img = cv2.adaptiveThreshold(lines_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    lines_img = np.array([lines_img]) / 255.0
    lines_img = np.expand_dims(lines_img, 3)

    colors_img = cv2.imdecode(np.fromstring(colors.file.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
    colors_img = np.array(cv2.resize(colors_img, (512,512)))
    colors_img = cv2.blur(colors_img, (100, 100))
    colors_img = np.array([colors_img]) / 255.0

    cv2.imwrite("uploaded/lines.jpg", lines_img[0]*255)
    cv2.imwrite("uploaded/colors.jpg", colors_img[0]*255)

    generated = c.sess.run(c.generated_images, feed_dict={c.line_images: lines_img, c.color_images: colors_img})

    cv2.imwrite("uploaded/gen.jpg", generated[0]*255)

    return static_file("uploaded/gen.jpg",
                       root=".",
                       mimetype='image/jpg')

@route('/upload_origin', method='POST')
def do_uploado():
    lines = request.files.get('lines')
    colors = request.files.get('colors')

    lines_img = cv2.imdecode(np.fromstring(lines.file.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
    lines_img = np.array(cv2.resize(lines_img, (512,512)))
    lines_img = cv2.adaptiveThreshold(cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    lines_img = np.array([lines_img]) / 255.0
    lines_img = np.expand_dims(lines_img, 3)

    colors_img = cv2.imdecode(np.fromstring(colors.file.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
    colors_img = np.array(cv2.resize(colors_img, (512,512)))
    colors_img = cv2.blur(colors_img, (100, 100))
    colors_img = np.array([colors_img]) / 255.0

    cv2.imwrite("uploaded/lines.jpg", lines_img[0]*255)
    cv2.imwrite("uploaded/colors.jpg", colors_img[0]*255)

    generated = c.sess.run(c.generated_images, feed_dict={c.line_images: lines_img, c.color_images: colors_img})

    cv2.imwrite("uploaded/gen.jpg", generated[0]*255)

    return static_file("uploaded/gen.jpg",
                       root=".",
                       mimetype='image/jpg')

run(host="localhost", port=8000)
