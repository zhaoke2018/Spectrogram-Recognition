# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Tektronix RSA_API Example
Author: Morgan Allison
Date created: 6/15
Date edited: 5/17
Windows 7 64-bit
RSA API version 3.9.0029
Python 3.6.1 64-bit (Anaconda 4.3.0)
NumPy 1.11.3, MatPlotLib 2.0.0
Download Anaconda: http://continuum.io/downloads
Anaconda includes NumPy and MatPlotLib
Download the RSA_API: http://www.tek.com/model/rsa306-software
Download the RSA_API Documentation:
http://www.tek.com/spectrum-analyzer/rsa306-manual-6

YOU WILL NEED TO REFERENCE THE API DOCUMENTATION
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ctypes import *
from os import chdir
from RSA_API import *

import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk


# acquire spectrogram from RSA306B-begin
# C:\Tektronix\RSA_API\lib\x64 needs to be added to the
# PATH system environment variable
chdir("C:\\Program Files\\Tektronix\\RSA_API\\lib\\x64")
rsa = cdll.LoadLibrary("RSA_API.dll")


"""################CLASSES AND FUNCTIONS################"""
def err_check(rs):
    if ReturnStatus(rs) != ReturnStatus.noError:
        raise RSAError(ReturnStatus(rs).name)

def search_connect():
    numFound = c_int(0)
    intArray = c_int * DEVSRCH_MAX_NUM_DEVICES
    deviceIDs = intArray()
    deviceSerial = create_string_buffer(DEVSRCH_SERIAL_MAX_STRLEN)
    deviceType = create_string_buffer(DEVSRCH_TYPE_MAX_STRLEN)
    apiVersion = create_string_buffer(DEVINFO_MAX_STRLEN)

    rsa.DEVICE_GetAPIVersion(apiVersion)
    #print('API Version {}'.format(apiVersion.value.decode()))

    err_check(rsa.DEVICE_Search(byref(numFound), deviceIDs,
                                deviceSerial, deviceType))

    if numFound.value < 1:
        # rsa.DEVICE_Reset(c_int(0))
        print('No instruments found. Exiting script.')
        exit()
    elif numFound.value == 1:
        #print('One device found.')
        print('\n\n\nOne {} found.'.format(deviceType.value.decode()))
        #print('Device serial number: {}'.format(deviceSerial.value.decode()))
        err_check(rsa.DEVICE_Connect(deviceIDs[0]))
    else:
        # corner case
        print('2 or more instruments found. Enumerating instruments, please wait.')
        for inst in deviceIDs:
            rsa.DEVICE_Connect(inst)
            rsa.DEVICE_GetSerialNumber(deviceSerial)
            rsa.DEVICE_GetNomenclature(deviceType)
            print('Device {}'.format(inst))
            print('Device Type: {}'.format(deviceType.value))
            print('Device serial number: {}'.format(deviceSerial.value))
            rsa.DEVICE_Disconnect()
        # note: the API can only currently access one at a time
        selection = 1024
        while (selection > numFound.value - 1) or (selection < 0):
            selection = int(input('Select device between 0 and {}\n> '.format(numFound.value - 1)))
        err_check(rsa.DEVICE_Connect(deviceIDs[selection]))
    rsa.CONFIG_Preset()

"""################DPX EXAMPLE################"""
def config_DPX(cf=1e9, refLevel=0, span=40e6, rbw=300e3):
    yTop = refLevel
    yBottom = yTop - 100
    yUnit = VerticalUnitType.VerticalUnit_dBm

    dpxSet = DPX_SettingStruct()
    rsa.CONFIG_SetCenterFreq(c_double(cf))
    rsa.CONFIG_SetReferenceLevel(c_double(refLevel))

    rsa.DPX_SetEnable(c_bool(True))
    rsa.DPX_SetParameters(c_double(span), c_double(rbw), c_int(801), c_int(1),
                          yUnit, c_double(yTop), c_double(yBottom), c_bool(False),
                          c_double(1.0), c_bool(False))
    rsa.DPX_SetSogramParameters(c_double(1e-3), c_double(1e-3),
                                c_double(refLevel), c_double(refLevel - 100))
    rsa.DPX_Configure(c_bool(True), c_bool(True))

    rsa.DPX_SetSpectrumTraceType(c_int32(0), c_int(2))
    rsa.DPX_SetSpectrumTraceType(c_int32(1), c_int(4))
    rsa.DPX_SetSpectrumTraceType(c_int32(2), c_int(0))

    rsa.DPX_GetSettings(byref(dpxSet))
    dpxFreq = np.linspace((cf - span / 2), (cf + span / 2), dpxSet.bitmapWidth)
    dpxAmp = np.linspace(yBottom, yTop, dpxSet.bitmapHeight)
    return dpxFreq, dpxAmp


def acquire_dpx_frame():
    frameAvailable = c_bool(False)
    ready = c_bool(False)
    fb = DPX_FrameBuffer()

    rsa.DEVICE_Run()
    rsa.DPX_Reset()

    while not frameAvailable.value:
        rsa.DPX_IsFrameBufferAvailable(byref(frameAvailable))
        while not ready.value:
            rsa.DPX_WaitForDataReady(c_int(100), byref(ready))
    rsa.DPX_GetFrameBuffer(byref(fb))
    rsa.DPX_FinishFrameBuffer()
    rsa.DEVICE_Stop()
    return fb

def extract_dpxogram(fb):
    # When converting a ctypes pointer to a numpy array, we need to
    # explicitly specify its length to dereference it correctly
    dpxogram = np.array(fb.sogramBitmap[:fb.sogramBitmapSize])
    dpxogram = dpxogram.reshape((fb.sogramBitmapHeight,
                                 fb.sogramBitmapWidth))
    dpxogram = dpxogram[:fb.sogramBitmapNumValidLines, :]

    return dpxogram


def dpx_example():

    search_connect()

    # Mobile-4G-LTE-TDD
    #cf = 2.345e9
    # FM
    #cf = 97.5e6
    # Wi-Fi: JLU.TEST
    #cf = 2.462e9
    refLevel = -30
    span = 40e6
    rbw = 100e3
    numTicks = 11
    plotFreq = np.linspace(float(entry1.get()) * 1000000 - span / 2.0, float(entry1.get()) * 1000000 + span / 2.0, numTicks) / 1e9

    for i in range(1) :
        dpxFreq, dpxAmp = config_DPX(float(entry1.get()) * 1000000, refLevel, span, rbw)
        fb = acquire_dpx_frame()
        dpxogram = extract_dpxogram(fb)
        fig = plt.gcf()
        fig.set_size_inches(7.89/3,14.19/3)
        plt.axis('off')
    	# Show the colorized DPXogram
        ax3 = fig.add_subplot(111)
        ax3.imshow(dpxogram, cmap='gist_stern')
        ax3.set_aspect(12)

        xTicks = map('{:.4}'.format, plotFreq)
        plt.xticks(np.linspace(0, fb.sogramBitmapWidth, numTicks), xTicks)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        #fig.savefig('E:\\test\\cell%i.jpg' %i, format='jpg', transparent=True, dpi=300, pad_inches = 0)
        #fig.savefig('E:\\test\\fm%i.jpg' %i, format='jpg', transparent=True, dpi=300, pad_inches = 0)
        fig.savefig(entry2.get(), format='jpg', transparent=True, dpi=300, pad_inches = 0)

def main_acquire():
    # uncomment the example you'd like to run
    dpx_example()
    global var
    var.set('获取成功')

    rsa.DEVICE_Disconnect()

# acquire spectrogram from RSA306B-end



def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label



def main_test():
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    model_file = 'C:/tmp/output_graph.pb'
    label_file = 'C:/tmp/output_labels.txt'
    input_layer = 'Placeholder'
    output_layer = 'final_result'
    file_name = entry3.get()

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        text.insert('end', (labels[i]+':'+str(results[i])+'\n'))

#GUI-begin
window = tk.Tk()
window.title('频谱图实时获取与识别')
window.geometry('600x300')

frm_l = tk.Frame(window)
frm_r = tk.Frame(window)
frm_l.pack(fill = 'both', side = 'left')
frm_r.pack(fill = 'both', side = 'right')
var = tk.StringVar()
var.set('获取')
label1 = tk.Label(frm_l, textvariable = var, bg = 'yellow', font = ('Arial', 24), width = 15, height = 2)
label1.pack()
frm1 = tk.Frame(frm_l)
frm1.pack()
frm2 = tk.Frame(frm_l)
frm2.pack()
label2 = tk.Label(frm1, text = '设置频率(MHz)', font = ('Arial', 12), width = 15, height = 2)
label2.pack(side = 'left')
entry1 = tk.Entry(frm1)
entry1.pack(side = 'left')
label3 = tk.Label(frm2, text = '存储路径', font = ('Arial', 12), width = 15, height = 2)
label3.pack(side = 'left')
entry2 = tk.Entry(frm2)
entry2.pack(side = 'left')
button1 = tk.Button(frm_l, text = '获取', width = 15, height = 2, command = main_acquire)
button1.pack()
label4 = tk.Label(frm_r, text = '测试', bg = 'yellow', font = ('Arial', 24), width = 15, height = 2)
label4.pack()
frm3 = tk.Frame(frm_r)
frm3.pack()
label5 = tk.Label(frm3, text = '读取路径', font = ('Arial', 12), width = 15, height = 2)
label5.pack(side = 'left')
entry3 = tk.Entry(frm3)
entry3.pack(side = 'left')
button2 = tk.Button(frm_r, text = '测试', width = 15, height = 2, command = main_test)
button2.pack()
text = tk.Text(frm_r, height = 10)
text.pack()
#GUI-end
window.mainloop()