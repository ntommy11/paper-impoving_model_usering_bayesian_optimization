import torch
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import json

def get_idx_from_lat_lon(lat,lon,min_lat,min_lon,row,col):
    row_idx = (lon-min_lon)//row
    idx = int(row_idx + 24*((lat-min_lat)//col))
    if 0<= idx < 24*24:
        return idx
    else:
        return 0 if idx<0 else 24*24-1
n = 39
path = f"./track_data/0{n}/Trajectory/"
cnn_train_data = {
    'train':[]
}

win_size = 100

for filename in glob.glob(os.path.join(path, '*.plt')):
    datalist = np.genfromtxt(filename, delimiter=' ',skip_header=6, dtype=str,autostrip=True)
    #datalist의 모습-> '39.999954,116.327404,0,229,39976.9191666667,2009-06-12,22:03:36'
    one_track = [] #현재 trajectory pattern의 인덱스(0~575)의 시퀀스
    max_lat,min_lat,max_lon,min_lon = -1,9999,-1,9999
    
    print(filename)
    for d in datalist:
        location = d.split(',')[:2]
        location[0],location[1] = float(location[0]),float(location[1])
        if max_lat < location[0]:
            max_lat = location[0]
        if min_lat > location[0]:
            min_lat = location[0]
        if max_lon < location[1]:
            max_lon = location[1]
        if min_lon > location[1]:
            min_lon = location[1]
        one_track.append([location[0],location[1]])
    
    if len(one_track) <= 101:
        continue
    print(len(one_track))
    
    cnn_x_train=[]
    cnn_y_train=[]
    
    row = (max_lon-min_lon)/24
    col = (max_lat-min_lat)/24
    
    
    for i in range(len(one_track)-win_size):
        tmp_track = one_track[i:i+win_size]
        tmp_img = [0]*(24*24)
        for tmp in tmp_track:
            tmp_img[get_idx_from_lat_lon(tmp[0],tmp[1],min_lat,min_lon,row,col)]+= 1
            
        current_idx = get_idx_from_lat_lon(tmp[0],tmp[1],min_lat,min_lon,row,col)
        next_idx = get_idx_from_lat_lon(one_track[i+win_size][0],one_track[i+win_size][1],min_lat,min_lon,row,col)
        
        tmp_img[current_idx] += 20 
        tmp_img = [tmp_img[i:i+24] for i in range(0,24*24,24)]
        
        if current_idx == next_idx: continue
        else: 
            cnn_x_train.append(tmp_img)
            cnn_y_train.append(next_idx)
            #print('next_idx:', next_idx)
        
    for i in range(len(cnn_x_train)):
        cnn_train_data['train'].append([cnn_x_train[i],cnn_y_train[i]])
    
    print("%s is complete "% filename)
    
    track_img = [0]*(24*24)
    print(len(one_track))
    
#     for one in one_track:
#         track_img[get_idx_from_lat_lon(one[0],one[1],min_lat,min_lon,row,col)]+=1
        
#     track_img = [track_img[i:i+24] for i in range(0,24*24,24)]
'''
    if len(cnn_x_train)>50:
        for i in range(30,35):
            plt.imshow(cnn_x_train[i], cmap=plt.cm.gray_r, interpolation='nearest')
            plt.pause(0.0001)
            print(cnn_y_train[i])
        for i in range(len(cnn_x_train)-5, len(cnn_x_train)):
            plt.imshow(cnn_x_train[i], cmap=plt.cm.gray_r, interpolation='nearest')
            plt.pause(0.0001)
            print(cnn_y_train[i])
    print(len(cnn_x_train))


'''


with open(f"cnn_data0{n}.json", 'w') as outfile:
    json.dump(cnn_train_data, outfile)
