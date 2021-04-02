import cv2
import numpy as np
import os.path
from glob import *


def cal_for_frames(video_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()
 
    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames[1:]):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr
 
    return flow
 
 
def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1=cv2.optflow.DualTVL1OpticalFlow_create()
    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    # TVL1=cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32
 
    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
 
    return flow
 
 
def save_flow(video_flows, flow_path):
    if not os.path.exists(os.path.join(flow_path, 'u')):
        os.mkdir(os.path.join(flow_path, 'u'))
    if not os.path.exists(os.path.join(flow_path, 'v')):
        os.mkdir(os.path.join(flow_path, 'v'))
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path,'u', "{:06d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path,'v', "{:06d}.jpg".format(i)),
                    flow[:, :, 1])
 
 
def extract_flow(video_path, flow_path):
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)
    print('complete:' + flow_path)
    return
 
 
if __name__ == '__main__':

    video_path = 'C:/Users/liang/Desktop/Graduate Works/Project/Dataset/masami/section'  #预先提取好的视频帧
    save_path = 'C:/Users/liang/Desktop/Graduate Works/Project/Dataset/masami/flow'  #保存光流的路径
    extract_flow(video_path,save_path)

#prepare phptos
    # cap = cv2.VideoCapture('C:/Users/liang/Videos/Videos/masami/ma_section.mp4')
    # while (1):
    #     ret, frame = cap.read()
    #     if ret == False:
    #         break
    #     frame_index = 0
    #     frame_count = 0
    #     if cap.isOpened():
    #         success = True
    #     else:
    #         success = False
    #     while (success):
    #         success, frame = cap.read()
    #         print("---> 正在读取第%d帧:" % frame_index, success)
    #         if frame_index % 1 == 0 and success:
    #             resize_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    #             cv2.imwrite('C:/Users/liang/Desktop/Graduate Works/Project/Dataset/masami/' + "%d.jpg" % frame_count, resize_frame)
    #             frame_count += 1
    #         frame_index += 1
    #     cap.release()
