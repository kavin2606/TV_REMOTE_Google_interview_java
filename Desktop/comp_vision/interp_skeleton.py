import cv2
import sys
import numpy as np
import pickle
import os
import math
BLUR_OCC = 3


def readFlowFile(file):
    '''
    credit: this function code is obtained from: https://github.com/Johswald/flow-code-python
    '''
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def find_holes(flow):
    '''
    Find a mask of holes in a given flow matrix
    Determine it is a hole if a vector length is too long: >10^9, of it contains NAN, of INF
    :param flow: an dense optical flow matrix of shape [h,w,2], containing a vector [ux,uy] for each pixel
    :return: a mask annotated 0=hole, 1=no hole
    '''
    flow = np.asarray(flow)
    holes = np.empty((380,420))
    
    h = flow.shape[0]
    w = flow.shape[1]
    u = flow[:,:,0]
    v = flow[:,:,1]
    
    for x in range(u.shape[0]):
        for y in range(u.shape[1]):
            if u[x][y] > 10**9:
                holes[x][y] = 0
            else:
                holes[x][y] = 1
    return holes

def find_good_neighbours(flow,holes,y,x):
    my_neighbours=[]
    h,w,_ = flow.shape
    checkneighbours={'lb':False,'b':False,'rb':False,'r':False,'ra':False,'a':False,'la':False,'l':False}
    if(y<379):
        checkneighbours['b']=True
        #hole below
        if(holes[y+1][x] != 0):
            my_neighbours.append(flow[y+1][x])
        #hole left lower diagonal
        checkneighbours['lb']=True
        if(x>0):
            if(holes[y+1][x-1] != 0):
                my_neighbours.append(flow[y+1][x-1])
        checkneighbours['rb']=True
        #hole right lower diagonal
        if(x<419):
            if(holes[y+1][x+1] != 0):
                my_neighbours.append(flow[y+1][x+1])

    if(y>0):
        #hole above
        checkneighbours['a']=True
        if(holes[y-1][x] != 0):
            my_neighbours.append(flow[y-1][x])
        #hole left above diagonal
        checkneighbours['la']=True
        if(x>0):
            if(holes[y-1][x-1] != 0):
                my_neighbours.append(flow[y-1][x-1])
        #hole right above diagonal
        checkneighbours['ra']==True
        if(x<419):
            if(holes[y-1][x+1] != 0):
                my_neighbours.append(flow[y-1][x+1])
    if(x>0):
        #hole left
        checkneighbours['l']=True
        if(holes[y][x-1] != 0):
            my_neighbours.append(flow[y][x-1])
    if(x<419):
        #hole right
        checkneighbours['r']=True
        if(holes[y][x+1] != 0):
            my_neighbours.append(flow[y][x+1])

    flow[y][x] = np.mean(my_neighbours)
    
    
#print("hi",len(my_neighbours))
    return flow

def holefill(flow, holes):
    '''
    fill holes in order: row then column, until fill in all the holes in the flow
    :param flow: matrix of dense optical flow, it has shape [h,w,2]
    :param holes: a binary mask that annotate the location of a hole, 0=hole, 1=no hole
    :return: flow: updated flow
    '''
    h,w,_ = flow.shape
    has_hole=1
        #while has_hole==1:
        # to be completed ...
        # ===== loop all pixel in x, then in y
    for y in range(0, h):
        for x in range(0,w):
            if holes[y][x] == 0:
                flow = find_good_neighbours(flow,holes,y,x)

#has_hole=0
    return flow

def occlusions(flow0, frame0, frame1):
    '''
    Follow the step 3 in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.
    :param flow0: dense optical flow
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :return:
    '''
    height,width,_ = flow0.shape
    occ0 = np.zeros([height,width],dtype=np.float32)
    occ1 = np.zeros([height,width],dtype=np.float32)
#
#     ==================================================
#     ===== step 4/ warp flow field to target frame
#     ==================================================
    flow1 = interpflow(flow0, frame0, frame1, 1.0)
    pickle.dump(flow1, open('flow1.step4.data', 'wb'))
#     ====== score
    flow1       = pickle.load(open('flow1.step4.data', 'rb'))
    flow1_step4 = pickle.load(open('flow1.step4.sample', 'rb'))
#    print("hi",flow1_step4[100,100])
#    print("hi",flow1[100,100])
#    print("hi",flow1_step4[200,200])
#    print("hi",flow1[200,200])
#    print("hi",flow1_step4[300,300])
#    print("hi",flow1[300,300])
#    print("hi",flow1_step4[-15,-15])
#    print("hi",flow1[-15,-15])
#    print(flow1,flow1_step4)
    diff = np.sum(np.abs(flow1-flow1_step4))
    print('flow1_step4',diff)

#     ==================================================
#     ===== main part of step 5
#     ==================================================
    h,w,_ = flow0.shape

    for y in range(0, h):
        for x in range(0,w):
            uv=flow0[y,x]
            x1,y1 =np.round(np.asarray([x,y])+uv).astype(int)
            if(x1 >=0 and x1 < width and y1 >=0 and y1 < height):
                uv1 = flow1_step4[y1,x1]
                if(not(math.sqrt(uv1[0]**2+uv1[1]**2)>10**9)):
                    diff = abs(uv-uv1)
                if (diff[0]+diff[1] > 0.5):
                    occ1[y,x] =1
            else:
                occ1[y,x] =1

    for y in range(0, h):
        for x in range(0,w):
            uv=(flow1_step4[y,x])
            if(math.sqrt(uv[0]**2+uv[1]**2)>10**9):
                occ0[y,x] = 1

    return occ0,occ1

def interpflowhelper(flow,frame0,frame1,iflow,mapping,entry,xi,yi,x,y,v,u):
    height,width,len = flow0.shape
    if(xi >=0 and xi < width and yi >=0 and yi<height):
        if(entry[yi,xi] == 0):
            iflow[yi,xi]=flow[y,x]
            mapping[yi,xi] = y,x,v,u
            entry[yi,xi] = 1
        else:
            my,mx,mv,mu=mapping[yi,xi]
            mx=(mx+mv).astype(int)
            my=(my+mu).astype(int)
            my_new=int(frame0[my,mx,0]+y)
            mx_new=int(frame0[my,mx,1]+x)
            if(mx_new >=0 and mx_new < width and my_new >=0 and my_new<height):
                diff1_temp=np.abs(frame0[my,mx]-frame1[my_new,mx_new])
                diff1=math.sqrt(diff1_temp[0]**2+diff1_temp[1]**2+diff1_temp[2]**2)
            else:
                diff1=float("inf")
            y_new=int(frame0[y,x,0]+u+y)
            x_new=int(frame0[y,x,1]+v+x)
            if(x_new >=0 and x_new < width and y_new >=0 and y_new<height):
                diff2_temp=np.abs(frame0[y,x]-frame1[y_new,x_new])
                diff2=math.sqrt(diff2_temp[0]**2+diff2_temp[1]**2+diff2_temp[2]**2)
            else:
                diff2=float("inf")
            if(diff1<diff2):
                iflow[yi,xi]=flow[my,mx]
                mapping[yi,xi] = my,mx,mv,mu
            else:
                iflow[yi,xi]=flow[y,x]
                mapping[yi,xi] = y,x,v,u

    return iflow,mapping


def interpflow(flow, frame0, frame1, t):
    '''
    Forward warping flow (from frame0 to frame1) to a position t in the middle of the 2 frames
    Follow the algorithm (1) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param flow: dense optical flow from frame0 to frame1
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param t: the intermiddite position in the middle of the 2 input frames
    :return: a warped flow
    '''
    iflow = None
    height,width,len = flow0.shape
    iflow = np.empty_like(flow)
    mapping = np.zeros([height,width,4])
    entry = np.zeros([height,width])
    for y in range(0,height):
        for x in range(0,width):
            x1,y1 = np.round(np.asarray([x,y]) + t*np.asarray([flow[y,x,0]+0.5,flow[y,x,1]])).astype(int)
            x2,y2 = np.round(np.asarray([x,y]) + t*np.asarray([flow[y,x,0]-0.5,flow[y,x,1]])).astype(int)
            x3,y3 = np.round(np.asarray([x,y]) + t*np.asarray([flow[y,x,0],flow[y,x,1]+0.5])).astype(int)
            x4,y4 = np.round(np.asarray([x,y]) + t*np.asarray([flow[y,x,0],flow[y,x,1]-0.5])).astype(int)
#            print(x1,y1,x2,y2,x3,y3,x4,y4)
            iflow,mapping = interpflowhelper(flow,frame0,frame1,iflow,mapping,entry,x1,y1,x,y,0.5,0)
            iflow,mapping = interpflowhelper(flow,frame0,frame1,iflow,mapping,entry,x2,y2,x,y,-0.5,0)
            iflow,mapping = interpflowhelper(flow,frame0,frame1,iflow,mapping,entry,x3,y3,x,y,0,0.5)
            iflow,mapping = interpflowhelper(flow,frame0,frame1,iflow,mapping,entry,x4,y4,x,y,0,-0.5)
#            if(x1 >=0 and x1 < width and y1 >=0 and y1<height):
#                if(iflow[y1,x1] is None):
#                    iflow[y1,x1]=flow[y,x]
#                    mapping[y1,x1] = y,x,0,0.5
#                else:
#                    my,mx,v,u=mapping[y1,x1]
#                    mx=(mx+v).astype(int)
#                    my=(my+u).astype(int)
#                    my_new=int(frame0[my,mx,0]+u+y)
#                    mx_new=int(frame0[my,mx,1]+x+v)
#                    if(mx_new >=0 and mx_new < width and my_new >=0 and my_new<height):
#                        diff1_temp=np.abs(frame0[my,mx]-frame1[my_new,mx_new])
#                        diff1=math.sqrt(diff1_temp[0]**2+diff1_temp[1]**2+diff1_temp[2]**2)
#                    else:
#                        diff=float("inf")
#                    y_new=int(frame0[y,x,0]+u+y)
#                    x_new=int(frame0[y,x,1]+x+v)
#                    if(x_new >=0 and x_new < width and y_new >=0 and y_new<height):
#                        diff2_temp=np.abs(frame0[y,x]-frame1[y_new,x_new])
#                        diff2=math.sqrt(diff2_temp[0]**2+diff2_temp[1]**2+diff2_temp[2]**2)
#                    else:
#                        diff=float("inf")
#                    if(diff1<diff2):
#                        iflow[y1,x1]=flow[my,mx]
#                        mapping[y1,x1] = my,mx,0,0.5
#                    else:
#                        iflow[y1,x1]=flow[y,x]
#                        mapping[y1,x1] = y,x,0,0.5
#            if(x2 >0 and x2 < width and y2 >0 and y2<height):
#                if(iflow[y2,x2] is None):
#                    iflow[y2,x2]=flow[y,x]
#                    mapping[y2,x2] = y,x,0,0.5
#                else:
#                    my,mx,v,u=mapping[y2,x2]
#                    mx=(mx+v).astype(int)
#                    my=(my+u).astype(int)
#            if(x3 >0 and x3 < width and y3 >0 and y3<height):
#                if(iflow[y3,x3] is None):
#                    iflow[y3,x3]=flow[y,x]
#                    mapping[y3,x3] = y,x,0,0.5
#                else:
#                    my,mx,v,u=mapping[y3,x3]
#                    mx=(mx+v).astype(int)
#                    my=(my+u).astype(int)
#            if(x4 >0 and x4 < width and y4 >0 and y4<height):
#                if(iflow[y4,x4] is None):
#                    iflow[y4,x4]=flow[y,x]
#                    mapping[y1,x1] = y,x,0,0.5
#                else:
#                    my,mx,v,u=mapping[y4,x4]
#                    mx=(mx+v).astype(int)
#                    my=(my+u).astype(int)
#
#    count1=0
#    for y in range(0,height):
#        for x in range(0,width):
#            if(entry[y,x]==1):
#                count1+=1
#    print((height*width)-count1)



    return iflow



def warpimages(iflow, frame0, frame1, occ0, occ1, t):
    '''
    Compute the colors of the interpolated pixels by inverse-warping frame 0 and frame 1 to the postion t based on the
    forwarded-warped flow iflow at t
    Follow the algorithm (4) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
     for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param iflow: forwarded-warped (from flow0) at position t
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param occ0: occlusion mask of frame 0
    :param occ1: occlusion mask of frame 1
    :param t: interpolated position t
    :return: interpolated image at position t in the middle of the 2 input frames
    '''

    iframe = np.zeros_like(frame0).astype(np.float32)
    height,width,_ = iflow.shape
    for y in range(0, height):
        for x in range(0,width):
            oc0=False
            oc1=False
            ut=np.asarray(iflow[y,x])
            x0,y0 = np.asarray([x,y]) - (t*ut).astype(int)
            x0,y0 = int(x0),int(y0)
            x1,y1 = np.asarray([x,y]) + ((1-t)*(ut)).astype(int)
            x1,y1 = int(x1),int(y1)
            
            if(x0 >0 and x0 < width and y0 >0 and y0<height):
                oc0 =True
            if(x1 >0 and x1 < width and y1 >0 and y1<height):
                oc1 =True
            if(oc0 and oc1):
                if(occ0[y0,x0] == 0 and occ1[y1,x1] == 0):
                    iframe[y,x] = (1-t)*frame0[y0,x0]+(t*frame1[y1,x1])
                else:
                    iframe[y,x] = frame0[y0,x0] if occ0[y0,x0] ==0 else frame1[y1,x1]
            else:
                if (oc0 and occ0[y0,x0] ==0):
                    iframe[y,x] = frame0[y0,x0]
                elif (oc1 and occ1[y1,x1] ==0):
                    iframe[y,x]=frame1[y1,x1]

    # to be completed ...
    return iframe

def blur(im):
    '''
    blur using a gaussian kernel [5,5] using opencv function: cv2.GaussianBlur, sigma=0
    :param im:
    :return updated im:
    '''
    im = cv2.GaussianBlur(im,(5,5),0)
    return im


def internp(frame0, frame1, t=0.5, flow0=None):
    '''
    :param frame0: beggining frame
    :param frame1: ending frame
    :return frame_t: an interpolated frame at time t
    '''
    print('==============================')
    print('===== interpolate an intermediate frame at t=',str(t))
    print('==============================')

    # ==================================================
    # ===== 1/ find the optical flow between the two given images: from frame0 to frame1,
    #  if there is no given flow0, run opencv function to extract it
    # ==================================================
    if flow0 is None:
        i1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        i2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        flow0 = cv2.calcOpticalFlowFarneback(i1, i2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # ==================================================
    # ===== 2/ find holes in the flow
    # ==================================================
    holes0 = find_holes(flow0)
    pickle.dump(holes0,open('holes0.step2.data','wb'))  # save your intermediate result
    # ====== score
    holes0       = pickle.load(open('holes0.step2.data','rb')) # load your intermediate result
    holes0_step2 = pickle.load(open('holes0.step2.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes0-holes0_step2))
    print('holes0_step2',diff)

    # ==================================================
    # ===== 3/ fill in any hole using an outside-in strategy
    # ==================================================
    flow0 = holefill(flow0,holes0)
    pickle.dump(flow0, open('flow0.step3.data', 'wb')) # save your intermediate result
    # ====== score
    flow0       = pickle.load(open('flow0.step3.data', 'rb')) # load your intermediate result
    flow0_step3 = pickle.load(open('flow0.step3.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow0-flow0_step3))
    print('flow0_step3',diff)

    # ==================================================
    # ===== 5/ estimate occlusion mask
    # ==================================================
    occ0, occ1 = occlusions(flow0,frame0,frame1)
    pickle.dump(occ0, open('occ0.step5.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step5.data', 'wb')) # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step5.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step5.data', 'rb')) # load your intermediate result
    occ0_step5  = pickle.load(open('occ0.step5.sample', 'rb')) # load sample result
    occ1_step5  = pickle.load(open('occ1.step5.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step5 - occ0))
    print('occ0_step5',diff)
    diff = np.sum(np.abs(occ1_step5 - occ1))
    print('occ1_step5',diff)

    # ==================================================
    # ===== step 6/ blur occlusion mask
    # ==================================================
    for iblur in range(0,BLUR_OCC):
        occ0 = blur(occ0)
        occ1 = blur(occ1)
    pickle.dump(occ0, open('occ0.step6.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step6.data', 'wb')) # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step6.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step6.data', 'rb')) # load your intermediate result
    occ0_step6  = pickle.load(open('occ0.step6.sample', 'rb')) # load sample result
    occ1_step6  = pickle.load(open('occ1.step6.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step6 - occ0))
    print('occ0_step6',diff)
    diff = np.sum(np.abs(occ1_step6 - occ1))
    print('occ1_step6',diff)

    # ==================================================
    # ===== step 7/ forward-warp the flow to time t to get flow_t
    # ==================================================
#flow_t = interpflow(flow0, frame0, frame1, t)
#    flow_t = np.asarray(flow_t)
#    pickle.dump(flow_t, open('flow_t.step7.data', 'wb')) # save your intermediate result
#    # ====== score
#    flow_t       = pickle.load(open('flow_t.step7.data', 'rb')) # load your intermediate result
#    flow_t_step7 = pickle.load(open('flow_t.step7.sample', 'rb')) # load sample result
#    diff = np.sum(np.abs(flow_t-flow_t_step7))
#    print('flow_t_step7',diff)

    # ==================================================
    # ===== step 8/ find holes in the estimated flow_t
    # ==================================================
#    holes1 = find_holes(flow_t)
##    print(flow_t.shape)
#    pickle.dump(holes1, open('holes1.step8.data', 'wb')) # save your intermediate result
##     ====== score
#    holes1       = pickle.load(open('holes1.step8.data','rb')) # load your intermediate result
#    holes1_step8 = pickle.load(open('holes1.step8.sample','rb')) # load sample result
#    diff = np.sum(np.abs(holes1-holes1_step8))
#    print('holes1_step8',diff)
#
#    # ===== fill in any hole in flow_t using an outside-in strategy
#    flow_t = holefill(flow_t, holes1)
#    pickle.dump(flow_t, open('flow_t.step8.data', 'wb')) # save your intermediate result
#    # ====== score
#    flow_t       = pickle.load(open('flow_t.step8.data', 'rb')) # load your intermediate result
    flow_t_step8 = pickle.load(open('flow_t.step8.sample', 'rb')) # load sample result
#    diff = np.sum(np.abs(flow_t-flow_t_step8))
#    print('flow_t_step8',diff)
#
#    # ==================================================
#    # ===== 9/ inverse-warp frame 0 and frame 1 to the target time t
#    # ==================================================
    frame_t = warpimages(flow_t_step8, frame0, frame1, occ0_step6, occ1_step6, t)
    pickle.dump(frame_t, open('frame_t.step9.data', 'wb')) # save your intermediate result
    # ====== score
    frame_t       = pickle.load(open('frame_t.step9.data', 'rb')) # load your intermediate result
    frame_t_step9 = pickle.load(open('frame_t.step9.sample', 'rb')) # load sample result
    diff = np.sqrt(np.mean(np.square(frame_t.astype(np.float32)-frame_t_step9.astype(np.float32))))
    print('frame_t',diff)
#
#   return frame_t


if __name__ == "__main__":

    print('==================================================')
    print('PSU CS 410/510, Winter 2019, HW3: video frame interpolation')
    print('==================================================')

    # ===================================
    # example:
    # python interp_skeleton.py frame0.png frame1.png flow0.flo frame05.png
    # ===================================
    path_file_image_0 = "/Users/kavinarasu/Desktop/hw3_code_skeleton/sample_data/frame0.png"
    path_file_image_1 = "/Users/kavinarasu/Desktop/hw3_code_skeleton/sample_data/frame1.png"
    path_file_flow    = "/Users/kavinarasu/Desktop/hw3_code_skeleton/sample_data/flow0.flo"
    path_file_image_result = "/Users/kavinarasu/result.png"

    # ===== read 2 input images and flow
    frame0 = cv2.imread(path_file_image_0)
    frame1 = cv2.imread(path_file_image_1)
    flow0  = readFlowFile(path_file_flow)
    #print(flow0.shape)

    # ===== interpolate an intermediate frame at t, t in [0,1]
    frame_t= internp(frame0=frame0, frame1=frame1, t=0.5, flow0=flow0)
#cv2.imwrite(filename=path_file_image_result, img=(frame_t * 1.0).clip(0.0, 255.0).astype(np.uint8))
