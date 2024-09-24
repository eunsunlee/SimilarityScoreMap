import numpy as np
import torch
import cv2
# from env.cached_env import CachedEnv
# from models.places365 import get_res_feats, load_places_resnet
# import quaternion
# import matplotlib.pyplot as plt
from realtime.models.superpoint import SuperPoint
from realtime.models.superglue import SuperGlue
from realtime.models.matching import Matching
from numpy.linalg import inv, norm
import matplotlib.cm as cm 


def make_matching_plot_fast_rgb(gray0, gray1, kpts0, kpts1, pred, pred0, pred1): 

    margin = 10

    H0, W0,_ = gray0.shape
    H1, W1, _ = gray1.shape
    H, W = max(H0, H1), W0 + W1 + margin
    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0] = gray0

    out[:H1, W0+margin:] = gray1
    # out = np.stack([out]*3, -1)
    # kpts0 = np.round(data['keypoints0'][0].cpu().numpy()).astype(int)
    # kpts1 = np.round(data['keypoints1'][0].cpu().numpy()).astype(int)

    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)

    kpts0_ori = np.round(pred0['original_keypoints'][0].cpu().numpy()).astype(int)
    kpts1_ori = np.round(pred1['original_keypoints'][0].cpu().numpy()).astype(int)

    for x, y in kpts0_ori:
        cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
    for x, y in kpts1_ori:
        cv2.circle(out, (x + margin + W0, y), 2, black, -1,  lineType=cv2.LINE_AA)
        cv2.circle(out, (x + margin + W0, y), 1, white, -1, lineType=cv2.LINE_AA)

    for x, y in kpts0:
        cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 1, red, -1, lineType=cv2.LINE_AA)
    for x, y in kpts1:
        cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                    lineType=cv2.LINE_AA)
        cv2.circle(out, (x + margin + W0, y), 1, red, -1,
                    lineType=cv2.LINE_AA)

    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].detach().cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    
    color = cm.jet(confidence[valid])

    conf = confidence[valid]


    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    # mkpts0, mkpts1 = np.round(pred['matches0'][0].cpu().numpy()).astype(int), np.round(pred['matches1'][0].cpu().numpy()).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]

    # import pdb; pdb.set_trace()
    for (x0, y0), (x1, y1), c, confidence in zip(mkpts0, mkpts1, color, conf):
        # import pdb; pdb.set_trace()
        c = c.tolist()

        cv2.line(out, (x0, y0), (x1 + margin + W0, y1), color=c, thickness=2, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    return out

def make_matching_plot_fast(gray0, gray1, kpts0, kpts1, pred): 
    margin = 10


    H0, W0 = gray0.shape
    H1, W1 = gray1.shape
    H, W = max(H0, H1), W0 + W1 + margin
    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = gray0
    out[:H1, W0+margin:] = gray1
    out = np.stack([out]*3, -1)
    # kpts0 = np.round(data['keypoints0'][0].cpu().numpy()).astype(int)
    # kpts1 = np.round(data['keypoints1'][0].cpu().numpy()).astype(int)

    white = (255, 255, 255)
    black = (0, 0, 0)

    for x, y in kpts0:
        cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
    for x, y in kpts1:
        cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                    lineType=cv2.LINE_AA)
        cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                    lineType=cv2.LINE_AA)


    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].detach().cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    
    color = cm.jet(confidence[valid])

    conf = confidence[valid]


    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    # mkpts0, mkpts1 = np.round(pred['matches0'][0].cpu().numpy()).astype(int), np.round(pred['matches1'][0].cpu().numpy()).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]

    # import pdb; pdb.set_trace()
    for (x0, y0), (x1, y1), c, confidence in zip(mkpts0, mkpts1, color, conf):
        # import pdb; pdb.set_trace()
        c = c.tolist()

        cv2.line(out, (x0, y0), (x1 + margin + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    return out

class ObjectSuperGlue(object):
    def __init__(self, device):

        self.nodes = set()
        self.node_by_id = {}
        self.edges = set()

        self.device = device
        # self.resnet = load_places_resnet(device)

        # superpoint configs 
        # config = {
        #             'nms_radius': 4,
        #             'keypoint_threshold': 0.010,
        #             'max_keypoints': 20 # -1 
        #         }

        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.80,
            }
        }

        self.superpoint = SuperPoint(config.get('superpoint', {})).to(device)
        self.superglue = SuperGlue(config.get('superglue', {})).to(device)

    def superpoint_result(self, rgb, bbox):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        frame_tensor = torch.from_numpy(gray/255.).float()[None, None].to(self.device)
        pred = self.superpoint({'image': frame_tensor, 'bbox': bbox})
        return frame_tensor, pred

    # def matching_pairs(self, rgb0, rgb1, bbox0, bbox1):
    def matching_pairs(self, rgb0, rgb1, frame_tensor0, pred0, frame_tensor1, pred1):

        # frame_tensor0, pred0 = self.superpoint_result(rgb0, bbox0)
        # frame_tensor1, pred1 = self.superpoint_result(rgb1, bbox1)

        # gray0 = cv2.cvtColor( rgb0 ,cv2.COLOR_BGR2GRAY)
        # frame_tensor0 = torch.from_numpy(gray0/255.).float()[ None, None].to(self.device)
        # pred0 = self.superpoint({'image': frame_tensor0, 'bbox': bbox0})  

        # gray1 = cv2.cvtColor( rgb1 ,cv2.COLOR_BGR2GRAY)
        # frame_tensor1 = torch.from_numpy(gray1/255.).float()[ None, None].to(self.device)
        # pred1 = self.superpoint({'image': frame_tensor1, 'bbox': bbox1})  

        data = {}   
        data['image0'] = frame_tensor0       
        data['image1'] = frame_tensor1
        data['keypoints0'] = pred0['keypoints'][0].unsqueeze(0) # torch.size([1,50,2])
        data['scores0'] = pred0['scores'][0].unsqueeze(0) # torch.size([1,50])
        data['descriptors0'] = pred0['descriptors'][0].unsqueeze(0) # [1,256, 50]
        data['keypoints1'] = pred1['keypoints'][0].unsqueeze(0)
        data['scores1'] = pred1['scores'][0].unsqueeze(0)
        data['descriptors1'] = pred1['descriptors'][0].unsqueeze(0)

        try:        
            pred = self.superglue(data)
        except: 
            return False, None, None
        # try:
        #     pred = self.superglue(data)
        # except: 
        #     # import pdb; pdb.set_trace()
        #     return False, None, None, None, None, None, None, None, None 

        kpts0 = np.round(data['keypoints0'][0].cpu().numpy()).astype(int)
        kpts1 = np.round(data['keypoints1'][0].cpu().numpy()).astype(int)

        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].detach().cpu().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = confidence[valid] 

        match_result = {} 
        match_result['matches'] = matches
        match_result['mconf'] = mconf
        match_result['mkpts0'] = mkpts0
        match_result['mkpts1'] = mkpts1 

        # gray_vis =  make_matching_plot_fast(gray0, gray1, kpts0, kpts1, pred)
        rgb_vis = make_matching_plot_fast_rgb(rgb0, rgb1, kpts0, kpts1, pred, pred0, pred1 )

        return True, match_result, rgb_vis