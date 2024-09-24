import numpy as np
from multi.object_superglue import ObjectSuperGlue
import cv2 
import matplotlib.cm as cm 
from multi.calc_matched_feats import CALC_FEAT_MATCH


def make_matching_plot_fast_rgb(gray0, gray1, pred): 
    margin = 10

    H0, W0,_ = gray0.shape
    H1, W1, _ = gray1.shape
    H, W = max(H0, H1), W0 + W1 + margin
    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0] = gray0
    out[:H1, W0+margin:] = gray1

    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)

    kpts0 = np.round(pred['mkpts0']).astype(int)
    kpts1 = np.round(pred['mkpts1']).astype(int)
    
    for x, y in kpts0:
        cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 1, red, -1, lineType=cv2.LINE_AA)
    for x, y in kpts1:
        cv2.circle(out, (x + margin + W0, y), 2, black, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x + margin + W0, y), 1, red, -1, lineType=cv2.LINE_AA)


    # matches = pred['matches']
    confidence = pred['mconf']

    color = cm.jet(confidence)
    conf = confidence
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c, confidence in zip(mkpts0, mkpts1, color, conf):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1), color=c, thickness=2, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)

    # # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    return out

class Landmarks(object):

    def __init__(self, args):
        self.objectglue = ObjectSuperGlue(args.device)
        self.feat_match_calc = CALC_FEAT_MATCH(args)

    def find_landmark(self, a_node, b_node): 

        landmark_added = 0 
        a_rgb = a_node.rgb 
        b_rgb = b_node.rgb 
        a_depth = a_node.depth
        b_depth = b_node.depth
        a_local_pose = a_node.pose
        b_local_pose = b_node.pose

        rgb_vis = np.ones((512, 512+512, 3)).astype(int)*255

        rgb_vis[:512, :512,:] = a_rgb
        rgb_vis[:512, 512:512+512, :] = b_rgb

        a_frame_tensor, a_pred = self.objectglue.superpoint_result(a_rgb, [0,0,0,0])
        b_frame_tensor, b_pred = self.objectglue.superpoint_result(b_rgb, [0,0,0,0])

        rgb_ret, rgb_match_result, rgb_vis = self.objectglue.matching_pairs(a_rgb, b_rgb, \
                a_frame_tensor, a_pred, b_frame_tensor, b_pred)

        # top_k_conf = 15
        top_k_conf =20
        if rgb_ret and rgb_match_result['mkpts0'].shape[0] > top_k_conf: 
            top_indices = [index for index, value in sorted(enumerate(rgb_match_result['mconf']), key=lambda x: x[1], reverse=True)[:top_k_conf]]
            top_conf_mask = [k in top_indices for k in range(len(rgb_match_result['mconf']))]
        
            new_rgb_match_result= {'mconf': rgb_match_result['mconf'][top_conf_mask],
                                    'mkpts0': rgb_match_result['mkpts0'][top_conf_mask],
                                    'mkpts1' : rgb_match_result['mkpts1'][top_conf_mask],
                                    'mconf' : rgb_match_result['mconf'][top_conf_mask]}

            out = make_matching_plot_fast_rgb(a_rgb, b_rgb, new_rgb_match_result)
            check, kptx0, kpty0, kpt_angle_deg0, kpt_dist0, kptx1, kpty1, kpt_angle_deg1, \
                kpt_dist1 = self.feat_match_calc( a_depth,  a_local_pose, \
                b_depth,  b_local_pose, new_rgb_match_result)
            
            if check and kpt_dist0 > 1 and kpt_dist1 > 1:  
                # plt.imsave("rgb_vis/rgb_vis_{}.png".format(j), out)
                print("landmark added ")
                landmark_added = 1 
                landmark_attr = (kptx0, kpty0, kpt_angle_deg0, kpt_dist0, kpt_angle_deg1, kpt_dist1)
                return landmark_added, landmark_attr
        return landmark_added, None