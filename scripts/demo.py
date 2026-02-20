import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda")
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path)

    frame_rate = 30
    if args.save_to_video:
        if osp.isdir(args.video_path):
            frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(args.video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]

            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        # img_w, img_h = 854, 480
        image = loaded_frames[1]
        img_h, img_w = image.shape[:2]
        # img_w, img_h = 1280, 720
        x = np.linspace(0, img_w, 600)
        y = np.linspace(0, img_h, 600)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        z_ret = None
        f_arr, iou_arr, kf_arr = [], [],[]

        for frame_idx, object_ids, masks, mean_val, conv_val, iou, kf_iou, features, heat_feat in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            if args.save_to_video:
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

                out.write(img)
            
            if frame_idx in [303,362,403]: #[1,23,33,34,54,94,116,126,152,185]:
                plt.subplots(nrows=1, ncols=1, figsize=(img_w*0.02, img_h*0.02))
                plt.axis('off')
                scaled_training_data = StandardScaler().fit_transform(features)
                n_components = 3
                pca = PCA(n_components = n_components)
                # Apply PCA [0]
                reduced_data = pca.fit_transform(scaled_training_data)
                rgb_img = reduced_data.reshape(256, 256, 3)
                heat_map = cv2.resize(rgb_img, (img_w,img_h))
                normed_mask = heat_map/heat_map.max() * 255
                img2 = normed_mask.astype(np.uint8)
                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img2, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)
                plt.imshow(img2, alpha=0.5, interpolation='nearest', cmap='jet')
                plt.tight_layout()
                plt.savefig(f"./result/{frame_idx}-feat.png", dpi=200)
                plt.close()
                # cv2.imwrite(f"./result2/{frame_idx}-feat.png", img2)

                plt.subplots(nrows=1, ncols=1, figsize=(img_w*0.02, img_h*0.02))
                plt.axis('off')
                # attention heatmap.
                heat_map = cv2.resize(heat_feat, (img_w,img_h))
                normed_mask = heat_map/heat_map.max()* 255
                img2 = normed_mask.astype(np.uint8)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img2, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)
                plt.imshow(img2, alpha=0.5, interpolation='nearest', cmap='jet')
                plt.tight_layout()
                plt.savefig(f"./result/{frame_idx}-att.png", dpi=200)
                plt.close()
                # cv2.imwrite(f"./result2/{frame_idx}-att.png", img2)
            # gaussian distribution pdf plot.
            if mean_val is not None:
                gaussian = multivariate_normal(mean=mean_val, cov=conv_val)
                Z = gaussian.pdf(pos)
                if z_ret is None:
                    z_ret = Z
                else:
                    z_ret += Z

                # plt.subplots(nrows=1, ncols=1, figsize=(img_w*0.02, img_h*0.02))
                # plt.axis('off')
                # plt.pcolormesh(X, Y, Z)
                # plt.colorbar()
                # plt.savefig(f"{frame_idx}-proba.png", dpi=300)
                # plt.show()
            f_arr.append(frame_idx)
            iou_arr.append(iou)
            kf_arr.append(kf_iou)
        
        if z_ret is not None:
            plt.subplots(nrows=1, ncols=1, figsize=(img_w*0.02, img_h*0.02))
            plt.axis('off')
            # accumulative guassian pdf.
            plt.pcolormesh(X, Y, z_ret)
            plt.tight_layout()
            plt.colorbar()
            plt.savefig("./result/g_pdf.png", dpi=300)
            plt.close()
        
        import pandas as pd
        df = pd.DataFrame()
        df['idx'] = f_arr
        df['iou'] = iou_arr
        df['kf_iou'] = kf_arr
        df.to_csv("./result/iou.csv")

        if args.save_to_video:
            out.release()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="/home/kemove/Downloads/videos/lvos1/KPDIQo5u/",help="Input video path or directory of frames.")
    # parser.add_argument("--video_path", default="video/KPDIQo5u/",help="Input video path or directory of frames.")
    # parser.add_argument("--video_path", default="video/davis_sunset/",help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", default="./bbox/bbox2.txt", help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    args = parser.parse_args()
    main(args)
