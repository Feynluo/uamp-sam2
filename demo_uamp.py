
import cv2
import torch
import numpy as np
from uamp.uamp_predictor import UAMPSamuraiPredictor
from sam2.build_sam import build_sam2
from sam2.sam2_model_registry import sam2_model_registry

# -------------------------- configure parameters --------------------------
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"  
SAM2_MODEL_TYPE = "sam2_hiera_l"
VIDEO_PATH = "your_video.mp4"  
OUTPUT_VIDEO_PATH = "uamp_output.mp4"  
# -------------------------------------------------------------

# 1. load model
sam2_model = sam2_model_registry[SAM2_MODEL_TYPE](checkpoint=SAM2_CHECKPOINT)
predictor = UAMPSamuraiPredictor(sam2_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam2_model.to(device)

# 2. read video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))

# 3. initial prompt
ret, first_frame = cap.read()
if not ret:
    raise ValueError("video read failed!")
predictor.set_image(first_frame)

input_points = np.array([[w//2, h//2], [w//4, h//4]])  # 前景点+背景点
input_labels = np.array([1, 0])
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True
)
# choose the best mask based on score
best_idx = np.argmax(scores)
predictor.initialize_video(mask=logits[best_idx])

# 4. UAMP inference
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # UAMP predict
    masks, scores, logits = predictor.predict_video_frame(frame)
    # visualize mask
    vis_frame = frame.copy()
    mask = (masks > 0.5).astype(np.uint8)
    vis_frame[mask > 0] = (0, 255, 0)
    vis_frame = cv2.addWeighted(vis_frame, 0.5, frame, 0.5, 0)
    # save result
    out.write(vis_frame)
    cv2.imshow("UAMP Video Segmentation/Tracking", vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. release resource
cap.release()
out.release()
cv2.destroyAllWindows()
predictor.reset_video()
