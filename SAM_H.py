!pip install git+https://github.com/facebookresearch/segment-anything.git -q
!pip install torch torchvision matplotlib opencv-python tqdm -q

import os, cv2, gc, torch, time, glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from google.colab import files
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

!mkdir -p images output_masks
uploaded = files.upload()
for fn in uploaded.keys():
    os.rename(fn, f"images/{fn}")


sam_checkpoint = "sam_vit_h_4b8939.pth"
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
if not os.path.exists(sam_checkpoint):
    !wget -q {url}


device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.8,
    min_mask_region_area=50,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    output_mode="binary_mask"
)

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    sharpen = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    edges = cv2.Canny(sharpen, 40, 120)
    edges_inv = cv2.bitwise_not(edges)
    return cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB)

def show_combined_mask(image, masks):
    combined = np.zeros_like(image)
    for m in masks:
        mask = m["segmentation"]
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        combined[mask] = color
    return cv2.addWeighted(image, 0.6, combined, 0.4, 0), combined

MAX_MASKS = 254
for img_name in tqdm(os.listdir("images")):
    torch.cuda.empty_cache(); gc.collect()
    img_path = f"images/{img_name}"
    base_name = os.path.splitext(img_name)[0]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enhanced = enhance_image(img)

    print(f"\n Processing {img_name} using SAM-HUGE...")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(device)/1024**2
    start_time = time.time()

    masks = mask_generator.generate(enhanced)

    torch.cuda.synchronize()
    end_time = time.time()
    mem_after = torch.cuda.memory_allocated(device)/1024**2
    mem_peak = torch.cuda.max_memory_allocated(device)/1024**2

    print(f"GPU RAM: Before {mem_before:.2f}MB | After {mem_after:.2f}MB | Peak {mem_peak:.2f}MB")
    print(f"Processing Time: {end_time - start_time:.3f}s")
    print(f"Total Masks Generated: {len(masks)}")

    binary_masks = np.stack([m["segmentation"] for m in masks])
    if binary_masks.shape[0] > MAX_MASKS:
        print(f" Trimming {binary_masks.shape[0]} → {MAX_MASKS} masks for UI compatibility.")
        binary_masks = binary_masks[:MAX_MASKS]

    npy_path = f"output_masks/{base_name}_mask.npy"
    np.save(npy_path, binary_masks)
    print(f" Saved mask: {npy_path} (shape: {binary_masks.shape})")

    overlay, _ = show_combined_mask(img, masks[:MAX_MASKS])
    cv2.imwrite(f"output_masks/{base_name}_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1); plt.imshow(img); plt.title("Original")
    plt.subplot(1,2,2); plt.imshow(overlay); plt.title("Overlay Mask")
    plt.suptitle(f"SAM-H Segmentation — {img_name}")
    plt.show()

!zip -r output_masks_HUGE.zip output_masks
files.download("output_masks_HUGE.zip")
