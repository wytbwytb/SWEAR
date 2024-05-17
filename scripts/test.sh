sh ./tools/dist_test.sh  \
  configs/oriented_rcnn/oriented_rcnn_r50_fpn_3x_roxray_p_le90.py \
  /media/datasets/gpu17_models/mmrotate/oriented_rcnn/oriented_rcnn_r50_fpn_3x_roxray_p_le90/epoch_36.pth 1 \
  --eval mAP

# for epoch in {25..40}
# do
#     echo ${epoch}
#     sh ./tools/dist_test.sh  \
#       configs/ours/ours_r50_fpn_3x_roxray_p_le90.py \
#       /media/datasets/gpu17_models/mmrotate/ours/ours_r50_fpn_3x_roxray_p_le90/epoch_${epoch}.pth 1 \
#       --eval mAP
# done

# sh ./tools/dist_test.sh  \
#   configs/ours/ours_r50_fpn_3x_roxray_p_le90.py \
#   /media/datasets/gpu17_models/mmrotate/ours/ours_r50_fpn_3x_roxray_p_le90/epoch_36.pth 1 \
#   --eval mAP
  # --show-dir /media/datasets/gpu17_models/mmrotate/oriented_rcnn/oriented_rcnn_r50_fpn_1x_roxray_le90/vis
  # --format-only \
  # --eval-options submission_dir=/media/datasets/gpu17_models/mmrotate/oriented_rcnn/oriented_rcnn_r50_fpn_1x_roxray_le90/task1


