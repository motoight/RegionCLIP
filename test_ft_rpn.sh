# RN50, COCO (Generalized: Novel + Base)
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_48_base_cls_emb.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
# MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
# OUTPUT_DIR /data/ovr/test_ovod_rpn/coco_ori


python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_48_base_cls_emb.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
MODEL.CLIP.NO_BOX_DELTA True \
OUTPUT_DIR /data/ovr/test_ovod_rpn/coco_ori_nodelta

# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
# MODEL.CLIP.CROP_REGION_TYPE RPN \
# MODEL.CLIP.MULTIPLY_RPN_SCORE True \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# OUTPUT_DIR /data/ovr/test_ovod_rpn/coco_wrpns



# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50.pth \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
# MODEL.CLIP.CROP_REGION_TYPE GT \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
# OUTPUT_DIR /data/ovr/test_ovod_rpn/coco_gt




python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
MODEL.CLIP.CROP_REGION_TYPE RPN \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
OUTPUT_DIR /data/ovr/test_ovod_rpn/coco_lvis866_worpns

python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
MODEL.CLIP.CROP_REGION_TYPE RPN \
MODEL.CLIP.MULTIPLY_RPN_SCORE True \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
OUTPUT_DIR /data/ovr/test_ovod_rpn/coco_lvis866_wrpns


# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/clip/teacher_RN50_student_RN50_OAI_CLIP.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
# MODEL.CLIP.CROP_REGION_TYPE GT \
# MODEL.CLIP.RPN_ALPHA -1 \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
# OUTPUT_DIR /data/ovr/test_ovod_rpn/coco_gt