# this scipt is to get results when proposal is rpn/gt 
# under zs and ft setting

# R50, RPN, LVIS, zs
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
# MODEL.CLIP.CROP_REGION_TYPE RPN \
# MODEL.CLIP.MULTIPLY_RPN_SCORE True \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
# OUTPUT_DIR /data/ovr/test_lvis_gt2rpn/zs_rpn

# RN50, GT, LVIS, zs
# python3 ./tools/train_net.py \
# --eval-only  \
# --num-gpus 1 \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
# MODEL.CLIP.CROP_REGION_TYPE GT \
# MODEL.CLIP.MULTIPLY_RPN_SCORE False \
# OUTPUT_DIR /data/ovr/test_lvis_gt2rpn/zs_gt

# RN50, RPN, LVIS, ft 
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-lvis_rn50.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
MODEL.CLIP.MULTIPLY_RPN_SCORE False \
OUTPUT_DIR /data/ovr/test_lvis_gt2rpn/ft_rpn

# RN50, GT, LVIS, ft 
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-lvis_rn50.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.CLIP.MULTIPLY_RPN_SCORE False \
OUTPUT_DIR /data/ovr/test_lvis_gt2rpn/ft_gt