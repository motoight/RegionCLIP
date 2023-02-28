python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/clip/teacher_RN50_student_RN50_OAI_CLIP.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
MODEL.CLIP.CROP_REGION_TYPE RPN \
MODEL.CLIP.MULTIPLY_RPN_SCORE True \
MODEL.CLIP.RPN_ALPHA 0.1 \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
OUTPUT_DIR ./output_rpn_alpha/1


python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/clip/teacher_RN50_student_RN50_OAI_CLIP.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
MODEL.CLIP.CROP_REGION_TYPE RPN \
MODEL.CLIP.MULTIPLY_RPN_SCORE True \
MODEL.CLIP.RPN_ALPHA 0.2 \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
OUTPUT_DIR ./output_rpn_alpha/2


python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/clip/teacher_RN50_student_RN50_OAI_CLIP.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
MODEL.CLIP.CROP_REGION_TYPE RPN \
MODEL.CLIP.MULTIPLY_RPN_SCORE True \
MODEL.CLIP.RPN_ALPHA 0.3 \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
OUTPUT_DIR ./output_rpn_alpha/3

python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/clip/teacher_RN50_student_RN50_OAI_CLIP.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
MODEL.CLIP.CROP_REGION_TYPE RPN \
MODEL.CLIP.MULTIPLY_RPN_SCORE True \
MODEL.CLIP.RPN_ALPHA 0.4 \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
OUTPUT_DIR ./output_rpn_alpha/4

python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/clip/teacher_RN50_student_RN50_OAI_CLIP.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
MODEL.CLIP.CROP_REGION_TYPE RPN \
MODEL.CLIP.MULTIPLY_RPN_SCORE True \
MODEL.CLIP.RPN_ALPHA 0.5 \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
OUTPUT_DIR ./output_rpn_alpha/5