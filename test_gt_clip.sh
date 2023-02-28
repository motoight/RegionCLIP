python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/clip/teacher_RN50_student_RN50_OAI_CLIP.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
MODEL.CLIP.CROP_REGION_TYPE GT \
OUTPUT_DIR /data/ovr/test_clip_gt_debug


python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.CLIP.MULTIPLY_RPN_SCORE False \
OUTPUT_DIR ./output_reclip_gt