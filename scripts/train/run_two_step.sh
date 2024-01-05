export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

MODEL_TYPE=MBart
MODEL_NAME=facebook/mbart-large-50

DATA_DIR=data/${DATASET}
OUTPUT_DIR=output_dir/${MODEL_TYPE}-two-step/${DATASET}${SUF}
OUTPUT_DIR_MS=output_dir/${MODEL_TYPE}-ms/${DATASET}${SUF}


python3 -u run_seq2seq.py train \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME} \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --src_file ${DATA_DIR}/${SRC_FILE} \
    --tgt_file ${DATA_DIR}/${ML_TGT_FILE} \
    --eval_src_file ${DATA_DIR}/${EVAL_SRC_FILE} \
    --eval_tgt_file ${DATA_DIR}/${EVAL_ML_TGT_FILE} \
    --src_lang ${SRC_LANG_TOKEN} \
    --tgt_lang ${SRC_LANG_TOKEN} \
    --max_src_len 512 \
    --max_tgt_len 128 \
    --seed 42 \
    --output_dir ${OUTPUT_DIR_MS} \
    --gradient_accumulation_steps ${GRADIENT_ACCUM} \
    --lr ${LR} \
    --num_train_epochs ${EPOCHS} \
    --save_strategy epoch \
    --fp16


CKPT_NAME=$(ls ${OUTPUT_DIR_MS} | grep checkpoint-)
CKPT_STEP=${CKPT_NAME:11}
MODEL_RECOVER_PATH=${OUTPUT_DIR_MS}/checkpoint-${CKPT_STEP}/pytorch_model.bin


python3 -u run_seq2seq.py train \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME} \
    --model_recover_path ${MODEL_RECOVER_PATH} \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --src_file ${DATA_DIR}/${SRC_FILE} \
    --tgt_file ${DATA_DIR}/${TGT_FILE} \
    --eval_src_file ${DATA_DIR}/${EVAL_SRC_FILE} \
    --eval_tgt_file ${DATA_DIR}/${EVAL_TGT_FILE} \
    --src_lang ${SRC_LANG_TOKEN} \
    --tgt_lang ${TGT_LANG_TOKEN} \
    --max_src_len 512 \
    --max_tgt_len 128 \
    --seed 42 \
    --output_dir ${OUTPUT_DIR} \
    --gradient_accumulation_steps ${GRADIENT_ACCUM} \
    --lr ${LR} \
    --num_train_epochs ${EPOCHS} \
    --save_strategy epoch \
    --fp16
