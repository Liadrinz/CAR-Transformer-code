export WANDB_DISABLED=true
export OMP_NUM_THREADS=1

MODEL_TYPE=MBart
MODEL_NAME=facebook/mbart-large-50

DATA_DIR=data/${DATASET}
OUTPUT_DIR=output_dir/${MODEL_TYPE}/${DATASET}${SUF}
CKPT_NAME=$(ls ${OUTPUT_DIR} | grep checkpoint-)
CKPT_STEP=${CKPT_NAME:11}
MODEL_RECOVER_PATH=${OUTPUT_DIR}/checkpoint-${CKPT_STEP}/pytorch_model.bin

python3 -u run_seq2seq.py decode \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME} \
    --model_recover_path ${MODEL_RECOVER_PATH} \
    --batch_size 64 \
    --src_file ${DATA_DIR}/${TEST_SRC_FILE} \
    --src_lang ${SRC_LANG_TOKEN} \
    --tgt_lang ${TGT_LANG_TOKEN} \
    --max_src_len 512 \
    --max_tgt_len 128 \
    --num_beams 1 \
    --seed 42 \
    --fp16

files2rouge ${DATA_DIR}/${TEST_TGT_FILE} ${MODEL_RECOVER_PATH}.decode.txt -a "-c 95 -r 1000 -n 2 -a"
