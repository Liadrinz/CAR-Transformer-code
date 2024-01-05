export DATASET=WikiLingua_data_splits/turkish
export SRC_FILE=train.src.tr
export TGT_FILE=train.tgt.en
export ML_SRC_FILE=train.src.tr
export ML_TGT_FILE=train.tgt.tr
export EVAL_SRC_FILE=val.src.tr
export EVAL_TGT_FILE=val.tgt.en
export EVAL_ML_TGT_FILE=val.tgt.tr
export TEST_SRC_FILE=test.src.tr
export TEST_TGT_FILE=test.tgt.en
export SRC_LANG_TOKEN=tr_TR
export TGT_LANG_TOKEN=en_XX

export BATCH_SIZE=8
export EVAL_BATCH_SIZE=32
export GRADIENT_ACCUM=2
export LR=0.00001
export PG_WEIGHT=0.5
export EPOCHS=30

TASKS=(mbart car_transformer cls_ms cls_mt two_step)

for TASK in ${TASKS[@]}
do

bash ./scripts/train/run_${TASK}.sh
bash ./scripts/decode/run_${TASK}.sh
rm output_dir/*/*/*/*/*.pt

done
