export DATASET=globalvoices_v0_proc/fr
export SRC_FILE=train.src
export TGT_FILE=train.tgt
export ML_TGT_FILE=train.tgt.mono
export EVAL_SRC_FILE=valid.src
export EVAL_TGT_FILE=valid.tgt
export EVAL_ML_TGT_FILE=valid.tgt.mono
export TEST_SRC_FILE=test.src
export TEST_TGT_FILE=test.tgt
export SRC_LANG_TOKEN=fr_XX
export TGT_LANG_TOKEN=en_XX

export BATCH_SIZE=8
export EVAL_BATCH_SIZE=32
export GRADIENT_ACCUM=2
export LR=0.000002
export PG_WEIGHT=1.0
export EPOCHS=30

TASKS=(mbart car_transformer cls_ms two_step)

for TASK in ${TASKS[@]}
do

bash ./scripts/train/run_${TASK}.sh
bash ./scripts/decode/run_${TASK}.sh
rm output_dir/*/*/*/*/*.pt

done
