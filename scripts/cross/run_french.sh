export DATASET=cross-sum
export SUF=/fr
export SRC_FILE=french-english_train.source
export TGT_FILE=french-english_train.target
export ML_TGT_FILE=french-english_train.target.mono
export EVAL_SRC_FILE=french-english_val.source
export EVAL_TGT_FILE=french-english_val.target
export EVAL_ML_TGT_FILE=french-english_val.target.mono
export TEST_SRC_FILE=french-english_test.source
export TEST_TGT_FILE=french-english_test.target
export SRC_LANG_TOKEN=fr_XX
export TGT_LANG_TOKEN=en_XX

export BATCH_SIZE=8
export EVAL_BATCH_SIZE=32
export GRADIENT_ACCUM=2
export LR=0.000001
export PG_WEIGHT=1.0
export EPOCHS=30

TASKS=(mbart car_transformer cls_ms two_step)

for TASK in ${TASKS[@]}
do

bash ./scripts/train/run_${TASK}.sh
bash ./scripts/decode/run_${TASK}.sh
rm output_dir/*/*/*/*/*.pt

done
