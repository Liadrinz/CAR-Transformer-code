export DATASET=cross-sum
export SUF=/ar
export SRC_FILE=arabic-english_train.source
export TGT_FILE=arabic-english_train.target
export ML_TGT_FILE=arabic-english_train.target.mono
export EVAL_SRC_FILE=arabic-english_val.source
export EVAL_TGT_FILE=arabic-english_val.target
export EVAL_ML_TGT_FILE=arabic-english_val.target.mono
export TEST_SRC_FILE=arabic-english_test.source
export TEST_TGT_FILE=arabic-english_test.target
export SRC_LANG_TOKEN=ar_AR
export TGT_LANG_TOKEN=en_XX

export BATCH_SIZE=8
export EVAL_BATCH_SIZE=32
export GRADIENT_ACCUM=2
export LR=0.0000002
export PG_WEIGHT=1.0
export EPOCHS=30

TASKS=(mbart car_transformer cls_ms two_step)

for TASK in ${TASKS[@]}
do

bash ./scripts/train/run_${TASK}.sh
bash ./scripts/decode/run_${TASK}.sh
rm output_dir/*/*/*/*/*.pt

done
