export CUDA_VISIBLE_DEVICES=2

seq_len=96
model_name=CSFformer
model_id_name=PEMS08

for pred_len in 12 24 48 96
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id $model_id_name'_'seq_len'_'$pred_len \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --learning_rate 0.001 \
    --mspi_layers 3 \
    --itr 1 \
    --use_norm 1
done