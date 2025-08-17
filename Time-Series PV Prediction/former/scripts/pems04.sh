export CUDA_VISIBLE_DEVICES=0

seq_len=96
model_name=CSFformer
model_id_name=PEMS04

for pred_len in 12 24 48 96
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id $model_id_name'_'seq_len'_'$pred_len \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 4 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 1024 \
    --learning_rate 0.0005 \
    --mspi_layers 3 \
    --use_norm 0 \
    --itr 1
done
