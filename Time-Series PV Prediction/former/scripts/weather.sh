export CUDA_VISIBLE_DEVICES=3

seq_len=96
model_name=CSFformer
model_id_name=Weather

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --mspi_layers 3 \
    --itr 1
done

