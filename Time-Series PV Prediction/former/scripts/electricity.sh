export CUDA_VISIBLE_DEVICES=2

seq_len=96
model_name=CSFformer
model_id_name=Electricity

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --mspi_layers 3 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --itr 1
done





