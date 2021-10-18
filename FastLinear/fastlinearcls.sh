# params
# task, pretrained_path, memorybank_path  

cd FastLinear/
echo python -m torch.distributed.launch --nproc_per_node=4 generate_memory_bank.py --task $1 --pretrained_path $2 --save_path $3 --feat_dim 2048 --batch_size 32 --feature_layer 'lastconv';
python3 -m torch.distributed.launch --nproc_per_node=4 generate_memory_bank.py --task $1 --pretrained_path $2 --save_path $3 --feat_dim 2048 --batch_size 32 --feature_layer 'lastconv';

echo python fastlinearcls.py --save_path './fixlin_exp/' --memorybank_path $3 --batch_size 512 --lr 10 --normalize 1;
python3 fastlinearcls.py --save_path './.fixlin_exp/' --memorybank_path $3 --batch_size 512 --lr 10 --normalize 1;
echo python fastlinearcls.py --save_path './fixlin_exp/' --memorybank_path $3 --batch_size 512 --lr 1.0 --normalize 0;
python3 fastlinearcls.py --save_path './.fixlin_exp/' --memorybank_path $3 --batch_size 512 --lr 1.0 --normalize 0;
#
cd ../
