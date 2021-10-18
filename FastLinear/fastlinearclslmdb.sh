# params
# task, pretrained_path, memorybank_path  

TYPE=$1
OUTPUT=$2
PRETRAIN=$3
MM=$4
BACKBONE=$5
cd FastLinear/

if [[ ${TYPE} =~ mm ]]; then
    if [[ ${BACKBONE} =~ vit_s ]]; then
        python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=$((RANDOM + 10000)) generate_memory_bank.py --task dino --pretrained_path ${PRETRAIN} --save_path ${MM} --feat_dim 384 --batch_size 128 --feature_layer 'lastconv' --use-lmdb --data_path /opt/tiger/wf/datasets/imagenet/ --backbone ${BACKBONE};
    elif [[ ${BACKBONE} =~ vit_b ]]; then
        python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=$((RANDOM + 10000)) generate_memory_bank.py --task dino --pretrained_path ${PRETRAIN} --save_path ${MM} --feat_dim 768 --batch_size 128 --feature_layer 'lastconv' --use-lmdb --data_path /opt/tiger/wf/datasets/imagenet/ --backbone ${BACKBONE};
    fi

fi

if [[ ${TYPE} =~ nlc ]]; then
    echo python fastlinearcls.py --save_path './fixlin_exp/' --memorybank_path ${MM} --batch_size 512 --lr 10 --normalize 1 --output_dir ${OUTPUT};
    python3 fastlinearcls.py --save_path './.fixlin_exp/' --memorybank_path ${MM} --batch_size 512 --lr 10 --normalize 1 --output_dir ${OUTPUT};
fi

# if [[ ${TYPE} =~ lc ]]; then
#     echo python fastlinearcls.py --save_path './fixlin_exp/' --memorybank_path ${MM} --batch_size 512 --lr 1.0 --normalize 0 --output_dir ${OUTPUT};
#     python3 fastlinearcls.py --save_path './.fixlin_exp/' --memorybank_path ${MM} --batch_size 512 --lr 1.0 --normalize 0 --output_dir ${OUTPUT};
# fi

#
cd ../
