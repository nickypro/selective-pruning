# Task bash files to be called by the overall bash file for all experiments of the specified model type.
# We provide all of the individual bash files per task and model to make individual experiments easy to call.

reset_cuda(){
    sleep 10
}

DEVICE=0
seed=123
#############################################################
################ CIFAR100 ROCKET FORGETTING #################
#############################################################
#declare -a StringArray=("rocket" "mushroom" "baby" "lamp" "sea") # classes to iterate over
declare -a StringArray=("rocket" "mushroom") # classes to iterate over
# declare -a StringArray=("mushroom") # classes to iterate over


dataset=Cifar100
n_classes=100
# Add the path to your ViT weights
weight_path=checkpoint/ViT/Tuesday_09_January_2024_14h_53m_51s/ViT-Cifar100-8-best.pth

echo "starting the stuff"
for val in "${StringArray[@]}"; do
    forget_class=$val
    #Run the Python script
    CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method baseline -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method selective_pruning -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method finetune -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method blindspot -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method UNSIR -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method amnesiac -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method ssd_tuning -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE poetry run python forget_full_class_main.py -net ViT -dataset $dataset -classes $n_classes -gpu -method retrain -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
done
