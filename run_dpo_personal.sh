#!/bin/bash
#SBATCH --job-name=globalopinionv4_sft
#SBATCH --nodes=1
#SBATCH --output=output_globalopinionv4_sft.txt
#SBATCH --error=error_globalopinionv4_sft.txt
#SBATCH --ntasks=1
#SBATCH -G 8
#SBATCH --mem=632G

source /home/ubuntu/multi-dpo-storage/multi_pref_DPO/kee-env/bin/activate

# added configuration for FSDP
ulimit -n 64000


# # program to run regular SFT
python -u em_dpo.py\
	save_model=true\
	model=mistral7b\
	exp_name=globalopinion_sft\
	datasets=[globalopinion]\
	loss=sft\
	num_users=800\
	num_groups=1\
	em_steps=1\
	gradient_accumulation_steps=2\
	batch_size=128\
	eval_batch_size=64\
	trainer=FSDPTrainer\
	sample_during_eval=false\
	seed=0\

cp temp/globalopinion_sft_seed0/group-0/group0_emiteration0_policy.pt policies/globalopinion/sft_policy.pt

# program to run EMDPO / naive DPO
python -u em_dpo.py\
	seed=0\
	save_model=false\
	model=mistral7b\
	exp_name=globalopinon_emdpo\
	datasets=[globalopinion]\
	loss=dpo\
	loss.beta=0.1\
	num_users=800\
	num_groups=4\
	em_steps=5\
	gradient_accumulation_steps=2\
	batch_size=256\
	eval_batch_size=128\
	trainer=FSDPTrainer\
	sample_during_eval=false\
	model.fsdp_policy_mp=bfloat16\
	model.archive=/home/ubuntu/multi-dpo-storage/multi_pref_DPO/policies/globalopinion/sft_policy.pt
	# model.archive=/home/ubuntu/multi-dpo-storage/multi_pref_DPO/policies/imdb/sft_policy.pt
	# model.archive=/scratch/m000086/multi_pref_DPO_marlowe/policies/globalopinion/sft_policy.pt


# # Program to run cluster DPO
for CLUSTER in {0..3}
do
    python -u em_dpo.py \
        seed=0 \
        save_model=true \
        model=mistral7b \
        exp_name=globalopinion_cluster_$CLUSTER \
        datasets=[globalopinion_cluster_$CLUSTER] \
        loss=dpo \
        loss.beta=0.1 \
		num_users=800\
        num_groups=1 \
        em_steps=1 \
        gradient_accumulation_steps=2 \
        batch_size=64 \
        eval_batch_size=32 \
        trainer=FSDPTrainer \
        sample_during_eval=false \
        model.fsdp_policy_mp=bfloat16 \
        model.archive=/scratch/m000086/multi_pref_DPO_marlowe/policies/globalopinion/sft_policy.pt
done

# python find_weightsv2.py --exp_name "emdpo" --seed 1 --sft_policy '/scratch/m000086/multi_pref_DPO_marlowe/policies/synthetic/sft_policy.pt'
# python find_weightsv2.py --exp_name "clusterdpo" --seed 5 --sft_policy '/scratch/m000086/multi_pref_DPO_marlowe/policies/synthetic/sft_policy.pt'

# # # program to run regular DPO
# for SEED in $(seq 5 9); do
# 	python -u em_dpo.py\
# 		seed=$SEED\
# 		save_model=true\
# 		model=mistral7b\
# 		exp_name=cluster1_allseeds\
# 		datasets=[imdb_cluster1]\
# 		loss=dpo\
# 		loss.beta=0.1\
# 		num_groups=1\
# 		em_steps=1\
# 		gradient_accumulation_steps=2\
# 		batch_size=64\
# 		eval_batch_size=32\
# 		trainer=FSDPTrainer\
# 		sample_during_eval=false\
# 		model.fsdp_policy_mp=bfloat16\
# 		model.archive=/scratch/m000086/multi_pref_DPO_marlowe/policies/personalllm/sft_policy.pt
# done

# Print a message to indicate the completion of the training
echo "Training process completed."

