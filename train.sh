source /data1/by/miniconda3/etc/profile.d/conda.sh
conda activate byTorch

LR=0.0002
MAMLLR=0.0001
MAXEPCOH=80
LP=8
BS=24
SOURCEDATA='dukemtmcreid'
ATTDATASET='market1501'
models_V="ide bot lsro mudeep pcb aligned mgn hacnn vit transreid pat"
TASKMUM=5
XISHU=10

GPU_NUM=3

MULTI_D=1
MULTI_M=1
PETRUB_E=1
MULTI_E=0
MS_MIX=0
LAMDA=0.5
BETA=10
EXTRA=0
MIX_THRE=0.8
BOTH=0

LOG='wTE_wPE_wG_3'
LOG_PATH="./log_eccv_test/"$LOG
G_RESUME=$LOG_PATH"/best_G.pth.tar"
D_RESUME=$LOG_PATH"/best_D.pth.tar"
TEST_LOG=$LOG_PATH"/rgb_test"

cd /data4/by/reid/github/MissRank_old
CUDA_VISIBLE_DEVICES=$GPU_NUM python train_MTGA.py --lp $LP --xishu $XISHU --log_dir $LOG_PATH --dataset $SOURCEDATA \
--epoch $MAXEPCOH --lr $LR --maml_lr $MAMLLR --MetaTrainTask_num $TASKMUM --train_batch $BS \
--multi_domain $MULTI_D --multi_model $MULTI_M --perturb_erasing $PETRUB_E --multi_pe $MULTI_E --style_mix $MS_MIX \
--lamda $LAMDA --Beta_Sum $BETA --extra_data $EXTRA --mix_thre $MIX_THRE --out_both $BOTH

for model in $models_V; do
  CUDA_VISIBLE_DEVICES=$GPU_NUM python test.py --G_resume_dir $G_RESUME --D_resume_dir $D_RESUME \
  --test_logDir $TEST_LOG --vis_dir $TEST_LOG --targetmodel $model --dataset $ATTDATASET --lp $LP
done

