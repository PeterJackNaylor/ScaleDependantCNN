
pyf = "src/python/nn"

bt_training = file("${pyf}/ssl_bt.py")

process ssl_bt {
    publishDir "${DATA}_output/ssl/BT", mode: 'symlink'

    input:
        tuple val(DATA), path(DATA_csv), path(DATA_npy)
        each MODEL_NAME
        each OPT
        each LAMBD
        each FDIM
        each REP
        each LR
        each WD
        each KS
        val EPOCH
        val BS
        each bs_int
        each AUG
    output:
        tuple val("${NAME}_ssl"), path("*_ssl.csv"), path(DATA_csv)
        path("${NAME}_ssl_training_statistics.csv")

    when:
        !(DATA.contains("padded")) || (MODEL_NAME == "ModelSRN")

    script:
        bs = BS[DATA][bs_int]
        epoch = EPOCH[DATA]
        NAME = "${DATA}_${MODEL_NAME}_${OPT}_LAMB-${LAMBD}_FDIM-${FDIM}_${LR}_${WD}_${KS}_${bs}_${AUG}"
        """

    	python ${bt_training} --lmbda ${LAMBD} --corr_zero \
                        --batch_size $bs --feature_dim ${FDIM} \
                        --epochs $epoch --output . --ks $KS \
                        --model_name $MODEL_NAME --name $NAME \
                        --data_path ${DATA_npy} --data_info ${DATA_csv} \
                        --workers 8 --lr $LR --wd $WD $OPT

        """
}

moco_training = file("${pyf}/ssl_moco.py")

process ssl_moco {
    publishDir "${DATA}_output/ssl/MoCo", mode: 'symlink'

    input:
        tuple val(DATA), path(DATA_csv), path(DATA_npy)
        each MODEL_NAME
        each OPT
        each REP
        each LR
        each WD
        each MB
        each KS
        val EPOCH
        val BS
        each bs_int
    output:
        tuple val("${NAME}"), path("*_moco.csv"), path(DATA_csv)
        path("${NAME}_training_statistics.csv")

    when:
        !(DATA.contains("padded")) || (MODEL_NAME == "ModelSRN")

    script:
        bs = BS[DATA][bs_int]
        epoch = EPOCH[DATA]
        NAME = "${DATA}_${MODEL_NAME}_${OPT}_${LR}_${WD}_${KS}_${MB}_${bs}_moco"
        """
    	python ${moco_training} --data_path ${DATA_npy} \
                        --data_info ${DATA_csv} \
                        --arch $MODEL_NAME -j 6 \
                        --name $NAME --ks $KS \
                        --batch-size $bs \
                        --memory-bank $MB \
                        --epochs $epoch --output . \
                        --gpu 0 \
                        --lr $LR --wd $WD $OPT
        """
}
