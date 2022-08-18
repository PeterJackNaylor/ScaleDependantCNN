
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
    output:
        tuple val("${NAME}_ssl"), path("*_ssl.csv"), path(DATA_csv)
        path("${NAME}_ssl_training_statistics.csv")

    when:
        !(DATA.contains("padded")) || (OPT == "--no_size" & MODEL_NAME == "ModelSRN")

    script:
        NAME = "${DATA}_${MODEL_NAME}_${OPT}_LAMB-${LAMBD}_FDIM-${FDIM}_${LR}_${WD}"
        if ("${DATA}" == "pannuke"){
            BS = 512
            EPOCH = 20
        } else {
            BS = 128
            EPOCH = 30
        }
        if (("${DATA}").contains("padded")){
            BS = 64
        }
        """
    	python ${bt_training} --lmbda ${LAMBD} --corr_zero \
                        --batch_size $BS --feature_dim ${FDIM} \
                        --epochs $EPOCH --output . \
                        --model_name $MODEL_NAME --name $NAME \
                        --data_path ${DATA_npy} --data_info ${DATA_csv} \
                        --workers 8 --lr $LR --wd $WD $OPT

        """
}


process ssl_bt_transform {
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
        each AUG
    output:
        tuple val("${NAME}_ssl"), path("*_ssl.csv"), path(DATA_csv)
        path("${NAME}_ssl_training_statistics.csv")

    when:
        !(DATA.contains("padded")) || (OPT == "--no_size" & MODEL_NAME == "ModelSRN")

    script:
        NAME = "${DATA}_${MODEL_NAME}_${OPT}_LAMB-${LAMBD}_FDIM-${FDIM}_${LR}_${WD}_${AUG}"
        if ("${DATA}" == "pannuke"){
            BS = 512
            EPOCH = 20
        } else {
            BS = 128
            EPOCH = 30
        }
        if (("${DATA}").contains("padded")){
            BS = 64
        }
        """
    	python ${bt_training} --lmbda ${LAMBD} --corr_zero \
                        --batch_size $BS --feature_dim ${FDIM} \
                        --epochs $EPOCH --output . \
                        --model_name $MODEL_NAME --name $NAME \
                        --data_path ${DATA_npy} --data_info ${DATA_csv} \
                        --workers 8 --lr $LR --wd $WD $OPT --transforms_type $AUG

        """
}



moco_training = file("${pyf}/ssl_moco.py")

process ssl_moco_benchmark {
    publishDir "${DATA}_output/ssl/MoCo", mode: 'symlink'

    input:
        tuple val(DATA), path(DATA_csv), path(DATA_npy)
        each MODEL_NAME
        each OPT
        each REP
        each LR
        each WD
        each MB
    output:
        tuple val("${NAME}"), path("*_moco.csv"), path(DATA_csv)
        path("${NAME}_training_statistics.csv")

    when:
        !(DATA.contains("padded")) || (OPT == "--no_size" & MODEL_NAME == "ModelSRN")

    script:
        NAME = "${DATA}_${MODEL_NAME}_${OPT}_${LR}_${WD}_${MB}_moco"
        if ("${DATA}" == "pannuke"){
            BS = 512
            EPOCH = 20
        } else {
            BS = 256
            EPOCH = 30
        }
        if (("${DATA}").contains("padded")){
            BS = 64
        }
        """
    	python ${moco_training} --data_path ${DATA_npy} \
                        --data_info ${DATA_csv} \
                        --arch $MODEL_NAME -j 6 \
                        --name $NAME \
                        --batch-size $BS \
                        --memory-bank $MB \
                        --epochs $EPOCH --output . \
                        --gpu 0 \
                        --lr $LR --wd $WD $OPT
        """
}


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
        each BS
        val EPOCH
    output:
        tuple val("${NAME}"), path("*_moco.csv"), path(DATA_csv)
        path("${NAME}_training_statistics.csv")

    when:
        !(DATA.contains("padded")) || (OPT == "--no_size" & MODEL_NAME == "ModelSRN")

    script:
        NAME = "${DATA}_${MODEL_NAME}_${OPT}_${LR}_${WD}_${MB}_${BS}_moco"
        """
    	python ${moco_training} --data_path ${DATA_npy} \
                        --data_info ${DATA_csv} \
                        --arch $MODEL_NAME -j 6 \
                        --name $NAME \
                        --batch-size $BS \
                        --memory-bank $MB \
                        --epochs $EPOCH --output . \
                        --gpu 0 \
                        --lr $LR --wd $WD $OPT
        """
}
