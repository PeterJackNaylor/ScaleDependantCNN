
pyf = "src/python/nn"
pretrained_py = file("${pyf}/pretrained.py")

process pretrained {
    publishDir "${DATA}_output/pretrained", mode: 'symlink'

    input:
        tuple val(DATA), path(DATA_csv), path(DATA_npy)
        each OPT
    output:
        tuple val("${NAME}_pretrained"), path("${NAME}_pretrained.csv"), path(DATA_csv)
        path("${NAME}_pretrained_training_statistics.csv")
        path("*.csv")
    script:
        NAME = "${DATA}_${OPT}"
        """
        python $pretrained_py --data_path $DATA_npy --data_info $DATA_csv \
                            --output ./ --name $NAME $OPT
        """
}

process duplicat_pretrain{
    input:
        tuple val(DATA), path(DATA_csv), path(DATA_npy)
        each REP
    output:
        tuple val(DATA), path(DATA_csv), path(DATA_npy)

    script:
        """
        echo "Doing nothing here, just duplicating"
        """
}

workflow pretrained_imagenet {
    take:
        manual_features
        opt
        repetition
    main:
        pretrained(manual_features, opt)
        duplicat_pretrain(pretrained.out[0], 1..repetition)
    emit:
        pt_encoding = duplicat_pretrain.out
}
        

s_learning = file("${pyf}/s_learning.py")

process supervised_extraction {
    publishDir "${DATA}_output/supervised", mode: 'symlink'

    input:
        tuple val(DATA), path(DATA_csv), path(DATA_npy)
        each MODEL_NAME
        each OPT
        each REP
        each LR
        each WD
    output:
        tuple val("${NAME}_supervised"), path("${NAME}_supervised.csv"), path(DATA_csv)
        path("${NAME}_supervised_training_statistics.csv")
        path("*.csv")
    script:
        NAME = "${DATA}_${MODEL_NAME}_${OPT}_${LR}_${WD}"
        if ("${DATA}" == "pannuke"){
            BS = 512
            EPOCH = 50
        } else {
            BS = 128
            EPOCH = 100
        }
        """
        python $s_learning --data_path $DATA_npy --data_info $DATA_csv \
                        --batch_size $BS --workers 8 --epochs $EPOCH \
                        $OPT --model_name $MODEL_NAME --wd $WD\
                        --lr $LR --output ./ --name $NAME
                    
        """
}