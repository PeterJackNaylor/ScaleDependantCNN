
pyf = "src/python/manual"

manual_extraction = file("${pyf}/main.py")
pannuke_extraction = file("${pyf}/pannuke_main.py")
merge = file("${pyf}/merge.py")

process extraction {
    publishDir "${DATA}_output/extraction", mode: 'symlink'
    
    input:
        path DATA
    output:
        tuple val("${DATA}"), path("${DATA}.csv"), path("${DATA}_tinycells.npy")
    
    script:
        OPTIONS = "--cell_resize 32 --cell_marge 5 --marge 10 --n_jobs 1 --out_path ."
        if ("${DATA}" == "tnbc")
            """
            python $manual_extraction --folder $DATA --name $DATA $OPTIONS
            """
        else if ("${DATA}" == "consep")
            """
            python $manual_extraction --folder $DATA/Train --name ${DATA}_train --type $DATA $OPTIONS
            python $manual_extraction --folder $DATA/Test  --name ${DATA}_test  --type $DATA $OPTIONS
            python $merge ${DATA}_train.csv ${DATA}_test.csv ${DATA}_train_tinycells.npy ${DATA}_test_tinycells.npy $DATA
            """
        else if ("${DATA}" == "pannuke")
            """
            python $pannuke_extraction --folder $DATA --name $DATA --type $DATA $OPTIONS
            """
        else
            error "Invalid data: ${DATA}"
}

manual_selection = file("${pyf}/selection_knn.py")

process stepwise_selection {
    publishDir "${DATA}_output/stepwise_${TYPE}", mode: 'symlink'

    input:
        tuple val(DATA), path(DATA_csv), path(DATA_npy)
        each TYPE
        each REP
    output:
        tuple val("${DATA}_${REP}"), path("selected_feat_${TYPE}.npy"), path("train_score_${TYPE}.npy"), path("test_score_${TYPE}.npy")
    script:
        """
        python $manual_selection ${DATA} ./ $TYPE
        """
}

pymanual = Channel.from([file("${pyf}/intersection.py"), file("${pyf}/union.py"), file("${pyf}/ascending.py"), file("${pyf}/descending.py")])


process cross_selection {
    publishDir "${DATA}_output/cross_selection", mode: 'symlink'

    input:
        tuple val(DATA), path(select), path(train), path(test), path(knn), path(DATA_csv), path(DATA_npy)
        each PY
    output:
        tuple val("${DATA}${tag}"), path("${DATA}_*data.csv"), path(DATA_csv)
        path("${DATA}_*_training_statistics.csv") 
    script:
        tag = "cs_${PY.baseName}"
        """
        python $PY $DATA ${DATA_csv} ${select[0]} ${select[1]} ${knn[0]} ${knn[1]} ${train[0]} ${train[1]}  ${test[0]} ${test[1]} 
        """
}

pad_mask = file("${pyf}/pad_mask.py")

process padding_mask {
    input:
        tuple val(DATAn), path(DATAcsv), path(DATAnpy)
        val SIZE
    output:
        tuple val("${DATAn}padded"), path(DATAcsv), path("${DATAn}_tinycells_paddedmask.npy")
    script:
        """
        python $pad_mask ${DATAnpy} ${DATAcsv} ${SIZE}
        """

}

workflow add_padding_mask {
    take:
        manual_features
        size
    main:
        padding_mask(manual_features, size)
        padding_mask.out.concat(manual_features).set{output}
    emit:
        output
}

workflow manual {
    take:
        manual_features
        selection_methods
        repetition
    main:
        stepwise_selection(manual_features, selection_methods, 1..repetition)
        stepwise_selection.out.groupTuple(by: 0).map{it -> [it[0].split('_')[0], it[1], it[2], it[3], it[4]] }.combine(manual_features, by: 0).set{ss_by_repeats}
        cross_selection(ss_by_repeats, pymanual)
    emit:
        manual_encoding = cross_selection.out[0]
        statistics = cross_selection.out[1]
}       