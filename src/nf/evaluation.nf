
evaluate = file("src/python/evaluate.py")

process LR_KNN_evaluation {
    input:
        tuple val(tag), path(Xy), path(CSV)
    output:
        path 'performance.csv'

    script:
        """
        python $evaluate $tag $Xy $CSV
        """
}

pytable = file("src/python/paper_table.py")

process publish {
    publishDir "paper_output/", mode: 'symlink'

    input:
        path(TESTING_TABLE)
        path(TRAINING_TABLE)
    output:
        path("paper_results.csv")
    script:
        """
        python $pytable $TESTING_TABLE
        """
}

workflow evaluation {
    take:
        cs_enc
        cs_stat
        pretrained
        sl_enc
        sl_stat
        ssl_moco_enc
        ssl_moco_stat
        ssl_bt_enc
        ssl_bt_stat
    main:
        sl_enc.concat(cs_enc, ssl_bt_enc, ssl_moco_enc, pretrained) .set {encodings}
        sl_stat.concat(cs_stat, ssl_bt_stat, ssl_moco_stat) .collectFile(skip: 1, keepHeader: true).collect() .set {training_score}

        LR_KNN_evaluation(encodings)

        LR_KNN_evaluation.out.collectFile(name: "./paper_output/performance.csv", skip: 1, keepHeader: true) .set{test_score}
        publish(test_score, training_score)
}