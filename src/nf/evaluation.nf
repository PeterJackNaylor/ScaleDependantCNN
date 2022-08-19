
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
        encodings
        training_score
    main:

        LR_KNN_evaluation(encodings)

        LR_KNN_evaluation.out.collectFile(name: "./paper_output/performance.csv", skip: 1, keepHeader: true) .set{test_score}
        publish(test_score, training_score)
}