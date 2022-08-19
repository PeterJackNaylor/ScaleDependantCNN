

include { extraction; manual; add_padding_mask } from './src/nf/manual_feat'
include { ssl_bt_transform } from './src/nf/self_supervised'
include { LR_KNN_evaluation } from './src/nf/evaluation'

// parameters
LAMBDA = [0.0078125]
LR = [0.001]
WD = [1e-6]
FEATURE_DIM = [64] 
models = ["ModelSDRN"]
opt = ["--no_size"]
augmentations = ["vanilla", "autocontrast", "jittersmall", "jittermed", "jitterlarge", "jitterverylarge", "greyscale"]
repetition = 20

aug_exp = file("src/python/aug_plot.py")

// data
// dataset = Channel.from([file("./data/tnbc"), file("./data/consep"), file("./data/pannuke")])
dataset = Channel.from([file("./data/tnbc")])

process plot {
    publishDir "paper_output/augmentation", mode: 'symlink'

    input:
        path(TESTING_TABLE)
        path(TRAINING_TABLE)
    output:
        path("*.png")
        path("*.html")
    script:
        """
        python $aug_exp $TESTING_TABLE
        """
}


workflow {
    main:
        extraction(dataset)
        ext = extraction.out

        ssl_bt_transform(ext, models, opt, LAMBDA, FEATURE_DIM, 1..repetition, LR, WD, augmentations)
        LR_KNN_evaluation(ssl_bt_transform.out[0])
        plot(LR_KNN_evaluation.out.collectFile(skip: 1, keepHeader: true).collect(), 
            ssl_bt_transform.out[1].collectFile(skip: 1, keepHeader: true).collect())
}