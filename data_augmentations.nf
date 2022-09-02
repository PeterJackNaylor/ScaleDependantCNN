

include { extraction; manual; add_padding_mask } from './src/nf/manual_feat'
include { ssl_bt } from './src/nf/self_supervised'
include { Evaluation } from './src/nf/evaluation'

// parameters
LAMBDA = [0.0078125]
LR = [0.001]
WD = [1e-6]
FEATURE_DIM = [64] 
KS = [3]

models = ["ModelSDRN"]
opt = ["--no_size"]
epochs = [ "tnbc": 100, "consep": 100, "pannuke": 50, "tnbcpadded": 100, "conseppadded": 100, "pannukepaddded": 50]
bs = [ "tnbc": [128], "consep": [128], "pannuke": [512], "tnbcpadded": [64], "conseppadded": [64], "pannukepadded": [64]]
number_bs = [0]

augmentations = ["normal", "vanilla", "autocontrast", "jittersmall", "jittermed", "jitterlarge", "jitterverylarge", "greyscale"]

repetition = 2

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

        ssl_bt(ext, models, opt, LAMBDA, FEATURE_DIM, 1..repetition, LR, WD, KS, epochs, bs, number_bs, augmentations)
        Evaluation(ssl_bt.out[0])
        plot(Evaluation.out.collectFile(skip: 1, keepHeader: true).collect(), 
            ssl_bt.out[1].collectFile(skip: 1, keepHeader: true).collect())
}