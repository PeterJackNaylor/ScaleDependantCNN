
include { extraction; manual; add_padding_mask } from './src/nf/manual_feat'
include { ssl_moco; ssl_bt } from './src/nf/self_supervised'
include { Evaluation } from './src/nf/evaluation'


// data
// dataset = Channel.from([file("./data/tnbc"), file("./data/consep"), file("./data/pannuke")])
dataset = Channel.from([file("./data/tnbc")])

// parameters
LR = [0.0001, 0.001, 0.01]
WD = [0, 1e-4, 1.0]
MB = [4096, 16384, 65536]
BS = [256, 512, 1024]

LR = [0.001]
WD = [0]
MB = [4096, 65536]
KS = [3]

models = ["ModelSRN"]
opt = ["--no_size"]

BS_RANGE = [128, 512, 1024]
number_bs = 0..2
epochs = [ "tnbc": 100, "consep": 100]
bs = [ "tnbc": BS_RANGE, "consep": BS_RANGE]

repetition = 2

moco_exp = file("src/python/moco_plot.py")


process plot {
    publishDir "paper_output/", mode: 'symlink'

    input:
        path(TESTING_TABLE)
        path(TRAINING_TABLE)
    output:
        path("moco_experiments.csv")
    script:
        """
        python $moco_exp $TESTING_TABLE
        """
}


workflow {
    main:
        extraction(dataset)
        ext = extraction.out

        ssl_moco(ext, models, opt, 1..repetition, LR, WD, MB, KS, epochs, bs, number_bs)
        Evaluation(ssl_moco.out[0])

        plot(Evaluation.out.collectFile(skip: 1, keepHeader: true).collect(), 
            ssl_moco.out[1].collectFile(skip: 1, keepHeader: true).collect())
}
