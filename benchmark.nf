
// Imports
include { extraction; manual; add_padding_mask } from './src/nf/manual_feat'
include { pretrained_imagenet; supervised_extraction } from './src/nf/supervised'
include { ssl_moco; ssl_bt } from './src/nf/self_supervised'
include { evaluation } from './src/nf/evaluation'

// data
// dataset = Channel.from([file("./data/tnbc"), file("./data/consep"), file("./data/pannuke")])
dataset = Channel.from([file("./data/tnbc"), file("./data/consep")])

// parameters
methods_selection = ["ascending", "descending"]
LAMBDA = [0.0078125]
LR = [0.01]
WD = [1e-4]
FEATURE_DIM = [64] 
models = ["ModelSDRN", "ModelSRN"]
opt = ["--inject_size", "--no_size"]
MB = [65536]
KS = [3]

padding_size = 128
repetition = 2

epochs = [ "tnbc": 100, "consep": 100, "pannuke": 50, "tnbcpadded": 100, "conseppadded": 100, "pannukepadded": 50]
bs = [ "tnbc": [128], "consep": [128], "pannuke": [512], "tnbcpadded": [64], "conseppadded": [64], "pannukepadded": [64]]
number_bs = [0]

workflow {
    main:
        extraction(dataset)
        // ext = extraction.out

        // manual(ext, methods_selection, repetition)

        // add_padding_mask(ext, padding_size) .set{data}

        // pretrained_imagenet(data, opt, repetition)

        // supervised_extraction(data, models, opt, 1..repetition, LR, WD, KS, epochs, bs, number_bs)

        // ssl_moco(data, models, opt, 1..repetition, LR, WD, MB, KS, epochs, bs, number_bs)

        // ssl_bt(data, models, opt, LAMBDA, FEATURE_DIM, 1..repetition, LR, WD, KS, epochs, bs, number_bs, ["normal"])


        // ssl_bt.out[0].concat(manual.out[0], supervised_extraction.out[0], pretrained_imagenet.out, ssl_moco.out[0]) .set {encodings}
        // ssl_bt.out[1].concat(manual.out[1], supervised_extraction.out[1], ssl_moco.out[1]) .collectFile(skip: 1, keepHeader: true).collect() .set {training_score}

        // evaluation(encodings, training_score)
}