
// Imports
include { extraction; manual } from './src/nf/manual_feat'
include { pretrained_imagenet; supervised_extraction } from './src/nf/supervised'
include { ssl_moco; ssl_bt } from './src/nf/self_supervised'
include { evaluation } from './src/nf/evaluation'

// data
// dataset = Channel.from([file("./data/tnbc"), file("./data/consep"), file("./data/pannuke")])
dataset = Channel.from([file("./data/tnbc")])

// parameters
methods_selection = ["ascending", "descending"]
LAMBDA = [0.0078125]
LR = [0.001]
WD = [1e-6]
FEATURE_DIM = [64] 
models = ["ModelSDRN"]//, "ModelSRN"]
opt = ["--inject_size"]//, "--no_size"]
repetition = 2

workflow {
    main:
        extraction(dataset)
        ext = extraction.out

        manual(ext, methods_selection, repetition)

        pretrained_imagenet(ext, opt, repetition)

        supervised_extraction(ext, models, opt, 1..repetition, LR, WD)

        ssl_moco(ext, models, opt, 1..repetition, LR, WD)

        ssl_bt(ext, models, opt, LAMBDA, FEATURE_DIM, 1..repetition, LR, WD)

        evaluation(manual.out[0],
                manual.out[1],
                pretrained_imagenet.out,
                supervised_extraction.out[0],
                supervised_extraction.out[1],
                ssl_moco.out[0],
                ssl_moco.out[1], 
                ssl_bt.out[0], 
                ssl_bt.out[1]
                )
}
