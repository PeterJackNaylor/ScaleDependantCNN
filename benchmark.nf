
// Imports
include { extraction; manual; add_padding_mask } from './src/nf/manual_feat'
include { pretrained_imagenet; supervised_extraction } from './src/nf/supervised'
include { ssl_moco_benchmark; ssl_bt } from './src/nf/self_supervised'
include { evaluation } from './src/nf/evaluation'

// data
// dataset = Channel.from([file("./data/tnbc"), file("./data/consep"), file("./data/pannuke")])
dataset = Channel.from([file("./data/tnbc"), file("./data/consep")])

// parameters
methods_selection = ["ascending", "descending"]
LAMBDA = [0.00078125, 0.0078125, 0.078125]
LR = [0.001, 0.01, 0.1]
WD = [0, 1e-3, 1.0]
FEATURE_DIM = [64] 
models = ["ModelSDRN", "ModelSRN"]
opt = ["--inject_size", "--no_size"]
MB = [65536]
padding_size = 128
repetition = 40

workflow {
    main:
        extraction(dataset)
        ext = extraction.out

        manual(ext, methods_selection, repetition)

        add_padding_mask(ext, padding_size) .set{data}

        pretrained_imagenet(data, opt, repetition)

        supervised_extraction(data, models, opt, 1..repetition, LR, WD)

        // ssl_moco_benchmark(data, models, opt, 1..repetition, LR, WD, MB)
        // ssl_moco_benchmark.out[0],
        // ssl_moco_benchmark.out[1], 

        ssl_bt(data, models, opt, LAMBDA, FEATURE_DIM, 1..repetition, LR, WD)


        ssl_bt.out[0].concat(manual.out[0], supervised_extraction.out[0], pretrained_imagenet.out) .set {encodings}
        ssl_bt.out[1].concat(manual.out[1], supervised_extraction.out[1]) .collectFile(skip: 1, keepHeader: true).collect() .set {training_score}

        evaluation(encodings, training_score)
}