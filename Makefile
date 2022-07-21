
download_data:
	bash data/prep_tnbc.sh
	bash data/prep_consep.sh
	bash data/prep_pannuke.sh

setup_conda:
	conda env create -f environment.yml
	pip install -r requirements.txt

experiment :
	nextflow run benchmark.nf -resume --config ./.nextflow.config