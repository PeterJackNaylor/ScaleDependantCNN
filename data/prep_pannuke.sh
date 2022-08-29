# download data
wget https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip
wget https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip
wget https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip
unzip fold_1.zip -d fold_1
unzip fold_2.zip -d fold_2
unzip fold_3.zip -d fold_3


# # create data folder
mkdir data/pannuke
mkdir data/pannuke/images
mkdir data/pannuke/masks

mv fold_1/Fold\ 1/images/fold1 data/pannuke/images/fold1
mv fold_1/Fold\ 1/masks/fold1 data/pannuke/masks/fold1
mv fold_2/Fold\ 2/images/fold2 data/pannuke/images/fold2
mv fold_2/Fold\ 2/masks/fold2 data/pannuke/masks/fold2
mv fold_3/Fold\ 3/images/fold3 data/pannuke/images/fold3
mv fold_3/Fold\ 3/masks/fold3 data/pannuke/masks/fold3

# remove otherfiles
rm fold_1.zip
rm fold_2.zip
rm fold_3.zip
rm -r fold_1
rm -r fold_2
rm -r fold_3
