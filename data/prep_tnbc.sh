#download data
wget https://zenodo.org/record/3552674/files/TNBC_and_Brain_dataset_celltype_integer.zip?download=1
mv TNBC_and_Brain_dataset_celltype_integer.zip\?download=1 tnbc.zip
unzip tnbc.zip -d tnbc
mv tnbc/TNBC_and_Brain_dataset_celltype_integer data/tnbc 

# remove brain files
rm -r data/tnbc/GT_12/
rm -r data/tnbc/GT_13/
rm -r data/tnbc/GT_14/
rm -r data/tnbc/Slide_12/
rm -r data/tnbc/Slide_13/
rm -r data/tnbc/Slide_14/

# remove otherfiles
rm tnbc.zip
rm -r tnbc