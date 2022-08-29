# download data
wget https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep.zip
unzip consep.zip -d consep
mv consep/CoNSeP data/consep

# remove otherfiles
rm consep.zip
rm -r consep
