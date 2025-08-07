declare -a arr=("007_ecc-0-05_rand-0-40_perlin-2"
                "001_ecc-0-05_rand-0-10_perlin-2"
                "019_ecc-0-80_rand-0-10_perlin-2"
                "013_ecc-0-60_rand-0-20_perlin-2"
                "025_ecc-0-80_rand-0-40_perlin-2"
                "016_ecc-0-60_rand-0-40_perlin-2"
                "010_ecc-0-60_rand-0-10_perlin-2"
                "022_ecc-0-80_rand-0-20_perlin-2"
                "004_ecc-0-05_rand-0-20_perlin-2"
               )


for i in "${arr[@]}"
do
	echo "Extracting forced_cell_shape/${i}/data/processed/*cell-seg.png"
    tar -xvf PCST.tar.gz "forced_cell_shape/${i}/data/processed/*cell-seg.png"
done

