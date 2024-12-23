python train.py \
--data-dir /home/23zdw/snap/snapd-desktop-integration/157/zdw/many_tools/basecalling_architectures-main/demo_data/nn_input \
--output-dir /home/23zdw/snap/snapd-desktop-integration/157/zdw/many_tools/HAnano_master/experences/result2 \
--window-size 2000 \
--num-epochs 5 \
--batch-size 64 \
--checkpoint /mnt/Data4/23zdw/model/my_model/CSnano-c4-c16-c324-TaaRes-gate-l3-train-best-best/checkpoints/checkpoint_440511.pt

python basecall.py \
--fast5-dir /mnt/Data4/23zdw/data/test_human/Homo_sapiens-FAB42828_guppy/workspace \
--checkpoint /home/23zdw/snap/snapd-desktop-integration/157/zdw/many_tools/HAnano_master/experences/model/HAnano_checkpoint.pt \
--output-file /home/23zdw/snap/snapd-desktop-integration/157/zdw/many_tools/HAnano_master/experences/result/test_4.fastq \
--batch-size 64

python data_prepare.py \
--fast5-dir  /home/23zdw/snap/snapd-desktop-integration/157/zdw/many_tools/basecalling_architectures-main/demo_data/fast5 \
--output-dir  /home/23zdw/snap/snapd-desktop-integration/157/zdw/many_tools/HAnano_master/data/test_nn \
--total-files  1 \
--window-size 2000 \
--window-slide 0 \
--n-cores 4 \
--verbose
