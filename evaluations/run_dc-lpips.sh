### meature D&C Density and coverage metrics
### change related path inside
python DC.py

### lpips
python PerceptualSimilarity/lpips_2dirs.py -d0 imgs/ex_dir0 -d1 imgs/ex_dir1 -o ./imgs/example_dists.txt --use_gpu
python PerceptualSimilarity/lpips_2dirs.py -d0 /data2/gyang/TAGAN/results/summer2winter-F64-mixer/test_350/images/real_B -d1 /data2/gyang/TAGAN/results/summer2winter-F64-mixer/test_350/images/fake_B -o ./example_dists.txt --use_gpu
### fid
python -m pytorch_fid /data2/gyang/TAGAN/results/horse2zebra-F64-unaligned/test_140/images/real_B /data2/gyang/TAGAN/results/horse2zebra-F64-unaligned/test_140/images/fake_B
### KID
pip install torch-fidelity
fidelity --gpu 1 --kid --input1 /data2/gyang/TAGAN/results/cat2dog/ours/fake_B --input2 /data2/gyang/TAGAN/results/cat2dog/ours/real_B --kid-subset-size 500
### cityscape
python3 segment.py test -d ./datasets/cityscapes -c 19 --arch drn_d_22 \
    --pretrained ./drn_d_22_cityscapes.pth --phase val --batch-size 1