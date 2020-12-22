python run_generator.py generate_images --network=checkpoints/stylegan2_512x512_with_pretrain/pretrain/Gs.pth --seeds=1234567890 --truncation_psi=0.5 --out=out_real

# python run_generator.py generate_images --network=checkpoints/stylegan2_512x512_with_pretrain_new/1600_2020-12-20_17-18-05/Gs.pth --seeds=100-200 --truncation_psi=0.5 --out=results3

# python run_generator.py generate_images --network=G_out.pth --seeds=100-200 --truncation_psi=0.5 --out=results

#python run_generator.py generate_images --network=G_out_2.pth --seeds=100-200 --truncation_psi=0.5 --out=results2

# python run_generator.py generate_images --network=G_out_3.pth --seeds=100-200 --truncation_psi=0.5 --out=results4

python run_generator.py generate_images --network=/home/wxr/stylegan2_pytorch_backup/checkpoints/stylegan2_512x512_with_pretrain_new_2/6000_2020-12-21_20-44-35/Gs.pth --seeds=1234567890 --truncation_psi=0.5 --out=results5
