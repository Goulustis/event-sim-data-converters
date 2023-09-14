SCENE_NAME=carpet_light

python gen_triggers.py --num_frames 4096
python main.py --outdir generated_imgs/$SCENE_NAME
python gen_triggers.py --num_frames 2048
python main.py --outdir generated_imgs/${SCENE_NAME}_2048
