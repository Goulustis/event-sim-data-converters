
set -e
scene_name=cat_fancy
### params #######
FRAME_DIR=3D-Graphics-Engine/generated_imgs/${scene_name}_4096
SIMULATOR=esim #[one of esim, dreb]
EV_THRESH=0.2
DEST_DIR=formatted_data/$scene_name
COARSE_FRAME_DIR=3D-Graphics-Engine/generated_imgs/${scene_name}_256
####################

WORK_DIR=$(pwd)
FRAME_DIR=$(realpath $FRAME_DIR)
COARSE_FRAME_DIR=$(realpath $COARSE_FRAME_DIR)

# mamba activate vid2e
which python

if [ ! -d "$COARSE_FRAME_DIR" ]; then
    echo creating coarse frames
    FRAME_DIR_2=$(dirname $FRAME_DIR)/${scene_name}_2048   # PARAMS NEEDED TO CREATE COARSE_FRAME
    python synth_datapipeline/misc_tasks.py --src_dir $FRAME_DIR_2 --dst_dir $COARSE_FRAME_DIR
fi

if [ ! -d "$DEST_DIR" ]; then
    mkdir $DEST_DIR
fi

DEST_DIR=$(realpath $DEST_DIR)
FRAME_DIR=$(realpath $FRAME_DIR)
COARSE_FRAME_DIR=$(realpath $COARSE_FRAME_DIR)

### generate variables
DST_EV_F=$DEST_DIR/event.hdf5

echo checking gen evs
if [ "$SIMULATOR" = "esim" ]; then
    echo creating evs with esim api
    python evs_generators/gen_esim_evs/gen_evs.py --frame_dir $FRAME_DIR --targ_f $DST_EV_F --ev_thresh $EV_THRESH
elif [ "$SIMULATOR" = "dreb" ]; then
    echo creating evs with dreb emulator
    python evs_generators/dreb_simulator/gen_events.py --frame_dir $FRAME_DIR --targ_f $DST_EV_F --ev_thresh $EV_THRESH
else
    echo "$SIMULATOR is not supported"
fi


cd synth_datapipeline
python format_ecam_set.py --dst_dir $DEST_DIR/ecam_set \
                          --coarse_rgb_dir $COARSE_FRAME_DIR \
                          --evs_f $DST_EV_F \


python format_col_set.py --dst_dir $DEST_DIR/colcam_set \
                         --coarse_rgb_dir $COARSE_FRAME_DIR \
                         --trig_ids_f $DEST_DIR/ecam_set/trig_ids.npy

echo $EV_THRESH >> $DEST_DIR/ecam_set/eimgs/ev_thresh.txt

cp $DEST_DIR/colcam_set/scene.json $DEST_DIR/ecam_set/

cd $DEST_DIR
cp -r colcam_set clear_linear_colcam_set
cp -r colcam_set clear_gamma_colcam_set
cp -r colcam_set blur_linear_colcam_set
cp -r colcam_set blur_gamma_colcam_set

cd -
python create_gamma_and_blur.py --dataset_dir $DEST_DIR --src_img_dir $FRAME_DIR

cd $WORK_DIR
