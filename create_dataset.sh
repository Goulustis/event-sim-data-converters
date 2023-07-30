
set -e

### params #######
FRAME_DIR=/home/hunter/projects/3D-Graphics-Engine/generated_imgs/cat_simple
SIMULATOR=esim #[one of esim, dreb]
EV_THRESH=0.2
DEST_DIR=formatted_data/cat_simple
COARSE_FRAME_DIR=/home/hunter/projects/3D-Graphics-Engine/generated_imgs/cat_simple_256
####################

WORK_DIR=$(pwd)

# mamba activate vid2e
which python

if [ ! -d "$COARSE_FRAME_DIR" ]; then
    echo creating coarse frames
    FRAME_DIR_2=/home/hunter/projects/3D-Graphics-Engine/generated_imgs/cat_simple_2048   # PARAMS NEEDED TO CREATE COARSE_FRAME
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

# echo creating events ....
# if [ "$SIMULATOR" = "esim" ]; then
#     python evs_generators/gen_esim_evs/gen_evs.py --frame_dir $FRAME_DIR --targ_f $DST_EV_F --ev_thresh $EV_THRESH
# elif [ "$SIMULATOR" = "dreb" ]; then
#     python evs_generators/dreb_simulator/gen_events.py --frame_dir $FRAME_DIR --targ_f $DST_EV_F --ev_thresh $EV_THRESH
# else
#     echo "$SIMULATOR is not supported"
# fi


cd synth_datapipeline
python format_ecam_set.py --dst_dir $DEST_DIR/ecam_set \
                          --coarse_rgb_dir $COARSE_FRAME_DIR \
                          --evs_f $DST_EV_F \


python format_col_set.py --dst_dir $DEST_DIR/colcam_set \
                         --coarse_rgb_dir $COARSE_FRAME_DIR \
                         --trig_ids_f $DEST_DIR/ecam_set/trig_ids.npy

echo $EV_THRESH >> $DEST_DIR/ecam_set/eimgs/1x/ev_thresh.txt

cp $DEST_DIR/colcam_set/scene.json $DEST_DIR/ecam_set/
cd $WORK_DIR