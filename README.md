# event-sim-data-converters
simulated event data converters and processing scripts. See the readme in each folder for details of how to use.

### e-nerf_synth_datapipeline contains:
- enerf -> nerfstudio

### esim_pipeline contains:
- raw esim -> nerfstudio

### synthetic_pipeline contains:
- raw synthetic data (robot only) -> nerfstudio format
- nerfstudio format (robot only) -> enerf
- nerfstudio format (any nerfstudio) -> enerf


### instructions
setup to run create\_dataset.sh
- install vid2e
- install nerfies

### scene generation
A scene can be generated with this [game engine](https://github.com/Goulustis/3D-Graphics-Engine/tree/old_scene)