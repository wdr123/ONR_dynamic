Basic one-way simulation case of one whisker in a flow

The simulation setup is defined in the `FSI_case_###.inp`, which includes the whisker model input file `whisker_001.beam.inp`.

# Input files

1. `FSI_case_001.inp`: defines the whisker simulation setup (total time of simulation, output frequency)
2. `whisker_001.beam.inp`: defines the whisker model (shape, location etc).
3. `whisker_001.cab`: the cab representation of the whisker shape. Needed for force calculation and outputs.

The above files need to be defined for each whisker in an array with the number 001 replaced by a 3 digit index.

**shared input**
4. `materials.inp`: defines the material properties of the whisker, currently shared by all whiskers in an array
5. `cdl.dat`: baseline drag lift coefficient data, for flow force calculation
6. `input_read_flow.dat`: defines flow data path and configuration
7. `input-whisker-signal.dat`: defines signal collection configuration

# Output

The simulation will create an output folder with a name corresponding to the flow data. Since usually the flow simulation is also parametric (multiple cases), subfolders are created for each flow case. The signal output is `FSI_case_001_signal.plt` for whisker 1.

# Array

Run `./wageng` to generate the files for whiskers in an array. The configuration is set up in `array_input.dat`.

Baseline whisker models (e.g. `whisker.cab`) are copied into the array configuration. Note that the baseline models must be placed at origin, i.e. (0,0,0).

`wageng` outputs `.plt` files for visually check the whisker array configuration and its global location.

# Virtual environment for tracking

Use the `whisker_array_driver.py` to drive the whisker simulations and communicate with RL agent.

