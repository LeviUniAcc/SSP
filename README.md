# SSP

## Setup
1. pip install requirements.txt
2. pip install sspspace_main (get it [here](https://github.com/ctn-waterloo/sspspace))

## Run

python .\src\build_SSPs.py --mode train --cpus 1  

## TODO

### 15.11 - 29.11

 - [x] change sspspace py file with library
 - [x] add vocab 2 with combined vectors of shape type (color)
 - [x] generate all ssps only once at the start
 - [ ] *make statistics for different length_scales*
 - [x] add prints for all the other attributes
 - [x] make code a bit cleaner

### 29.11 - 13.12

- [ ] *multiprocessing with fixxed seed* -> probably not needed due to saveing the initial ssps with pikle
- [X] save data with pikle
- [X] save vocab and default environment with pikle
- [ ] load data with dataset
- [ ] *new plotting script*
- [X] demo main
