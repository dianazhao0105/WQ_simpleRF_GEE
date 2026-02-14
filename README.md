# WQ_simpleRF_GEE

This repository provides a simple, practical Random Forest (RF) workflow for water-quality parameter (WQP) retrieval and deployment on Google Earth Engine (GEE).

## Features
- Pearson-correlation screening for spectral features
- Optional environmental covariate selection
- Export trained RF trees to GEE assets via `geemap.ml`

## Inputs
You need:
1. `train_val.csv`: matchup table for training/validation  
2. `independent_test.csv`: independent test table  
3. `feature_flags.csv`: feature list with flags (`single`, `combo`, `env`, `keep`)
