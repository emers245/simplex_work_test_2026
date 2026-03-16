# Simplex Work Test - 2026

This code repository is for Joe's MATS Summer 2026 Simplex stream work test.

## Set-up

The environment can be set up using an Anaconda or Miniconda distribution:

```
conda env create -f environment.yml
```

Otherwise the packages listed in environment.yml can be installed individually.

## Code

analysis_notebook.ipynb - 
  The main code for creating the models and analyzing the model internals.

explore_mess3.ipynb -
  A notebook for familiarizing yourself with Mess3 processes.

mess3.py -
  A custom Mess3 python class.

## Files

MATS_Summer_2026__Simplex_Work_Test.pdf -
  A PDF write-up of the work test

process_*_data.h5 -
  Some stored sequence data from the Mess3 processes

simplex_transformer_streamed.pth -
  The first transformer model created in analysis_notebook.ipynb

simplex_transformer_streamed_2.pth
  The second transformer model created in analysis_notebook.ipynb with a longer context window

## Directories

plots/ -
  Contains PNG plots
  model1/ -
    The first transformer model created in analysis_notebook.ipynb
  model2_context16/ - 
    The second transformer model created in analysis_notebook.ipynb with a longer context window
  Figures/ -
    Some figures made in SVG and PNG format
  old/ -
    Old plots
