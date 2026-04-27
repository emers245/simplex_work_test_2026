# Mixed Mess3 Belief-State Geometry

How does a simple transformer model track belief state geometry when multiple Mess3 processes have generated the training data? This repository establishes a conjecture about the dimensionality of the latent belief state geometry in the residual stream, then tests this using a simple training set that uses two separate Mess3 processes to generate sequence data for training.

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

Mult_Mess3_BSG.pdf -
  A PDF write-up of predictions and results.

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
