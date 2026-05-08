.. geomaps-landcover documentation master file

*******************************
Odeon-landcover's documentation
*******************************

ODEON is a PyTorch Lightning + Hydra deep-learning pipeline for semantic
segmentation on aerial and satellite imagery.

Installation
============

These instructions assume that you already have `conda <https://conda.io/>`_ installed.

.. code-block:: console

   $ git clone https://gitlab.com/dai-projets/geomaps-landcover
   $ cd geomaps-landcover
   $ conda env create --file=environment.yml
   $ conda activate geomaps-lightning
   $ pip install .

In order to use CUDA and NVIDIA devices, ``cudatoolkit`` must be installed too.


Quickstart
==========

Odeon is run through Hydra-driven CLI tasks:

.. code-block:: console

   $ geomaps task=train +experiment=ocsge
   $ geomaps task=test  +experiment=ocsge_test
   $ geomaps task=pred  +experiment=ocsge_pred

Configuration files live under ``geomaps/configs/conf/``.

.. toctree::
    :caption: User Guide
    :maxdepth: 1

    Training <train.rst>

.. toctree::
    :caption: Advanced Feature
    :maxdepth: 1

    Setup <setup.rst>

.. toctree::
    :caption: API Reference
    :maxdepth: 1

    Tools <tools.rst>
    Models <models.rst>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
