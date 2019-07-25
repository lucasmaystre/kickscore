kickscore
=========

|build-status| |coverage|

``kickscore`` is the dynamic skill rating system powering `Kickoff.ai
<https://kickoff.ai/>`_.

In short, ``kickscore`` can be used to understand & visualize the skill of
players (or teams) competing in pairwise matches, and to predict outcomes of
future matches. It extends the `Elo rating system
<https://en.wikipedia.org/wiki/Elo_rating_system>`_ and `TrueSkill
<https://en.wikipedia.org/wiki/TrueSkill>`_.

|nba-history|

Getting started
---------------

To install the latest release directly from PyPI, simply type::

    pip install kickscore

To get started, you might want to explore one of these notebooks:

- `Basic example illustrating the API <examples/kickscore-basics.ipynb>`_
  (`interactive version
  <https://colab.research.google.com/github/lucasmaystre/kickscore/blob/master/examples/kickscore-basics.ipynb>`__)
- `Visualizing the history of the NBA <examples/nba-history.ipynb>`_
  (`interactive version
  <https://colab.research.google.com/github/lucasmaystre/kickscore/blob/master/examples/nba-history.ipynb>`__)

References
----------

- Lucas Maystre, Victor Kristof, Matthias Grossglauser,
  `Pairwise Comparisons with Flexible Time-Dynamics`_, KDD 2019


.. _Pairwise Comparisons with Flexible Time-Dynamics:
   https://arxiv.org/abs/1903.07746

.. |build-status| image:: https://travis-ci.org/lucasmaystre/kickscore.svg?branch=master
   :alt: build status
   :scale: 100%
   :target: https://travis-ci.org/lucasmaystre/kickscore

.. |coverage| image:: https://codecov.io/gh/lucasmaystre/kickscore/branch/master/graph/badge.svg
   :alt: code coverage
   :scale: 100%
   :target: https://codecov.io/gh/lucasmaystre/kickscore

.. |nba-history| image:: https://lum-public.s3-eu-west-1.amazonaws.com/kickscore-nba-history.svg
   :alt: evolution of NBA teams' skill over history
   :scale: 100%
