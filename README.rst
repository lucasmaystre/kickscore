kickscore
=========

|build-status| |coverage|

``kickscore`` is a dynamic skill rating system, similar to the `Elo rating
system <https://en.wikipedia.org/wiki/Elo_rating_system>`_ and to `TrueSkill
<https://en.wikipedia.org/wiki/TrueSkill>`_. It comes as a Python 3 library.

In short, ``kickscore`` can be used to understand & visualize the skill of
players competing in pairwise matches, and to predict outcomes of future
matches.

Getting started
---------------

To install the latest release directly from PyPI, simply type::

    pip install kickscore

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
