"""Seismogenic patch identification and catalog declustering.

Declustering is a nearest-neighbour separation of background events.
The clustering stage is an alpha-filtration of the DPS density: a scan
over the exponent ``q``, a condensed tree, and persistent cores.
``q`` is chosen by the user from the diagnostics. The package does not
claim an automatic best ``q``.

Run ``python -m spad.cli --help`` from the repository root.
"""
