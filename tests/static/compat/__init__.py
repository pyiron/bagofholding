"""
Frozen object definitions for load-compatibility artefacts.

Each module `vX_Y_Z` holds the round-trip cases supported since bagofholding
X.Y.Z, and its artefacts in `artefacts/` were saved by that version. The
artefacts reference these objects by import path, so once version X.Y.Z is
released, everything the module defines or imports is frozen: any edit can
silently break loading them. Add a new module instead.

A module for a not-yet-released version (generated from a dev build) may still
be extended, as long as its artefacts are regenerated alongside.

Every round-trip case must live in a compat module, so new cases get a
compatibility test from the start.
"""
