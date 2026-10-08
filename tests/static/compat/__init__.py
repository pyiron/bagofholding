"""
Frozen object definitions for load-compatibility artefacts.

Once a module here has artefacts in `artefacts/`, everything it defines or
imports is frozen: the artefacts reference these objects by import path, so any
edit can silently break loading them. Add a new module instead.
"""
