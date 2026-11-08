"""Ranking Studio: RLHF preference data built from existing galleries.

Modules:
* ``models`` — SQLAlchemy tables (``ranking_hands``, ``image_embeddings``,
  ``ranking_state``) registered on the shared gallery ``Base``.
* ``store``  — ``RankingStore`` async persistence service.

Wiring into ``main.py`` / API routes happens in a later task; the model
module is imported by ``backend.gallery.db.init_db()`` so the tables are
created at startup.
"""
