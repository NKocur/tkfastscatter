"""pytest configuration for tkfastscatter tests."""
# No special hooks needed: test_smoke.py uses pytest.importorskip so the
# whole module skips gracefully when the Rust extension has not been built.
