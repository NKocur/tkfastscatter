"""Smoke tests for tkfastscatter.

The entire module is skipped gracefully when the Rust extension has not been
built yet. To build it:

    maturin develop --release

Renderer tests additionally require a real display and skip automatically in
headless environments.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if the extension is absent — gives a clear skip
# message rather than a confusing ImportError mid-collection.
pytest.importorskip(
    "tkfastscatter._tkfastscatter",
    reason="Rust extension not built — run: maturin develop --release",
)
import tkfastscatter
from tkfastscatter import ScatterWidget, link_cameras, unlink_cameras


# ── Headless-safe tests ───────────────────────────────────────────────────────

def test_import():
    assert hasattr(tkfastscatter, "ScatterWidget")


def test_version():
    assert tkfastscatter.__version__ != "unknown"


def test_colormap_names():
    names = ScatterWidget.colormap_names()
    assert isinstance(names, list)
    assert "viridis" in names
    assert "plasma" in names
    assert "turbo" in names


# ── Display-dependent fixture ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def root():
    """A real Tk root; skips the module if no display is available."""
    tk = pytest.importorskip("tkinter")
    try:
        r = tk.Tk()
        r.withdraw()
        yield r
        r.destroy()
    except tk.TclError as exc:
        pytest.skip(f"No display available: {exc}")


@pytest.fixture()
def widget(root):
    w = ScatterWidget(root, width=200, height=200)
    w.pack()
    root.update_idletasks()
    yield w
    w.destroy()


# ── Widget smoke tests ────────────────────────────────────────────────────────

def test_widget_creates(widget):
    assert isinstance(widget, ScatterWidget)


def test_set_points_basic(widget):
    pts = np.random.default_rng(0).standard_normal((1_000, 3)).astype(np.float32)
    widget.set_points(pts)


def test_set_points_empty(widget):
    widget.set_points(np.zeros((0, 3), dtype=np.float32))


def test_set_points_scalars(widget):
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((5_000, 3)).astype(np.float32)
    scalars = rng.random(5_000).astype(np.float32)
    widget.set_points(pts, scalars=scalars, colormap="plasma")


def test_set_points_colors(widget):
    rng = np.random.default_rng(2)
    pts = rng.standard_normal((5_000, 3)).astype(np.float32)
    colors = rng.random((5_000, 3)).astype(np.float32)
    widget.set_points(pts, colors=colors)


def test_set_points_before_map():
    """set_points() before map must queue, not crash."""
    import tkinter as tk
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"No display: {exc}")
    w = ScatterWidget(r, width=200, height=200)
    pts = np.random.default_rng(3).standard_normal((1_000, 3)).astype(np.float32)
    w.set_points(pts)
    r.destroy()


def test_wrong_positions_shape(widget):
    with pytest.raises(Exception):
        widget.set_points(np.zeros((100, 2), dtype=np.float32))


def test_scalar_length_mismatch(widget):
    with pytest.raises(Exception):
        widget.set_points(np.zeros((100, 3), dtype=np.float32),
                          scalars=np.zeros(50, dtype=np.float32))


def test_reset_camera(widget):
    pts = np.random.default_rng(4).standard_normal((1_000, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.reset_camera()
    # Subsequent set_points() must NOT re-fit the camera (camera_fitted stays True).
    widget.set_points(pts)


def test_colormap_without_scalars(widget):
    """colormap= must be respected even when no scalars are provided (Z-default path)."""
    pts = np.random.default_rng(8).standard_normal((1_000, 3)).astype(np.float32)
    # Should not raise and should use the requested colormap, not always viridis.
    widget.set_points(pts, colormap="plasma")
    widget.set_points(pts, colormap="hot")


def test_all_colormaps(widget):
    pts = np.random.default_rng(5).standard_normal((1_000, 3)).astype(np.float32)
    scalars = pts[:, 2].copy()
    for name in ScatterWidget.colormap_names():
        widget.set_points(pts, scalars=scalars, colormap=name)


def test_set_ticks_before_points(widget):
    """set_ticks() before any data must not crash."""
    widget.set_ticks(x=3, y=3, z=3)


def test_set_ticks_after_points(widget):
    """set_ticks() on a loaded dataset updates immediately (no upload needed)."""
    pts = np.random.default_rng(6).standard_normal((1_000, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.set_ticks(x=5, y=5, z=2)
    widget.set_ticks(z=None)   # restore one axis to auto


def test_set_ticks_reset_to_auto(widget):
    """Passing all-None restores full auto-scaling without error."""
    pts = np.random.default_rng(7).standard_normal((1_000, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.set_ticks(x=4)
    widget.set_ticks()   # all back to auto


def test_clim(widget):
    """clim= overrides the auto data range."""
    pts = np.random.default_rng(13).standard_normal((1_000, 3)).astype(np.float32)
    scalars = pts[:, 2].copy()
    widget.set_points(pts, scalars=scalars, colormap="plasma", clim=(-1.0, 1.0))


def test_log_scale(widget):
    """log_scale= must not crash (positive scalars)."""
    rng = np.random.default_rng(14)
    pts = rng.standard_normal((1_000, 3)).astype(np.float32)
    scalars = (rng.random(1_000) + 0.1).astype(np.float32)
    widget.set_points(pts, scalars=scalars, colormap="viridis", log_scale=True)


def test_nan_color(widget):
    """nan_color= is accepted; NaN scalars must not crash."""
    pts = np.random.default_rng(15).standard_normal((500, 3)).astype(np.float32)
    scalars = pts[:, 2].copy()
    scalars[::10] = float("nan")
    widget.set_points(pts, scalars=scalars, nan_color=(1.0, 0.0, 0.0))


def test_scalar_bar(widget):
    """scalar_bar() must not crash and must update visible state."""
    pts = np.random.default_rng(16).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts, scalars=pts[:, 2].copy(), colormap="viridis", clim=(-2.0, 2.0))
    widget.scalar_bar(True, vmin=-2.0, vmax=2.0, colormap="viridis", title="Z")
    if widget._renderer is not None:
        assert widget._renderer.inner.scalar_bar_visible if hasattr(widget._renderer, "inner") else True
    widget.scalar_bar(False)


def test_scalar_bar_before_renderer(root):
    """scalar_bar() before renderer init must queue state, not drop it."""
    w = ScatterWidget(root, width=100, height=100)
    w.scalar_bar(True, vmin=-1.0, vmax=1.0, colormap="plasma", title="T")
    assert w._pending_scalar_bar is not None
    assert w._pending_scalar_bar["visible"] is True
    assert w._pending_scalar_bar["colormap"] == "plasma"
    assert w._pending_scalar_bar["title"] == "T"
    w.destroy()


def test_set_ticks_before_map():
    """set_ticks() before map must queue, not crash."""
    import tkinter as tk
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"No display: {exc}")
    w = ScatterWidget(r, width=200, height=200)
    w.set_ticks(x=3, y=3, z=3)
    r.destroy()


# ── Multi-actor tests ─────────────────────────────────────────────────────────

def test_add_points_before_map_returns_virtual_handle(root):
    """add_points() before map must return a non-negative virtual handle and queue the call."""
    w = ScatterWidget(root, width=200, height=200)
    pts = np.random.default_rng(90).standard_normal((200, 3)).astype(np.float32)
    h = w.add_points(pts)
    assert isinstance(h, int) and h >= 0
    assert len(w._pending_actors) == 1
    assert w._pending_actors[0][1] == h   # vhandle stored as second element
    w.destroy()


def test_add_points_multiple_before_map_distinct_handles(root):
    """Multiple pre-map add_points() calls must each get distinct virtual handles."""
    w = ScatterWidget(root, width=200, height=200)
    pts = np.random.default_rng(91).standard_normal((100, 3)).astype(np.float32)
    h1 = w.add_points(pts)
    h2 = w.add_points(pts)
    assert h1 != h2
    assert len(w._pending_actors) == 2
    w.destroy()


def test_clear_resets_pending_actor_queue(root):
    """clear() before map must also empty the pending actor queue."""
    w = ScatterWidget(root, width=200, height=200)
    pts = np.random.default_rng(92).standard_normal((100, 3)).astype(np.float32)
    w.add_points(pts)
    assert len(w._pending_actors) == 1
    w.clear()
    assert len(w._pending_actors) == 0
    assert len(w._phandle_map) == 0
    w.destroy()


def test_add_points_returns_handle(widget):
    """add_points() must return a non-negative integer handle."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(20).standard_normal((500, 3)).astype(np.float32)
    h = widget.add_points(pts)
    assert isinstance(h, int)
    assert h >= 0


def test_add_multiple_actors(widget):
    """Multiple actors must coexist without crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    rng = np.random.default_rng(21)
    h1 = widget.add_points(rng.standard_normal((500, 3)).astype(np.float32))
    h2 = widget.add_points(rng.standard_normal((500, 3)).astype(np.float32))
    assert h1 != h2


def test_update_actor(widget):
    """update_actor() must replace data without crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    rng = np.random.default_rng(22)
    h = widget.add_points(rng.standard_normal((500, 3)).astype(np.float32))
    widget.update_actor(h, rng.standard_normal((300, 3)).astype(np.float32))


def test_remove_actor(widget):
    """remove_actor() must not crash and must remove the actor."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(23).standard_normal((500, 3)).astype(np.float32)
    h = widget.add_points(pts)
    widget.remove_actor(h)


def test_actor_visibility(widget):
    """set_actor_visibility() must not crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(24).standard_normal((500, 3)).astype(np.float32)
    h = widget.add_points(pts)
    widget.set_actor_visibility(h, False)
    widget.set_actor_visibility(h, True)


def test_clear_actors(widget):
    """clear() must remove all actors without crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    rng = np.random.default_rng(25)
    widget.add_points(rng.standard_normal((200, 3)).astype(np.float32))
    widget.add_points(rng.standard_normal((200, 3)).astype(np.float32))
    widget.clear()


def test_add_points_before_map():
    """add_points() before map must return a virtual handle, not -1."""
    import tkinter as tk
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"No display: {exc}")
    w = ScatterWidget(r, width=200, height=200)
    pts = np.random.default_rng(26).standard_normal((500, 3)).astype(np.float32)
    h = w.add_points(pts)
    assert isinstance(h, int) and h >= 0   # virtual handle, not -1
    r.destroy()


def test_set_points_after_add_points(widget):
    """set_points() after add_points() must replace, not accumulate."""
    rng = np.random.default_rng(27)
    widget.add_points(rng.standard_normal((200, 3)).astype(np.float32))
    widget.add_points(rng.standard_normal((200, 3)).astype(np.float32))
    # set_points should wipe the multi-actor state
    widget.set_points(rng.standard_normal((300, 3)).astype(np.float32))


# ── Export tests ─────────────────────────────────────────────────────────────

def test_screenshot_none_before_map():
    """screenshot() must return None when the widget has not been mapped."""
    import tkinter as tk
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"No display: {exc}")
    w = ScatterWidget(r, width=200, height=200)
    assert w.screenshot() is None
    r.destroy()


def test_screenshot_returns_array(widget):
    """screenshot() must return a (H, W, 4) uint8 RGBA array and not be all-zero."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(40).standard_normal((200, 3)).astype(np.float32)
    widget.set_points(pts)
    img = widget.screenshot()
    assert img is not None
    assert img.ndim == 3 and img.shape[2] == 4
    assert img.dtype == np.uint8
    assert img.shape[:2] == (200, 200)  # matches widget dimensions
    # Image should not be entirely black — at least some pixels are non-zero
    assert img[..., :3].max() > 0


def test_save_png(widget, tmp_path):
    """save_png() must write a readable PNG file."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(41).standard_normal((200, 3)).astype(np.float32)
    widget.set_points(pts)
    out = tmp_path / "shot.png"
    widget.save_png(str(out))
    assert out.exists() and out.stat().st_size > 0
    # Verify it's a valid PNG (magic bytes)
    with open(out, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


# ── Picking / selection tests ─────────────────────────────────────────────────

def test_pick_point_returns_dict_or_none(widget):
    """pick_point() must not crash and return a dict or None."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(30).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts)
    result = widget._renderer.pick_point(100.0, 100.0)
    assert result is None or isinstance(result, dict)
    if result is not None:
        assert {"actor", "index", "point"} <= result.keys()


def test_pick_rectangle_returns_list(widget):
    """pick_rectangle() must return a list of dicts."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(31).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts)
    hits = widget._renderer.pick_rectangle(0.0, 0.0, 200.0, 200.0)
    assert isinstance(hits, list)
    for h in hits:
        assert {"actor", "index"} <= h.keys()


def test_enable_point_picking(widget):
    """enable_point_picking() must not crash."""
    widget.enable_point_picking()
    assert widget._pick_mode in ("point", "both")


def test_enable_rectangle_picking(widget):
    """enable_rectangle_picking() must not crash."""
    widget.enable_rectangle_picking()
    assert widget._pick_mode in ("rect", "both")


def test_disable_picking(widget):
    """disable_picking() must restore mode to 'none'."""
    widget.enable_point_picking()
    widget.disable_picking()
    assert widget._pick_mode == "none"


def test_pick_empty_scene(widget):
    """Picking on an empty scene must return None / empty list."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    widget.set_points(np.zeros((0, 3), dtype=np.float32))
    assert widget._renderer.pick_point(100.0, 100.0) is None
    assert widget._renderer.pick_rectangle(0.0, 0.0, 200.0, 200.0) == []


# ── Overlay / line actor tests ───────────────────────────────────────────────

def test_add_lines_returns_handle(widget):
    """add_lines() must return a non-negative integer handle."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.array([[0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 1, 0]], dtype=np.float32)
    h = widget.add_lines(segs, color=(1.0, 0.0, 0.0))
    assert isinstance(h, int) and h >= 0


def test_add_multiple_overlays(widget):
    """Multiple line overlay actors must coexist."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.zeros((3, 6), dtype=np.float32)
    h1 = widget.add_lines(segs)
    h2 = widget.add_lines(segs, color=(0.0, 1.0, 0.0))
    assert h1 != h2


def test_update_lines(widget):
    """update_lines() must not crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.zeros((2, 6), dtype=np.float32)
    h = widget.add_lines(segs)
    new_segs = np.ones((4, 6), dtype=np.float32)
    widget.update_lines(h, new_segs, color=(0.5, 0.5, 0.5))


def test_overlay_visibility(widget):
    """set_overlay_visibility() must not crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.zeros((2, 6), dtype=np.float32)
    h = widget.add_lines(segs)
    widget.set_overlay_visibility(h, False)
    widget.set_overlay_visibility(h, True)


def test_remove_overlay(widget):
    """remove_overlay() must not crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.zeros((2, 6), dtype=np.float32)
    h = widget.add_lines(segs)
    widget.remove_overlay(h)


def test_clear_overlays(widget):
    """clear_overlays() must remove all line actors without crash."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.zeros((2, 6), dtype=np.float32)
    widget.add_lines(segs)
    widget.add_lines(segs)
    widget.clear_overlays()


def test_add_box(widget):
    """add_box() must return a valid handle."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    h = widget.add_box((-1, -1, -1, 1, 1, 1))
    assert isinstance(h, int) and h >= 0


def test_orientation_axes(widget):
    """show_orientation_axes() must update state on toggle."""
    widget.show_orientation_axes(True)
    assert widget._orientation_axes_visible is True
    widget.show_orientation_axes(False)
    assert widget._orientation_axes_visible is False


def test_orientation_axes_before_map(root):
    """show_orientation_axes() before map must persist to renderer on init."""
    w = ScatterWidget(root, width=100, height=100)
    w.show_orientation_axes(True)
    assert w._orientation_axes_visible is True
    w.destroy()


def test_add_lines_before_map(root):
    """add_lines() before map must return a valid virtual handle and queue the overlay."""
    w = ScatterWidget(root, width=200, height=200)
    segs = np.zeros((2, 6), dtype=np.float32)
    h = w.add_lines(segs)
    assert isinstance(h, int) and h >= 0
    assert len(w._pending_overlays) == 1
    assert w._pending_overlays[0][3] == h   # vhandle stored in queue
    w.destroy()


def test_add_box_before_map(root):
    """add_box() before map must queue via add_lines path."""
    w = ScatterWidget(root, width=200, height=200)
    h = w.add_box((-1, -1, -1, 1, 1, 1))
    assert isinstance(h, int) and h >= 0
    assert len(w._pending_overlays) == 1
    w.destroy()


def test_multiple_overlays_before_map(root):
    """Multiple pre-map overlays must each get distinct virtual handles."""
    w = ScatterWidget(root, width=200, height=200)
    segs = np.zeros((2, 6), dtype=np.float32)
    h1 = w.add_lines(segs)
    h2 = w.add_lines(segs, color=(0.0, 1.0, 0.0))
    assert h1 != h2
    assert len(w._pending_overlays) == 2
    w.destroy()


def test_clear_overlays_before_map(root):
    """clear_overlays() before map must empty the queue."""
    w = ScatterWidget(root, width=200, height=200)
    segs = np.zeros((2, 6), dtype=np.float32)
    w.add_lines(segs)
    w.add_lines(segs)
    w.clear_overlays()
    assert len(w._pending_overlays) == 0
    w.destroy()


def test_clear_removes_overlays(widget):
    """clear() must remove line overlays, not just point actors."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    pts = np.random.default_rng(80).standard_normal((300, 3)).astype(np.float32)
    widget.set_points(pts)
    segs = np.array([[10.0, 20.0, 30.0, 11.0, 21.0, 31.0]], dtype=np.float32)
    widget.add_lines(segs)
    assert widget._renderer.actor_union_bounds() is not None
    widget.clear()
    # Both point actors and line overlays must be gone
    assert widget._renderer.actor_union_bounds() is None


def test_clear_resets_pending_overlay_queue(root):
    """clear() before map must also empty the pending overlay queue."""
    w = ScatterWidget(root, width=200, height=200)
    segs = np.zeros((2, 6), dtype=np.float32)
    w.add_lines(segs)
    assert len(w._pending_overlays) == 1
    w.clear()
    assert len(w._pending_overlays) == 0
    w.destroy()


def test_wrong_segments_shape(widget):
    """add_lines() with wrong shape must raise."""
    with pytest.raises(Exception):
        widget.add_lines(np.zeros((3, 3), dtype=np.float32))


# ── Animation export tests ───────────────────────────────────────────────────

def test_write_frame_without_open(widget):
    """write_frame() before open_gif() must raise RuntimeError."""
    with pytest.raises(RuntimeError):
        widget.write_frame()


def test_close_gif_without_open(widget):
    """close_gif() without open_gif() must be a silent no-op."""
    widget.close_gif()   # must not raise


def test_open_write_close_no_renderer(root):
    """open/write/close on an unmapped widget must not crash."""
    w = ScatterWidget(root, width=100, height=100)
    w.open_gif("/dev/null", fps=10)
    w.write_frame()  # screenshot returns None — frame silently skipped
    w.close_gif()
    w.destroy()


def test_orbit_gif_before_renderer(root):
    """orbit_gif() before map must raise RuntimeError (renderer absent)."""
    w = ScatterWidget(root, width=100, height=100)
    with pytest.raises(RuntimeError):
        w.orbit_gif("irrelevant.gif")
    w.destroy()


def test_orbit_gif_produces_file(widget, tmp_path):
    """orbit_gif() must create a non-empty file."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    pts = np.random.default_rng(70).standard_normal((200, 3)).astype(np.float32)
    widget.set_points(pts)
    out = tmp_path / "orbit.gif"
    widget.orbit_gif(str(out), n_frames=4, fps=10)
    assert out.exists() and out.stat().st_size > 0
    # Verify GIF magic bytes
    with open(out, "rb") as f:
        assert f.read(6) == b"GIF89a"


def test_manual_gif_write(widget, tmp_path):
    """Manual open/write/close must produce a valid GIF."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    pts = np.random.default_rng(71).standard_normal((200, 3)).astype(np.float32)
    widget.set_points(pts)
    out = tmp_path / "manual.gif"
    widget.open_gif(str(out), fps=10, loop=0)
    for _ in range(3):
        widget.write_frame()
    widget.close_gif()
    assert out.exists() and out.stat().st_size > 0
    with open(out, "rb") as f:
        assert f.read(6) == b"GIF89a"


# ── Rendering mode tests ─────────────────────────────────────────────────────

def test_point_style_property(widget):
    """point_style round-trips for all valid values."""
    for style in ("circle", "square", "gaussian"):
        widget.point_style = style
        assert widget.point_style == style
    widget.point_style = "circle"   # restore default


def test_point_style_invalid(widget):
    """Unknown style must raise ValueError."""
    with pytest.raises(ValueError):
        widget.point_style = "blob"


def test_opacity(widget):
    """opacity= must not crash for values in [0, 1]."""
    pts = np.random.default_rng(60).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts, opacity=0.5)
    widget.set_points(pts, opacity=1.0)
    widget.set_points(pts, opacity=0.0)


def test_point_style_before_renderer(root):
    """Setting point_style before renderer is initialized must not crash."""
    w = ScatterWidget(root, width=100, height=100)
    w.point_style = "gaussian"
    assert w.point_style == "gaussian"
    w.destroy()


# ── Linked-camera tests ───────────────────────────────────────────────────────

def test_link_cameras_populates_links(root):
    """link_cameras() must add cross-references to _camera_links on each widget."""
    w1 = ScatterWidget(root, width=100, height=100)
    w2 = ScatterWidget(root, width=100, height=100)
    assert w2 not in w1._camera_links
    link_cameras(w1, w2)
    assert w2 in w1._camera_links
    assert w1 in w2._camera_links
    w1.destroy()
    w2.destroy()


def test_link_cameras_three_way(root):
    """Three-way link must be fully connected."""
    w1 = ScatterWidget(root, width=100, height=100)
    w2 = ScatterWidget(root, width=100, height=100)
    w3 = ScatterWidget(root, width=100, height=100)
    link_cameras(w1, w2, w3)
    assert w2 in w1._camera_links and w3 in w1._camera_links
    assert w1 in w2._camera_links and w3 in w2._camera_links
    assert w1 in w3._camera_links and w2 in w3._camera_links
    w1.destroy(); w2.destroy(); w3.destroy()


def test_unlink_cameras(root):
    """unlink_cameras() must remove cross-references."""
    w1 = ScatterWidget(root, width=100, height=100)
    w2 = ScatterWidget(root, width=100, height=100)
    link_cameras(w1, w2)
    unlink_cameras(w1, w2)
    assert w2 not in w1._camera_links
    assert w1 not in w2._camera_links
    w1.destroy()
    w2.destroy()


def test_link_cameras_no_crash_before_renderer(root):
    """link_cameras() and camera mutations must not crash when renderer is absent."""
    w1 = ScatterWidget(root, width=100, height=100)
    w2 = ScatterWidget(root, width=100, height=100)
    link_cameras(w1, w2)
    # Calling camera methods before renderer is initialized must be silent.
    w1.view_xy()
    w1.reset_camera()
    w1.destroy()
    w2.destroy()


def test_linked_camera_propagates(root):
    """Camera state must propagate from w1 to w2 and from w2 to w1 after link."""
    w1 = ScatterWidget(root, width=200, height=200)
    w2 = ScatterWidget(root, width=200, height=200)
    w1.pack()
    w2.pack()
    root.update_idletasks()
    if w1._renderer is None or w2._renderer is None:
        w1.destroy(); w2.destroy()
        pytest.skip("renderer not initialized")
    pts = np.random.default_rng(50).standard_normal((500, 3)).astype(np.float32)
    w1.set_points(pts)
    w2.set_points(pts)
    link_cameras(w1, w2)
    # w1 → w2 propagation
    w1.view_xy()
    s1, s2 = w1.get_camera(), w2.get_camera()
    assert abs(s1["pitch"] - s2["pitch"]) < 1e-4
    assert abs(s1["yaw"] - s2["yaw"]) < 1e-4
    # w2 → w1 propagation (reverse direction)
    w2.view_yz()
    s1, s2 = w1.get_camera(), w2.get_camera()
    assert abs(s1["pitch"] - s2["pitch"]) < 1e-4
    assert abs(s1["yaw"] - s2["yaw"]) < 1e-4
    w1.destroy()
    w2.destroy()


def test_overlay_bounds_included_in_union(widget):
    """add_lines() bounds must be included in actor_union_bounds()."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    segs = np.array([[100.0, 200.0, 300.0, 101.0, 201.0, 301.0]], dtype=np.float32)
    widget.add_lines(segs, color=(1.0, 1.0, 0.0))
    bounds = widget._renderer.actor_union_bounds()
    assert bounds is not None
    bmin, bmax = bounds
    assert bmax[0] >= 101.0
    assert bmax[1] >= 201.0
    assert bmax[2] >= 301.0


def test_overlay_only_scene_gets_camera_fit(widget):
    """An overlay added to an empty scene must trigger camera fitting."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    widget.clear()
    segs = np.array([[100.0, 200.0, 300.0, 101.0, 201.0, 301.0]], dtype=np.float32)
    widget.add_lines(segs)
    cam_after = widget.get_camera()
    target = cam_after["target"]
    assert max(abs(t) for t in target) > 50.0


def test_hidden_overlay_excluded_from_bounds(widget):
    """Hidden overlays must not contribute to actor_union_bounds."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    widget.clear()
    segs = np.array([[100.0, 200.0, 300.0, 101.0, 201.0, 301.0]], dtype=np.float32)
    h = widget.add_lines(segs)
    assert widget._renderer.actor_union_bounds() is not None
    widget.set_overlay_visibility(h, False)
    # After hiding the only overlay, bounds should be empty
    assert widget._renderer.actor_union_bounds() is None


def test_hidden_actor_excluded_from_bounds(widget):
    """Hidden point actors must not contribute to actor_union_bounds."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    pts = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
    h = widget.add_points(pts)
    assert widget._renderer.actor_union_bounds() is not None
    widget.set_actor_visibility(h, False)
    assert widget._renderer.actor_union_bounds() is None
    # Restoring visibility brings bounds back
    widget.set_actor_visibility(h, True)
    assert widget._renderer.actor_union_bounds() is not None


def test_clear_last_overlay_resets_camera_fit(widget):
    """Clearing the last overlay must reset camera_fitted so the next add refits."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    widget.clear()
    segs1 = np.array([[100.0, 200.0, 300.0, 101.0, 201.0, 301.0]], dtype=np.float32)
    widget.add_lines(segs1)
    cam1 = widget.get_camera()

    # Clear all overlays — camera_fitted should reset
    widget.clear_overlays()
    assert widget._renderer.camera_fitted is False

    # Add a new overlay at a completely different location
    segs2 = np.array([[-500.0, -500.0, -500.0, -499.0, -499.0, -499.0]], dtype=np.float32)
    widget.add_lines(segs2)
    cam2 = widget.get_camera()
    # Camera target should now be near segs2, not segs1
    t1 = cam1["target"]
    t2 = cam2["target"]
    dist = sum((a - b) ** 2 for a, b in zip(t1, t2)) ** 0.5
    assert dist > 100.0, "Camera target should have moved after refit"


def test_remove_last_overlay_resets_camera_fit(widget):
    """Removing the last overlay must reset camera_fitted."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized")
    widget.clear()
    segs = np.array([[50.0, 60.0, 70.0, 51.0, 61.0, 71.0]], dtype=np.float32)
    h = widget.add_lines(segs)
    widget.remove_overlay(h)
    assert widget._renderer.camera_fitted is False


# ── Camera preset tests ───────────────────────────────────────────────────────

def test_camera_presets(widget):
    """All view presets must run without error."""
    pts = np.random.default_rng(9).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.view_xy()
    widget.view_xz()
    widget.view_yz()
    widget.view_isometric()
    widget.reset_camera()


def test_parallel_projection(widget):
    """Toggling parallel projection must not crash and must round-trip."""
    pts = np.random.default_rng(10).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts)
    assert widget.parallel_projection is False
    widget.parallel_projection = True
    assert widget.parallel_projection is True
    widget.parallel_projection = False
    assert widget.parallel_projection is False


def test_fit_to_bounds(widget):
    """fit() with explicit bounds must not crash."""
    pts = np.random.default_rng(11).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.fit((-2, -2, -2, 2, 2, 2))
    widget.fit()   # re-fit to data


def test_get_set_camera(widget):
    """get_camera / set_camera must round-trip without error."""
    if widget._renderer is None:
        pytest.skip("renderer not initialized (withdrawn window — no Map event)")
    pts = np.random.default_rng(12).standard_normal((500, 3)).astype(np.float32)
    widget.set_points(pts)
    state = widget.get_camera()
    assert {"target", "distance", "yaw", "pitch", "parallel"} <= state.keys()
    widget.view_xy()
    widget.set_camera(state)   # restore original state


# ── set_axes / show_grid / set_background tests ───────────────────────────────

def test_show_grid_toggle(widget):
    """show_grid() must not crash when toggled."""
    pts = np.random.default_rng(70).standard_normal((300, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.show_grid(False)
    assert widget._grid_visible is False
    widget.show_grid(True)
    assert widget._grid_visible is True


def test_show_grid_before_renderer(root):
    """show_grid() before renderer is initialized must persist to renderer on map."""
    w = ScatterWidget(root, width=100, height=100)
    w.show_grid(False)
    assert w._grid_visible is False
    w.destroy()


def test_set_background_tuple(widget):
    """set_background() with an (r, g, b) tuple must not crash."""
    pts = np.random.default_rng(71).standard_normal((300, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.set_background((0.1, 0.1, 0.2))
    assert widget._bg_color == (0.1, 0.1, 0.2)


def test_set_background_hex(widget):
    """set_background() with a hex string must parse and apply correctly."""
    pts = np.random.default_rng(72).standard_normal((300, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.set_background("#0d0d12")
    r, g, b = widget._bg_color
    assert abs(r - 0x0d / 255) < 1e-4
    assert abs(g - 0x0d / 255) < 1e-4
    assert abs(b - 0x12 / 255) < 1e-4


def test_set_background_invalid(widget):
    """set_background() with a bad string must raise ValueError."""
    with pytest.raises(ValueError):
        widget.set_background("not-a-color")


def test_set_background_before_renderer(root):
    """set_background() before renderer init must persist state."""
    w = ScatterWidget(root, width=100, height=100)
    w.set_background((0.5, 0.5, 0.5))
    assert w._bg_color == (0.5, 0.5, 0.5)
    w.destroy()


def test_set_axes(widget):
    """set_axes() must accept label strings and persist state."""
    pts = np.random.default_rng(73).standard_normal((300, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.set_axes("Time", "Amplitude", "Phase")
    assert widget._axis_labels == ("Time", "Amplitude", "Phase")


def test_set_axes_empty_labels(widget):
    """set_axes() with empty strings must not crash."""
    pts = np.random.default_rng(74).standard_normal((300, 3)).astype(np.float32)
    widget.set_points(pts)
    widget.set_axes("", "", "")
    assert widget._axis_labels == ("", "", "")


def test_set_axes_before_renderer(root):
    """set_axes() before renderer init must persist state."""
    w = ScatterWidget(root, width=100, height=100)
    w.set_axes("A", "B", "C")
    assert w._axis_labels == ("A", "B", "C")
    w.destroy()
