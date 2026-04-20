"""Tkinter widget that wraps the Rust/wgpu ScatterRenderer."""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from typing import Optional

import numpy as np

from ._tkfastscatter import ScatterRenderer


def _gif_lzw_compress(indices: "list[int]", min_code_size: int = 8) -> bytes:
    """GIF LZW compression.  Returns bytes in GIF sub-block format."""
    clear = 1 << min_code_size
    eoi = clear + 1

    table: "dict[tuple, int]" = {(i,): i for i in range(clear)}
    next_code = eoi + 1
    code_size = min_code_size + 1

    pairs: "list[tuple[int, int]]" = [(clear, code_size)]

    if indices:
        prefix: "tuple[int, ...]" = (indices[0],)
        for sym in indices[1:]:
            ext = prefix + (sym,)
            if ext in table:
                prefix = ext
            else:
                pairs.append((table[prefix], code_size))
                if next_code < 4096:
                    table[ext] = next_code
                    next_code += 1
                    if next_code > (1 << code_size) and code_size < 12:
                        code_size += 1
                else:
                    pairs.append((clear, code_size))
                    table = {(i,): i for i in range(clear)}
                    next_code = eoi + 1
                    code_size = min_code_size + 1
                prefix = (sym,)
        pairs.append((table[prefix], code_size))

    pairs.append((eoi, code_size))

    # Pack codes LSB-first into bytes
    raw = bytearray()
    acc = acc_bits = 0
    for code, nbits in pairs:
        acc |= code << acc_bits
        acc_bits += nbits
        while acc_bits >= 8:
            raw.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        raw.append(acc & 0xFF)

    # Wrap in GIF sub-blocks (max 255 bytes each)
    result = bytearray([min_code_size])
    i = 0
    while i < len(raw):
        block = raw[i : i + 255]
        result.append(len(block))
        result.extend(block)
        i += 255
    result.append(0)
    return bytes(result)


def _write_gif_stdlib(
    path: str,
    frames: "list[np.ndarray]",
    fps: int,
    loop: int,
) -> None:
    """Write an animated GIF using only stdlib + numpy (3-3-2 colour quantisation)."""
    import struct

    if not frames:
        return
    h, w = frames[0].shape[:2]
    delay = max(2, round(100 / fps))  # centiseconds

    # Build 3-3-2 palette: index = (r3 << 5) | (g3 << 2) | b2
    pal = bytearray(256 * 3)
    for idx in range(256):
        r3 = (idx >> 5) & 7
        g3 = (idx >> 2) & 7
        b2 = idx & 3
        pal[idx * 3]     = (r3 * 255) // 7 if r3 else 0
        pal[idx * 3 + 1] = (g3 * 255) // 7 if g3 else 0
        pal[idx * 3 + 2] = (b2 * 255) // 3 if b2 else 0

    buf = bytearray()
    buf += b"GIF89a"
    buf += struct.pack("<HH", w, h)
    buf += bytes([0xF7, 0, 0])  # global CT=1, 256 colours, bg=0, aspect=0
    buf += pal

    # Netscape loop extension
    buf += b"\x21\xFF\x0BNETSCAPE2.0\x03\x01"
    buf += struct.pack("<HB", loop & 0xFFFF, 0)

    for rgba in frames:
        r = rgba[:, :, 0].astype(np.uint16)
        g = rgba[:, :, 1].astype(np.uint16)
        b = rgba[:, :, 2].astype(np.uint16)
        qi = (((r >> 5) << 5) | ((g >> 5) << 2) | (b >> 6)).astype(np.uint8)
        indices = qi.flatten().tolist()

        # Graphic Control Extension
        buf += b"\x21\xF9\x04\x00"
        buf += struct.pack("<HB", delay, 0)

        # Image Descriptor
        buf += b"\x2C"
        buf += struct.pack("<HHHHB", 0, 0, w, h, 0)

        buf += _gif_lzw_compress(indices, 8)

    buf += b"\x3B"
    with open(path, "wb") as f:
        f.write(buf)


def _write_png(path: str, rgba: "np.ndarray") -> None:
    """Minimal PNG writer using only stdlib — no Pillow required."""
    import struct, zlib
    h, w = rgba.shape[:2]

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    # IHDR: width, height, bit_depth=8, color_type=6 (RGBA), compression=0, filter=0, interlace=0
    ihdr = chunk(b"IHDR", struct.pack(">II", w, h) + bytes([8, 6, 0, 0, 0]))
    # IDAT: filter byte 0 (None) prepended to every scanline
    raw_rows = b"".join(b"\x00" + bytes(row.tobytes()) for row in rgba)
    idat = chunk(b"IDAT", zlib.compress(raw_rows, 6))
    iend = chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend)


def _get_display_id() -> int:
    if sys.platform != "linux":
        return 0
    try:
        xlib = ctypes.cdll.LoadLibrary("libX11.so.6")
        xlib.XOpenDisplay.restype = ctypes.c_void_p
        ptr = xlib.XOpenDisplay(None)
        return int(ptr) if ptr else 0
    except OSError:
        return 0


def _platform_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


_DISPLAY_ID: int = _get_display_id()


class ScatterWidget(tk.Frame):
    """A Tkinter widget that renders a 3-D scatter plot using wgpu (Rust).

    Usage
    -----
    ::

        import tkinter as tk
        import numpy as np
        from tkfastscatter import ScatterWidget

        root = tk.Tk()
        w = ScatterWidget(root, width=800, height=600)
        w.pack(fill="both", expand=True)

        pts = np.random.rand(250_000, 3).astype(np.float32)
        w.set_points(pts)

        root.mainloop()
    """

    def __init__(
        self,
        master: tk.Misc,
        width: int = 800,
        height: int = 600,
        fps: int = 60,
        vsync: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(master, width=width, height=height, **kwargs)

        self._fps = fps
        self._vsync = vsync
        self._renderer: Optional[ScatterRenderer] = None

        # Pending data: set before the renderer is ready
        self._pending: Optional[dict] = None
        self._pending_actors: list = []   # queued add_points() calls: (kwargs, vhandle)
        self._next_phandle: int = 0
        self._phandle_map: "dict[int, int]" = {}   # virtual → real actor handle
        self._pending_ticks: Optional[tuple] = None
        # Python-side shadow so parallel_projection is readable before renderer init
        self._parallel_projection: bool = False

        # Dirty-frame model: only call render() when something changed
        self._dirty: bool = False

        # Drag state
        self._drag_btn: Optional[int] = None
        self._drag_x: int = 0
        self._drag_y: int = 0

        # Picking state
        self._pick_mode: str = "none"   # "none" | "point" | "rect" | "both"
        self._press_x: int = 0          # ButtonPress coords, for click vs drag
        self._press_y: int = 0
        self._sel_x0: int = 0           # rectangle start (screen coords)
        self._sel_y0: int = 0
        self._rect_active: bool = False  # Shift+drag rectangle in progress
        self._pick_threshold: int = 5   # px — less than this = click, not drag

        # Public result attributes — read after virtual events
        self.picked_point: "list[float] | None" = None
        self.picked_index: "int | None" = None
        self.picked_actor: "int | None" = None
        self.selected: "list[dict] | None" = None  # [{"actor":int,"index":int},...]

        # Linked-camera state
        self._camera_links: "set[ScatterWidget]" = set()
        self._propagating: bool = False  # re-entrancy guard

        # Animation recording state
        self._gif_frames: "list | None" = None
        self._gif_path: "str | None" = None
        self._gif_fps: int = 20
        self._gif_loop: int = 0

        # Rendering modes
        self._point_style: str = "circle"
        self._lod_enabled: bool = True
        self._lod_threshold: int = 200_000  # activate LOD above this many total points
        self._lod_factor: int = 8           # draw 1-in-8 points during interaction
        self._total_n: int = 0              # running total of uploaded points

        # Visual appearance
        self._grid_visible: bool = True
        self._bg_color: tuple = (0.05, 0.05, 0.07)
        self._axis_labels: tuple = ("X", "Y", "Z")

        # Pre-map overlay queue
        self._orientation_axes_visible: bool = False
        self._pending_scalar_bar: "dict | None" = None
        self._pending_overlays: "list[tuple]" = []  # (method, segments, color, vhandle)
        self._next_vhandle: int = 0
        self._vhandle_map: "dict[int, int]" = {}   # virtual → real handle

        self._after_id: Optional[str] = None
        self._resize_after_id: Optional[str] = None

        self.pack_propagate(False)
        self.grid_propagate(False)

        self.bind("<Map>", self._on_map, add="+")
        self.bind("<Configure>", self._on_configure, add="+")

        self.bind("<ButtonPress-1>", lambda e: self._drag_start(e, 1))
        self.bind("<ButtonPress-2>", lambda e: self._drag_start(e, 2))
        self.bind("<B1-Motion>", lambda e: self._drag_move(e, 1))
        self.bind("<B2-Motion>", lambda e: self._drag_move(e, 2))
        self.bind("<ButtonRelease-1>", self._drag_end)
        self.bind("<ButtonRelease-2>", self._drag_end)
        self.bind("<MouseWheel>", self._on_scroll)
        self.bind("<Button-4>", self._on_scroll_up_x11)
        self.bind("<Button-5>", self._on_scroll_down_x11)
        self.bind("<Double-Button-1>", lambda _e: self.reset_camera())

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_map(self, _event: tk.Event) -> None:
        if self._renderer is not None:
            return
        self.update_idletasks()
        self._init_renderer()

    def _init_renderer(self) -> None:
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        try:
            self._renderer = ScatterRenderer(
                self.winfo_id(),
                _DISPLAY_ID,
                w,
                h,
                _platform_name(),
                self._vsync,
            )
        except Exception as exc:
            import warnings
            warnings.warn(f"tkfastscatter: renderer init failed: {exc}", stacklevel=2)
            return

        # Apply any pre-renderer state that was set before the window was mapped.
        if self._parallel_projection:
            self._renderer.set_parallel_projection(True)
        if self._point_style != "circle":
            self._renderer.set_point_style(self._STYLE_MAP[self._point_style])
        if not self._grid_visible:
            self._renderer.show_grid(False)
        if self._bg_color != (0.05, 0.05, 0.07):
            self._renderer.set_background_color(*self._bg_color)
        if self._axis_labels != ("X", "Y", "Z"):
            self._renderer.set_axis_labels(*self._axis_labels)
        if self._orientation_axes_visible:
            self._renderer.show_orientation_axes(True)
        if self._pending_scalar_bar is not None:
            sb = self._pending_scalar_bar
            self._pending_scalar_bar = None
            self._renderer.show_scalar_bar(
                sb["visible"], sb["vmin"], sb["vmax"],
                sb["log_scale"], sb["colormap"], sb["title"],
            )
        for _method, segments, color, vhandle in self._pending_overlays:
            real = int(self._renderer.add_lines(segments, color))
            self._vhandle_map[vhandle] = real
        self._pending_overlays.clear()
        if self._pending_ticks is not None:
            x, y, z = self._pending_ticks
            self._pending_ticks = None
            self._renderer.set_ticks(x=x, y=y, z=z)
        did_something = False
        if self._pending is not None:
            pending = self._pending
            self._pending = None
            self.set_points(**pending)   # calls _mark_dirty internally
            did_something = True
        if self._pending_actors:
            for kwargs, vhandle in self._pending_actors:
                real = int(self._renderer.add_points(**kwargs))
                if real != 0xFFFFFFFF:   # u32::MAX means empty dataset
                    self._phandle_map[vhandle] = real
            self._pending_actors.clear()
            self._mark_dirty()
            did_something = True
        if not did_something:
            self._schedule_render()

    def _on_configure(self, event: tk.Event) -> None:
        if self._renderer is None:
            return
        # Debounce: resize is expensive — only execute 50 ms after the last event
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(
            50, lambda: self._do_resize(event.width, event.height)
        )

    def _do_resize(self, w: int, h: int) -> None:
        self._resize_after_id = None
        if self._renderer is not None:
            self._renderer.resize(max(w, 1), max(h, 1))
            self._mark_dirty()

    def destroy(self) -> None:
        for id_ in (self._after_id, self._resize_after_id):
            if id_ is not None:
                self.after_cancel(id_)
        self._renderer = None
        super().destroy()

    # ── Render loop ───────────────────────────────────────────────────────────

    def _schedule_render(self) -> None:
        interval = max(1, 1000 // self._fps)
        self._after_id = self.after(interval, self._render_tick)

    def _mark_dirty(self) -> None:
        """Mark a redraw needed; restarts the timer if it was stopped."""
        self._dirty = True
        if self._after_id is None and self._renderer is not None:
            self._schedule_render()

    def _render_tick(self) -> None:
        self._after_id = None  # cleared first so _mark_dirty can re-arm
        if self._renderer is not None and self._dirty:
            try:
                self._renderer.render()
                self._dirty = False  # only clear on success; error keeps dirty for retry
            except Exception:
                pass
        # Re-arm only while there is work to do; stops firing when idle
        if self._dirty and self._after_id is None:
            self._schedule_render()

    # ── Data API ──────────────────────────────────────────────────────────────

    def set_points(
        self,
        positions: "np.ndarray",
        *,
        colors: Optional["np.ndarray"] = None,
        scalars: Optional["np.ndarray"] = None,
        colormap: str = "viridis",
        point_size: float = 4.0,
        clim: "tuple[float, float] | None" = None,
        nan_color: "tuple[float, float, float] | None" = None,
        log_scale: bool = False,
        opacity: float = 1.0,
    ) -> None:
        """Replace all point actors with a single new point cloud.

        Line overlays added via :meth:`add_lines` or :meth:`add_box` are
        **not** affected. Call :meth:`clear` first if you want to reset the
        entire scene including overlays.

        Parameters
        ----------
        positions : (N, 3) float32 array
            XYZ coordinates.
        colors : (N, 3) float32 array, optional
            Per-point RGB in [0, 1]. Highest priority; ignores *scalars* and
            *colormap* when provided.
        scalars : (N,) float32 array, optional
            Per-point scalar values mapped through *colormap*. Used when
            *colors* is not provided.
        colormap : str
            Colormap applied to *scalars*, or to the Z coordinate when neither
            *colors* nor *scalars* are given. Available names:
            ``viridis``, ``plasma``, ``inferno``, ``magma``, ``coolwarm``,
            ``hot``, ``gray``, ``turbo``, ``cividis``, ``blues``, ``greens``,
            ``reds``. Defaults to ``"viridis"``.
        point_size : float
            Point diameter in pixels.
        clim : (vmin, vmax), optional
            Fix the colormap range. Values outside the range are clamped.
            Defaults to the data min/max.
        nan_color : (r, g, b), optional
            RGB color in [0, 1] for NaN / non-finite scalars. Defaults to
            a neutral gray ``(0.4, 0.4, 0.4)``.
        log_scale : bool
            Apply logarithmic normalization before colormap sampling.
        """
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must be shape (N, 3), got {pos.shape}")
        n = pos.shape[0]

        clr = np.ascontiguousarray(colors, dtype=np.float32) if colors is not None else None
        if clr is not None and (clr.ndim != 2 or clr.shape[0] != n or clr.shape[1] != 3):
            raise ValueError(f"colors must be shape (N, 3), got {clr.shape}")

        scl = np.ascontiguousarray(scalars, dtype=np.float32) if scalars is not None else None
        if scl is not None and scl.shape[0] != n:
            raise ValueError(f"scalars length {scl.shape[0]} must match N={n}")

        clim_arr = list(clim) if clim is not None else None
        nan_arr  = list(nan_color) if nan_color is not None else None

        if self._renderer is None:
            self._pending = dict(
                positions=pos, colors=clr, scalars=scl,
                colormap=colormap, point_size=point_size,
                clim=clim, nan_color=nan_color, log_scale=log_scale,
                opacity=opacity,
            )
            return

        self._renderer.set_points(pos, clr, scl, colormap, float(point_size),
                                   clim_arr, nan_arr, bool(log_scale), float(opacity))
        self._total_n = n
        self._mark_dirty()

    def add_points(
        self,
        positions: "np.ndarray",
        *,
        colors: Optional["np.ndarray"] = None,
        scalars: Optional["np.ndarray"] = None,
        colormap: str = "viridis",
        point_size: float = 4.0,
        clim: "tuple[float, float] | None" = None,
        nan_color: "tuple[float, float, float] | None" = None,
        log_scale: bool = False,
        opacity: float = 1.0,
    ) -> int:
        """Add a new point cloud actor on top of the existing scene.

        Returns an integer handle. Pass it to ``update_actor()``,
        ``remove_actor()``, or ``set_actor_visibility()`` to manipulate the
        actor later. A virtual handle (non-negative int) is returned even before
        the widget is mapped; it is resolved to a real renderer handle when the
        widget is first mapped.
        """
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must be shape (N, 3), got {pos.shape}")
        n = pos.shape[0]
        clr = np.ascontiguousarray(colors, dtype=np.float32) if colors is not None else None
        scl = np.ascontiguousarray(scalars, dtype=np.float32) if scalars is not None else None

        kwargs = dict(positions=pos, colors=clr, scalars=scl,
                      colormap=colormap, point_size=point_size,
                      clim=clim, nan_color=nan_color, log_scale=log_scale,
                      opacity=float(opacity))

        if self._renderer is None:
            vhandle = self._next_phandle
            self._next_phandle += 1
            self._pending_actors.append((kwargs, vhandle))
            return vhandle

        handle = int(self._renderer.add_points(**kwargs))
        self._total_n += n
        self._mark_dirty()
        return handle

    def _resolve_actor_handle(self, handle: int) -> int:
        """Translate a virtual (pre-map) actor handle to the real renderer handle."""
        return self._phandle_map.get(handle, handle)

    def update_actor(
        self,
        handle: int,
        positions: "np.ndarray",
        *,
        colors: Optional["np.ndarray"] = None,
        scalars: Optional["np.ndarray"] = None,
        colormap: str = "viridis",
        point_size: float = 4.0,
        clim: "tuple[float, float] | None" = None,
        nan_color: "tuple[float, float, float] | None" = None,
        log_scale: bool = False,
        opacity: float = 1.0,
    ) -> None:
        """Replace the data of an existing actor in place."""
        if self._renderer is None:
            return
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        clr = np.ascontiguousarray(colors, dtype=np.float32) if colors is not None else None
        scl = np.ascontiguousarray(scalars, dtype=np.float32) if scalars is not None else None
        self._renderer.update_actor(self._resolve_actor_handle(handle), pos,
                                     colors=clr, scalars=scl,
                                     colormap=colormap, point_size=point_size,
                                     clim=clim, nan_color=nan_color, log_scale=log_scale,
                                     opacity=float(opacity))
        self._mark_dirty()

    def remove_actor(self, handle: int) -> None:
        """Remove a point cloud actor by handle."""
        if self._renderer is not None:
            self._renderer.remove_actor(self._resolve_actor_handle(handle))
            self._mark_dirty()

    def set_actor_visibility(self, handle: int, visible: bool) -> None:
        """Show or hide an actor without removing it."""
        if self._renderer is not None:
            self._renderer.set_actor_visibility(self._resolve_actor_handle(handle), visible)
            self._mark_dirty()

    def clear(self) -> None:
        """Remove all point actors, all line overlays, and clear the scene."""
        self._pending = None
        self._pending_actors.clear()
        self._phandle_map.clear()
        self._next_phandle = 0
        self._total_n = 0
        self._pending_overlays.clear()
        self._vhandle_map.clear()
        self._next_vhandle = 0
        if self._renderer is not None:
            self._renderer.clear_actors()
            self._renderer.clear_overlays()
            self._mark_dirty()

    # ── Line / overlay actors ─────────────────────────────────────────────────

    def add_lines(
        self,
        segments: "np.ndarray",
        color: "tuple[float, float, float]" = (1.0, 1.0, 1.0),
    ) -> int:
        """Add a set of line segments as an overlay actor in world space.

        Parameters
        ----------
        segments : (N, 6) float32 array
            Each row is ``[x0, y0, z0, x1, y1, z1]`` defining one segment.
        color : (r, g, b)
            RGB colour in ``[0, 1]``.

        Returns
        -------
        int
            A handle to pass to ``update_lines``, ``remove_overlay``, etc.
            Returns a virtual handle (non-negative int) even before the renderer
            is initialized; the handle is resolved to a real renderer handle when
            the widget is first mapped.
        """
        seg = np.ascontiguousarray(segments, dtype=np.float32)
        if seg.ndim != 2 or seg.shape[1] != 6:
            raise ValueError(f"segments must be shape (N, 6), got {seg.shape}")
        if self._renderer is None:
            vhandle = self._next_vhandle
            self._next_vhandle += 1
            self._pending_overlays.append(("lines", seg, color, vhandle))
            return vhandle
        handle = int(self._renderer.add_lines(seg, color))
        self._mark_dirty()
        return handle

    def add_box(
        self,
        bounds: "tuple[float, float, float, float, float, float]",
        color: "tuple[float, float, float]" = (1.0, 1.0, 0.0),
    ) -> int:
        """Add a wireframe bounding box.

        Parameters
        ----------
        bounds : (xmin, ymin, zmin, xmax, ymax, zmax)
        color : (r, g, b) in ``[0, 1]``.

        Returns
        -------
        int
            A handle usable with ``remove_overlay``, ``set_overlay_visibility``,
            etc. Returns a virtual handle before the renderer is initialized;
            see :meth:`add_lines`.
        """
        x0, y0, z0, x1, y1, z1 = bounds
        edges = np.array([
            [x0, y0, z0, x1, y0, z0], [x1, y0, z0, x1, y1, z0],
            [x1, y1, z0, x0, y1, z0], [x0, y1, z0, x0, y0, z0],
            [x0, y0, z1, x1, y0, z1], [x1, y0, z1, x1, y1, z1],
            [x1, y1, z1, x0, y1, z1], [x0, y1, z1, x0, y0, z1],
            [x0, y0, z0, x0, y0, z1], [x1, y0, z0, x1, y0, z1],
            [x1, y1, z0, x1, y1, z1], [x0, y1, z0, x0, y1, z1],
        ], dtype=np.float32)
        return self.add_lines(edges, color)

    def _resolve_handle(self, handle: int) -> int:
        """Translate a virtual (pre-map) handle to the real renderer handle."""
        return self._vhandle_map.get(handle, handle)

    def update_lines(
        self,
        handle: int,
        segments: "np.ndarray",
        color: "tuple[float, float, float]" = (1.0, 1.0, 1.0),
    ) -> None:
        """Replace the geometry of an existing line overlay actor."""
        if self._renderer is None:
            return
        seg = np.ascontiguousarray(segments, dtype=np.float32)
        if seg.ndim != 2 or seg.shape[1] != 6:
            raise ValueError(f"segments must be shape (N, 6), got {seg.shape}")
        self._renderer.update_lines(self._resolve_handle(handle), seg, color)
        self._mark_dirty()

    def remove_overlay(self, handle: int) -> None:
        """Remove a line overlay actor by handle."""
        if self._renderer is not None:
            self._renderer.remove_overlay(self._resolve_handle(handle))
            self._mark_dirty()

    def set_overlay_visibility(self, handle: int, visible: bool) -> None:
        """Show or hide a line overlay actor."""
        if self._renderer is not None:
            self._renderer.set_overlay_visibility(self._resolve_handle(handle), visible)
            self._mark_dirty()

    def clear_overlays(self) -> None:
        """Remove all line overlay actors."""
        self._pending_overlays.clear()
        self._vhandle_map.clear()
        self._next_vhandle = 0
        if self._renderer is not None:
            self._renderer.clear_overlays()
            self._mark_dirty()

    def show_orientation_axes(self, visible: bool = True) -> None:
        """Show or hide the orientation axes widget in the bottom-left corner."""
        self._orientation_axes_visible = visible
        if self._renderer is not None:
            self._renderer.show_orientation_axes(visible)
            self._mark_dirty()

    # ── Export ────────────────────────────────────────────────────────────────

    def screenshot(self) -> "np.ndarray | None":
        """Capture the current scene as an RGBA uint8 NumPy array of shape (H, W, 4).

        Returns ``None`` when the renderer has not been initialized yet (i.e.
        the widget has never been mapped on screen).
        """
        if self._renderer is None:
            return None
        w, h, raw = self._renderer.screenshot()
        return np.frombuffer(bytes(raw), dtype=np.uint8).reshape(h, w, 4).copy()

    def save_png(self, path: str) -> None:
        """Save the current scene to a PNG file.

        Uses *Pillow* when available; falls back to a pure-stdlib PNG writer
        otherwise, so no optional dependency is required.

        Parameters
        ----------
        path : str
            Destination file path (should end in ``.png``).
        """
        img = self.screenshot()
        if img is None:
            raise RuntimeError("Widget has not been mapped — call after the window is shown.")
        try:
            from PIL import Image as _PILImage
            _PILImage.fromarray(img, mode="RGBA").save(path)
        except ImportError:
            _write_png(path, img)

    # ── Animation export ──────────────────────────────────────────────────────

    def open_gif(self, path: str, fps: int = 20, loop: int = 0) -> None:
        """Begin recording frames for an animated GIF.

        Call :meth:`write_frame` for each frame you want, then
        :meth:`close_gif` to write the file.

        Parameters
        ----------
        path : str
            Output ``.gif`` path.
        fps : int
            Target playback speed in frames per second.
        loop : int
            Number of times the GIF loops; ``0`` = infinite.
        """
        self._gif_frames = []
        self._gif_path = path
        self._gif_fps = fps
        self._gif_loop = loop

    def write_frame(self) -> None:
        """Capture the current scene and append it to the active GIF recording.

        Raises ``RuntimeError`` if called before :meth:`open_gif`.
        """
        if self._gif_frames is None:
            raise RuntimeError("Call open_gif() before write_frame().")
        img = self.screenshot()
        if img is not None:
            self._gif_frames.append(img)

    def close_gif(self) -> None:
        """Finalise and write the GIF file started by :meth:`open_gif`.

        Uses Pillow when available for better colour quality; falls back to a
        pure-stdlib encoder with 3-3-2 quantisation.  Safe to call multiple
        times (second call is a no-op).
        """
        if self._gif_frames is None:
            return
        frames, path, fps, loop = (
            self._gif_frames, self._gif_path, self._gif_fps, self._gif_loop
        )
        self._gif_frames = None
        self._gif_path = None
        if not frames or path is None:
            return
        try:
            from PIL import Image as _Im
            imgs = [_Im.fromarray(f, "RGBA").convert("P", palette=_Im.Palette.ADAPTIVE)
                    for f in frames]
            delay_ms = max(20, round(1000 / fps))
            imgs[0].save(
                path,
                save_all=True,
                append_images=imgs[1:],
                duration=delay_ms,
                loop=loop,
                optimize=False,
            )
        except ImportError:
            _write_gif_stdlib(path, frames, fps, loop)

    def orbit_gif(
        self,
        path: str,
        n_frames: int = 60,
        fps: int = 20,
        loop: int = 0,
        elevation: "float | None" = None,
        on_progress: "callable | None" = None,
    ) -> None:
        """Orbit the camera 360° and save as an animated GIF.

        Parameters
        ----------
        path : str
            Output ``.gif`` path.
        n_frames : int
            Number of frames in the animation (default 60 = 3 s at 20 fps).
        fps : int
            Playback speed.
        loop : int
            ``0`` = infinite loop.
        elevation : float or None
            Camera pitch in radians.  ``None`` keeps the current pitch.
        on_progress : callable or None
            Called as ``on_progress(frame_index, n_frames)`` after each frame.
            Useful for updating a progress label.
        """
        if self._renderer is None:
            raise RuntimeError("Widget must be mapped before recording.")
        import math
        saved = self._renderer.get_camera()
        pitch = elevation if elevation is not None else saved.get("pitch", 0.3)
        yaw0 = float(saved.get("yaw", 0.0))

        self.open_gif(path, fps=fps, loop=loop)
        try:
            for i in range(n_frames):
                state = dict(saved)
                state["yaw"] = yaw0 + 2.0 * math.pi * i / n_frames
                state["pitch"] = pitch
                self._renderer.set_camera(state)
                self.write_frame()
                if on_progress is not None:
                    on_progress(i, n_frames)
        finally:
            self.close_gif()
            self._renderer.set_camera(saved)
            self._mark_dirty()

    def scalar_bar(
        self,
        visible: bool = True,
        *,
        vmin: float = 0.0,
        vmax: float = 1.0,
        log_scale: bool = False,
        colormap: str = "viridis",
        title: str = "",
    ) -> None:
        """Show or hide the scalar bar overlay.

        Parameters
        ----------
        visible : bool
            Show the scalar bar when True, hide it when False.
        vmin, vmax : float
            The data range the colormap spans.
        log_scale : bool
            Mirror the log_scale used in set_points so tick labels are correct.
        colormap : str
            Colormap name (should match the one used in set_points).
        title : str
            Optional label drawn above the bar.
        """
        if self._renderer is not None:
            self._renderer.show_scalar_bar(visible, vmin, vmax, log_scale, colormap, title)
            self._mark_dirty()
        else:
            self._pending_scalar_bar = {
                "visible": visible, "vmin": vmin, "vmax": vmax,
                "log_scale": log_scale, "colormap": colormap, "title": title,
            }

    # ── Linked-camera support ─────────────────────────────────────────────────

    def _propagate_camera(self) -> None:
        """Push our current camera state to all linked widgets."""
        if self._propagating or self._renderer is None or not self._camera_links:
            return
        state = self._renderer.get_camera()
        dead: list = []
        for other in self._camera_links:
            try:
                other._receive_camera(state)
            except Exception:
                dead.append(other)
        for d in dead:
            self._camera_links.discard(d)

    def _receive_camera(self, state: dict) -> None:
        """Apply a camera state coming from a linked widget (no further propagation)."""
        if self._renderer is None:
            return
        self._propagating = True
        try:
            self._renderer.set_camera(state)
            self._mark_dirty()
        finally:
            self._propagating = False

    def reset_camera(self) -> None:
        """Reset the camera to the fitted view for the current dataset."""
        if self._renderer is not None:
            self._renderer.reset_camera()
            self._mark_dirty()
            self._propagate_camera()

    # ── Camera presets ────────────────────────────────────────────────────────

    def view_xy(self) -> None:
        """Look along +Z down onto the XY plane."""
        if self._renderer is not None:
            self._renderer.view_xy()
            self._mark_dirty()
            self._propagate_camera()

    def view_xz(self) -> None:
        """Look along +Y onto the XZ plane (front view)."""
        if self._renderer is not None:
            self._renderer.view_xz()
            self._mark_dirty()
            self._propagate_camera()

    def view_yz(self) -> None:
        """Look along -X onto the YZ plane (side view)."""
        if self._renderer is not None:
            self._renderer.view_yz()
            self._mark_dirty()
            self._propagate_camera()

    def view_isometric(self) -> None:
        """45°/45° isometric view."""
        if self._renderer is not None:
            self._renderer.view_isometric()
            self._mark_dirty()
            self._propagate_camera()

    @property
    def parallel_projection(self) -> bool:
        """True when orthographic projection is active."""
        return self._parallel_projection

    @parallel_projection.setter
    def parallel_projection(self, on: bool) -> None:
        self._parallel_projection = bool(on)
        if self._renderer is not None:
            self._renderer.set_parallel_projection(self._parallel_projection)
            self._mark_dirty()
            self._propagate_camera()

    def fit(self, bounds: "tuple[float,...] | None" = None) -> None:
        """Fit camera to *bounds* ``(xmin,ymin,zmin,xmax,ymax,zmax)`` or to the current dataset."""
        if self._renderer is not None:
            self._renderer.fit(list(bounds) if bounds is not None else None)
            self._mark_dirty()
            self._propagate_camera()

    def get_camera(self) -> dict:
        """Return the current camera state as a dict (serialisable, passable to set_camera)."""
        if self._renderer is not None:
            return self._renderer.get_camera()
        return {}

    def set_camera(self, state: dict) -> None:
        """Restore a camera state dict previously returned by get_camera()."""
        if self._renderer is not None:
            self._renderer.set_camera(state)
            self._mark_dirty()
            self._propagate_camera()

    # ── Rendering modes ───────────────────────────────────────────────────────

    _STYLE_MAP = {"circle": 0, "square": 1, "gaussian": 2}

    @property
    def point_style(self) -> str:
        """Point rendering style: ``"circle"`` (default), ``"square"``, or ``"gaussian"``."""
        return self._point_style

    @point_style.setter
    def point_style(self, style: str) -> None:
        code = self._STYLE_MAP.get(style)
        if code is None:
            raise ValueError(f"point_style must be 'circle', 'square', or 'gaussian', got {style!r}")
        self._point_style = style
        if self._renderer is not None:
            self._renderer.set_point_style(code)
            self._mark_dirty()

    def set_ticks(
        self,
        x: int | None = None,
        y: int | None = None,
        z: int | None = None,
    ) -> None:
        """Set max tick count per axis. Pass None to restore auto-scaling."""
        if self._renderer is not None:
            self._renderer.set_ticks(x=x, y=y, z=z)
            self._mark_dirty()
        else:
            self._pending_ticks = (x, y, z)

    # ── Visual appearance ─────────────────────────────────────────────────────

    def show_grid(self, visible: bool = True) -> None:
        """Show or hide the grid lines and tick labels."""
        self._grid_visible = visible
        if self._renderer is not None:
            self._renderer.show_grid(visible)
            self._mark_dirty()

    def set_background(self, color) -> None:
        """Set the background colour.

        Parameters
        ----------
        color : tuple or str
            Either an ``(r, g, b)`` tuple with float values in ``[0, 1]`` or a
            hex string such as ``"#0d0d12"``.
        """
        if isinstance(color, str):
            h = color.lstrip("#")
            if len(h) == 6:
                r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            else:
                raise ValueError(f"set_background: expected '#RRGGBB' hex string, got {color!r}")
        else:
            r, g, b = float(color[0]), float(color[1]), float(color[2])
        self._bg_color = (r, g, b)
        if self._renderer is not None:
            self._renderer.set_background_color(r, g, b)
            self._mark_dirty()

    def set_axes(
        self,
        x: str = "X",
        y: str = "Y",
        z: str = "Z",
    ) -> None:
        """Set the axis title labels displayed at the grid extents.

        Parameters
        ----------
        x, y, z : str
            Label text for each axis. Pass an empty string ``""`` to hide a
            title without affecting the other two.
        """
        self._axis_labels = (x, y, z)
        if self._renderer is not None:
            self._renderer.set_axis_labels(x, y, z)
            self._mark_dirty()

    # ── Picking API ───────────────────────────────────────────────────────────

    def enable_point_picking(self, on_pick=None) -> None:
        """Activate point picking.

        A left-click with no drag finds the nearest visible point and fires
        ``<<PointPicked>>``. Read ``widget.picked_point``,
        ``widget.picked_index``, and ``widget.picked_actor`` in the handler.

        Parameters
        ----------
        on_pick : callable, optional
            Convenience callback bound to ``<<PointPicked>>``. Receives the Tk
            event object; read widget attributes for pick results.
        """
        self._pick_mode = "both" if self._pick_mode == "rect" else "point"
        if on_pick is not None:
            self.bind("<<PointPicked>>", on_pick, add="+")

    def enable_rectangle_picking(self, on_select=None) -> None:
        """Activate rectangle selection via Shift+left-drag.

        On release, fires ``<<SelectionChanged>>``. Read
        ``widget.selected`` (list of ``{"actor": int, "index": int}`` dicts)
        in the handler.

        Parameters
        ----------
        on_select : callable, optional
            Convenience callback bound to ``<<SelectionChanged>>``.
        """
        self._pick_mode = "both" if self._pick_mode == "point" else "rect"
        if on_select is not None:
            self.bind("<<SelectionChanged>>", on_select, add="+")

    def disable_picking(self) -> None:
        """Return to orbit-only mode (no picking)."""
        self._pick_mode = "none"
        if self._renderer is not None:
            self._renderer.clear_selection_rect()
            self._mark_dirty()

    @staticmethod
    def colormap_names() -> list[str]:
        return ScatterRenderer.colormap_names()

    # ── Mouse handling ────────────────────────────────────────────────────────

    def _engage_lod(self) -> None:
        if self._lod_enabled and self._total_n > self._lod_threshold and self._renderer is not None:
            self._renderer.set_lod_factor(self._lod_factor)

    def _disengage_lod(self) -> None:
        if self._renderer is not None:
            self._renderer.set_lod_factor(1)
            self._mark_dirty()

    def _drag_start(self, event: tk.Event, button: int) -> None:
        self._drag_btn = button
        self._drag_x = event.x
        self._drag_y = event.y
        self._press_x = event.x
        self._press_y = event.y
        self.focus_set()

        # Start selection rectangle when Shift+left in rect-picking modes
        shift = bool(event.state & 0x0001)
        if button == 1 and shift and self._pick_mode in ("rect", "both"):
            self._sel_x0 = event.x
            self._sel_y0 = event.y
            self._rect_active = True
        else:
            self._engage_lod()

    def _drag_move(self, event: tk.Event, button: int) -> None:
        if self._renderer is None or self._drag_btn != button:
            return

        # Rectangle selection: Shift+left-drag
        if button == 1 and self._rect_active:
            self._renderer.set_selection_rect(
                float(self._sel_x0), float(self._sel_y0),
                float(event.x), float(event.y),
            )
            self._mark_dirty()
            return

        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        self._drag_x = event.x
        self._drag_y = event.y
        # Shift+left-drag → pan (button 2); plain left-drag → orbit (button 1)
        shift = bool(event.state & 0x0001)
        effective = 2 if (button == 1 and shift) else button
        self._renderer.mouse_drag(float(dx), float(dy), effective)
        self._mark_dirty()
        self._propagate_camera()

    def _drag_end(self, event: tk.Event) -> None:
        if self._rect_active:
            self._rect_active = False
            if self._renderer is not None:
                self._renderer.clear_selection_rect()
                x0, y0 = float(self._sel_x0), float(self._sel_y0)
                x1, y1 = float(event.x), float(event.y)
                if abs(x1 - x0) > 2 and abs(y1 - y0) > 2:
                    hits = self._renderer.pick_rectangle(x0, y0, x1, y1)
                    self.selected = hits
                    self.event_generate("<<SelectionChanged>>")
                self._mark_dirty()
            self._disengage_lod()
            self._drag_btn = None
            return

        # Point pick: left-click with minimal drag
        if (self._drag_btn == 1
                and self._pick_mode in ("point", "both")
                and self._renderer is not None):
            dx = abs(event.x - self._press_x)
            dy = abs(event.y - self._press_y)
            if dx <= self._pick_threshold and dy <= self._pick_threshold:
                result = self._renderer.pick_point(float(event.x), float(event.y))
                if result is not None:
                    self.picked_actor = result["actor"]
                    self.picked_index = result["index"]
                    self.picked_point = result["point"]
                else:
                    self.picked_actor = None
                    self.picked_index = None
                    self.picked_point = None
                self.event_generate("<<PointPicked>>")

        self._disengage_lod()
        self._drag_btn = None

    def _on_scroll(self, event: tk.Event) -> None:
        if self._renderer is None:
            return
        self._renderer.scroll(event.delta / 120.0)
        self._mark_dirty()
        self._propagate_camera()

    def _on_scroll_up_x11(self, _event: tk.Event) -> None:
        if self._renderer is not None:
            self._renderer.scroll(1.0)
            self._mark_dirty()
            self._propagate_camera()

    def _on_scroll_down_x11(self, _event: tk.Event) -> None:
        if self._renderer is not None:
            self._renderer.scroll(-1.0)
            self._mark_dirty()
            self._propagate_camera()


# ── Linked-camera module API ──────────────────────────────────────────────────

def link_cameras(*widgets: ScatterWidget) -> None:
    """Synchronise the camera across two or more ``ScatterWidget`` instances.

    After linking, orbiting, panning, zooming, or applying any camera preset on
    any widget immediately mirrors the view on all others.  Widgets can be in
    different Tk windows.

    Call ``unlink_cameras(*widgets)`` to break the link.

    Parameters
    ----------
    *widgets
        Two or more ``ScatterWidget`` instances to link together.
    """
    if len(widgets) < 2:
        return
    for i, w in enumerate(widgets):
        for j, other in enumerate(widgets):
            if i != j:
                w._camera_links.add(other)


def unlink_cameras(*widgets: ScatterWidget) -> None:
    """Remove camera synchronisation between the given widgets.

    Removes all cross-links *among* the supplied widgets.  Links to widgets
    not listed are left intact.

    Parameters
    ----------
    *widgets
        The widgets to unlink from each other.
    """
    widget_set = set(widgets)
    for w in widgets:
        w._camera_links -= widget_set
