use glam::Vec3;

/// Expand raw data bounds to the nearest "nice" round numbers so the grid
/// stays visually stable between frames that share a similar data range.
/// Targets ~5 ticks per axis; step rounds to 1 / 2 / 5 × 10^n.
pub fn nice_bounds(min: Vec3, max: Vec3) -> (Vec3, Vec3) {
    let nice_axis = |lo: f32, hi: f32| -> (f32, f32) {
        let range = (hi - lo).abs();
        if range < 1e-10 {
            return (lo - 0.5, hi + 0.5);
        }
        let rough_step = range / 5.0;
        let mag = 10_f32.powf(rough_step.log10().floor());
        let norm = rough_step / mag;
        let nice_step = if norm <= 1.0 { 1.0 }
            else if norm <= 2.0 { 2.0 }
            else if norm <= 5.0 { 5.0 }
            else { 10.0 } * mag;
        let nice_min = (lo / nice_step).floor() * nice_step;
        let nice_max = (hi / nice_step).ceil()  * nice_step;
        (nice_min, nice_max)
    };
    let (x0, x1) = nice_axis(min.x, max.x);
    let (y0, y1) = nice_axis(min.y, max.y);
    let (z0, z1) = nice_axis(min.z, max.z);
    (Vec3::new(x0, y0, z0), Vec3::new(x1, y1, z1))
}

/// Generate tick positions at multiples of the nice step for [lo, hi].
/// Targets ~5 ticks and caps at MAX_TICKS; steps up to the next nice
/// increment if the initial step produces too many.
/// Returns an empty vec for degenerate ranges.
/// `max_ticks` is the caller-supplied cap; never exceeds this count.
fn axis_ticks(lo: f32, hi: f32, max_ticks: usize) -> Vec<f32> {
    let range = hi - lo;
    if range < 1e-10 || max_ticks == 0 {
        return vec![];
    }

    let initial_step = {
        let rough = range / max_ticks as f32;
        let mag = 10_f32.powf(rough.log10().floor());
        let norm = rough / mag;
        (if norm <= 1.0 { 1.0 } else if norm <= 2.0 { 2.0 } else if norm <= 5.0 { 5.0 } else { 10.0 }) * mag
    };

    let mut step = initial_step;
    loop {
        let first = (lo / step).ceil() * step;
        let mut ticks = Vec::with_capacity(max_ticks + 1);
        let mut t = first;
        while t <= hi + step * 1e-4 {
            ticks.push(t);
            t += step;
        }
        if ticks.len() <= max_ticks {
            return ticks;
        }
        // Too many — move to next nice step up (1→2→5→10 pattern).
        let mag = 10_f32.powf(step.log10().floor());
        let norm = (step / mag).round() as i32;
        step = match norm { 1 => 2.0, 2 => 5.0, _ => 10.0 } * mag;
    }
}

/// A line segment vertex: position + RGB color.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

pub struct LabelAnchor {
    pub world_pos: Vec3,
    /// The point on the grid edge the label refers to (before the world-space offset).
    /// Used at render time to compute the screen-space push direction and enforce a
    /// minimum pixel gap, so labels stay readable in any projection mode.
    pub tick_pos: Vec3,
    pub text: String,
}

pub struct GridGeometry {
    pub vertices: Vec<LineVertex>,
    pub labels: Vec<LabelAnchor>,
}

/// Builds the bounding-box wireframe plus axis tick marks and their text anchors.
///
/// `data_min`/`data_max` are the raw point-cloud bounds (used to detect flat
/// axes and suppress their ticks). `nice_min`/`nice_max` are the rounded bounds
/// used for the box extent and tick value positions.
/// `tick_override` lets callers pin the max tick count per axis [x, y, z].
/// `None` means auto (proportional to axis length).
pub fn build_grid(
    data_min: Vec3,
    data_max: Vec3,
    nice_min: Vec3,
    nice_max: Vec3,
    tick_override: [Option<usize>; 3],
) -> GridGeometry {
    let mut verts: Vec<LineVertex> = Vec::new();
    let mut labels: Vec<LabelAnchor> = Vec::new();

    let extent = nice_max - nice_min;
    let box_color = [0.45_f32, 0.45, 0.45];
    let x_col = [0.90_f32, 0.30, 0.30];
    let y_col = [0.30_f32, 0.90, 0.30];
    let z_col = [0.30_f32, 0.50, 0.90];

    // ── Bounding box (12 edges) ───────────────────────────────────────────────
    let c = [
        Vec3::new(nice_min.x, nice_min.y, nice_min.z),
        Vec3::new(nice_max.x, nice_min.y, nice_min.z),
        Vec3::new(nice_min.x, nice_max.y, nice_min.z),
        Vec3::new(nice_max.x, nice_max.y, nice_min.z),
        Vec3::new(nice_min.x, nice_min.y, nice_max.z),
        Vec3::new(nice_max.x, nice_min.y, nice_max.z),
        Vec3::new(nice_min.x, nice_max.y, nice_max.z),
        Vec3::new(nice_max.x, nice_max.y, nice_max.z),
    ];
    let edges: [(usize, usize); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7), // X-parallel
        (0, 2), (1, 3), (4, 6), (5, 7), // Y-parallel
        (0, 4), (1, 5), (2, 6), (3, 7), // Z-parallel
    ];
    for (a, b) in edges {
        verts.push(LineVertex { position: c[a].to_array(), color: box_color });
        verts.push(LineVertex { position: c[b].to_array(), color: box_color });
    }

    // ── Detect flat axes ──────────────────────────────────────────────────────
    // An axis is "flat" when its data range is negligible relative to the
    // overall diagonal. Flat axes get no tick marks (just the bounding box edge).
    let data_range = data_max - data_min;
    let diagonal = data_range.length().max(1e-10);
    let flat_x = data_range.x.abs() / diagonal < 0.01;
    let flat_y = data_range.y.abs() / diagonal < 0.01;
    let flat_z = data_range.z.abs() / diagonal < 0.01;

    let tick_len = extent.length() * 0.025;
    let label_offset = tick_len * 2.0;
    let pad = extent.length() * 0.12;

    // Scale max ticks per axis by its fraction of the longest axis.
    // Short axes get fewer ticks so labels don't overlap when the box is flat.
    let max_ne = extent.x.max(extent.y).max(extent.z).max(1e-10);
    let ticks_for = |e: f32| -> usize {
        let r = e / max_ne;
        if r < 0.15 { 2 } else if r < 0.40 { 3 } else { 5 }
    };
    let x_ticks = tick_override[0].unwrap_or_else(|| ticks_for(extent.x));
    let y_ticks = tick_override[1].unwrap_or_else(|| ticks_for(extent.y));
    let z_ticks = tick_override[2].unwrap_or_else(|| ticks_for(extent.z));

    // ── X ticks — front-bottom edge, labels hanging in −Y ────────────────────
    // Explicit tick override bypasses flat-axis suppression so set_ticks() works
    // even on degenerate planar/linear datasets.
    if !flat_x || tick_override[0].is_some() {
        for &val in &axis_ticks(nice_min.x, nice_max.x, x_ticks) {
            let v = Vec3::new(val, nice_min.y, nice_min.z);
            let end = v - Vec3::new(0.0, tick_len, 0.0);
            verts.push(LineVertex { position: v.to_array(), color: x_col });
            verts.push(LineVertex { position: end.to_array(), color: x_col });
            labels.push(LabelAnchor {
                world_pos: end - Vec3::new(0.0, label_offset, 0.0),
                tick_pos: end,
                text: format_tick(val),
            });
        }
        let x_mid = Vec3::new((nice_min.x + nice_max.x) * 0.5, nice_min.y, nice_min.z);
        labels.push(LabelAnchor {
            world_pos: x_mid - Vec3::new(0.0, pad, 0.0),
            tick_pos: x_mid,
            text: "X".to_string(),
        });
    }

    // ── Y ticks — left-front edge, labels extending in −X ────────────────────
    if !flat_y || tick_override[1].is_some() {
        for &val in &axis_ticks(nice_min.y, nice_max.y, y_ticks) {
            let v = Vec3::new(nice_min.x, val, nice_min.z);
            let end = v - Vec3::new(tick_len, 0.0, 0.0);
            verts.push(LineVertex { position: v.to_array(), color: y_col });
            verts.push(LineVertex { position: end.to_array(), color: y_col });
            labels.push(LabelAnchor {
                world_pos: end - Vec3::new(label_offset, 0.0, 0.0),
                tick_pos: end,
                text: format_tick(val),
            });
        }
        let y_mid = Vec3::new(nice_min.x, (nice_min.y + nice_max.y) * 0.5, nice_min.z);
        labels.push(LabelAnchor {
            world_pos: y_mid - Vec3::new(pad, 0.0, 0.0),
            tick_pos: y_mid,
            text: "Y".to_string(),
        });
    }

    // ── Z ticks — right-front edge (x=max), labels extending in +X ───────────
    // Placed on the opposite side from X so their labels never share a corner
    // direction: X labels go −Y, Z labels go +X — orthogonal, no overlap.
    if !flat_z || tick_override[2].is_some() {
        for &val in &axis_ticks(nice_min.z, nice_max.z, z_ticks) {
            let v = Vec3::new(nice_max.x, nice_min.y, val);
            let end = v + Vec3::new(tick_len, 0.0, 0.0);
            verts.push(LineVertex { position: v.to_array(), color: z_col });
            verts.push(LineVertex { position: end.to_array(), color: z_col });
            labels.push(LabelAnchor {
                world_pos: end + Vec3::new(label_offset, 0.0, 0.0),
                tick_pos: end,
                text: format_tick(val),
            });
        }
        let z_mid = Vec3::new(nice_max.x, nice_min.y, (nice_min.z + nice_max.z) * 0.5);
        labels.push(LabelAnchor {
            world_pos: z_mid + Vec3::new(pad, 0.0, 0.0),
            tick_pos: z_mid,
            text: "Z".to_string(),
        });
    }

    GridGeometry { vertices: verts, labels }
}

pub fn format_tick_pub(v: f32) -> String { format_tick(v) }

fn format_tick(v: f32) -> String {
    if v.abs() >= 1000.0 || (v.abs() < 0.01 && v != 0.0) {
        format!("{:.2e}", v)
    } else {
        format!("{:.3}", v)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}
