use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use wgpu::util::DeviceExt;

use crate::camera::{Camera, CameraState};
use crate::grid::{build_grid, LineVertex};

// ── GPU data structures ───────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    screen_size: [f32; 2],
    /// 0 = circle (soft), 1 = square, 2 = gaussian
    style: u32,
    _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PointInstance {
    pub position: [f32; 3],
    pub size: f32,
    pub color: [f32; 3],
    pub alpha: f32,
}

// ── Growable GPU buffer ───────────────────────────────────────────────────────

struct GrowableBuffer {
    buf: Option<wgpu::Buffer>,
    capacity: u64,
    usage: wgpu::BufferUsages,
}

impl GrowableBuffer {
    fn new(usage: wgpu::BufferUsages) -> Self {
        Self { buf: None, capacity: 0, usage }
    }

    /// Upload `data` bytes. Reallocates (with 1.5x headroom) only when capacity is exceeded.
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) {
        let needed = data.len() as u64;
        if needed == 0 {
            return;
        }
        if needed > self.capacity {
            let new_cap = needed + needed / 2;
            self.buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: new_cap,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.capacity = new_cap;
        }
        queue.write_buffer(self.buf.as_ref().unwrap(), 0, data);
    }

    fn slice(&self) -> Option<wgpu::BufferSlice<'_>> {
        self.buf.as_ref().map(|b| b.slice(..))
    }
}

// ── Line overlay actor ────────────────────────────────────────────────────────

struct LineActor {
    id: u32,
    buf: GrowableBuffer,
    vertex_count: u32,
    visible: bool,
    data_min: Vec3,
    data_max: Vec3,
}

// ── Actor (a single uploadable point cloud) ───────────────────────────────────

struct Actor {
    id: u32,
    buf: GrowableBuffer,
    positions: Vec<[f32; 3]>,   // CPU copy for picking
    count: u32,
    visible: bool,
    data_min: Vec3,
    data_max: Vec3,
}

// ── Cached label (pre-shaped, world position only) ────────────────────────────

struct CachedLabel {
    glyph_buf: Buffer,
    world_pos: Vec3,
    tick_pos: Vec3,
    is_axis_title: bool,
}

/// Screen-space label for the scalar bar (position in pixels).
struct ScalarBarLabel {
    glyph_buf: Buffer,
    px: f32,
    py: f32,
}

// ── Renderer ─────────────────────────────────────────────────────────────────

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    point_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,

    actors: Vec<Actor>,
    next_actor_id: u32,

    line_buf: GrowableBuffer,
    line_count: u32,

    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    pub camera: Camera,
    fit_center: Vec3,
    fit_radius: f32,

    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    viewport: Viewport,
    cached_labels: Vec<CachedLabel>,
    atlas_trim_counter: u32,
    last_grid_min: Option<Vec3>,
    last_grid_max: Option<Vec3>,
    last_data_min: Option<Vec3>,
    last_data_max: Option<Vec3>,
    tick_override: [Option<usize>; 3],

    // Scalar bar overlay (screen-space, drawn with identity view_proj)
    scalar_bar_buf: GrowableBuffer,
    scalar_bar_line_count: u32,
    overlay_bind_group: wgpu::BindGroup,  // identity-matrix uniform
    scalar_bar_labels: Vec<ScalarBarLabel>,
    scalar_bar_visible: bool,

    // Selection rectangle overlay (screen-space, same pipeline as scalar bar)
    sel_rect_buf: GrowableBuffer,
    sel_rect_visible: bool,

    // User-defined line overlay actors (depth-tested, world space)
    line_actors: Vec<LineActor>,
    next_line_actor_id: u32,

    // Orientation axes (computed per-frame from camera rotation, drawn as overlay)
    axes_buf: GrowableBuffer,
    axes_visible: bool,

    line_pipeline_nodepth: wgpu::RenderPipeline,

    surface_format: wgpu::TextureFormat,
    width: u32,
    height: u32,

    /// Active point style: 0 = circle, 1 = square, 2 = gaussian
    point_style: u32,
    /// LOD divisor: draw only first `count / lod_factor` instances (1 = full quality)
    lod_factor: u32,

    // ── Visual appearance ─────────────────────────────────────────────────────
    grid_visible: bool,
    bg_color: [f64; 4],
    axis_label_texts: [String; 3],
}

impl Renderer {
    pub fn new(
        raw_window_handle: raw_window_handle::RawWindowHandle,
        raw_display_handle: raw_window_handle::RawDisplayHandle,
        width: u32,
        height: u32,
        present_mode: wgpu::PresentMode,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface: wgpu::Surface<'static> = unsafe {
            let s = instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle,
                raw_window_handle,
            })?;
            std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(s)
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or("No suitable GPU adapter found")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("tkfastscatter"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let (depth_texture, depth_view) = make_depth_texture(&device, width.max(1), height.max(1));

        let dummy_uniforms = Uniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            screen_size: [width as f32, height as f32],
            style: 0,
            _pad: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&dummy_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform_bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Overlay uniform: identity view_proj so vertices are in NDC space directly.
        let overlay_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("overlay_uniforms"),
            contents: bytemuck::bytes_of(&dummy_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let overlay_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay_bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: overlay_uniform_buf.as_entire_binding(),
            }],
        });

        let point_pipeline = build_point_pipeline(
            &device,
            &uniform_layout,
            surface_format,
            include_str!("shaders/points.wgsl"),
        );
        let line_pipeline = build_line_pipeline(
            &device, &uniform_layout, surface_format,
            include_str!("shaders/lines.wgsl"), true,
        );
        let line_pipeline_nodepth = build_line_pipeline(
            &device, &uniform_layout, surface_format,
            include_str!("shaders/lines.wgsl"), false,
        );

        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let glyph_cache = Cache::new(&device);
        let viewport = Viewport::new(&device, &glyph_cache);
        let mut text_atlas = TextAtlas::new(&device, &queue, &glyph_cache, surface_format);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        );

        let camera = Camera::fit(Vec3::ZERO, 1.0, width as f32 / height.max(1) as f32);

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            depth_texture,
            depth_view,
            point_pipeline,
            line_pipeline,
            actors: Vec::new(),
            next_actor_id: 0,
            line_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            line_count: 0,
            uniform_buffer,
            uniform_bind_group,
            camera,
            fit_center: Vec3::ZERO,
            fit_radius: 1.0,
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            viewport,
            cached_labels: Vec::new(),
            atlas_trim_counter: 0,
            last_grid_min: None,
            last_grid_max: None,
            last_data_min: None,
            last_data_max: None,
            tick_override: [None; 3],
            scalar_bar_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            scalar_bar_line_count: 0,
            overlay_bind_group,
            scalar_bar_labels: Vec::new(),
            scalar_bar_visible: false,
            sel_rect_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            sel_rect_visible: false,
            line_actors: Vec::new(),
            next_line_actor_id: 0,
            axes_buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            axes_visible: false,
            line_pipeline_nodepth,
            surface_format,
            width,
            height,
            point_style: 0,
            lod_factor: 1,
            grid_visible: true,
            bg_color: [0.05, 0.05, 0.07, 1.0],
            axis_label_texts: ["X".to_string(), "Y".to_string(), "Z".to_string()],
        })
    }

    // ── Scalar bar ────────────────────────────────────────────────────────────

    /// Build or update the scalar bar overlay.
    /// `cpts` is the resolved colormap table; `vmin`/`vmax` are the display limits;
    /// `log_scale` mirrors the normalization used for the points.
    pub fn set_scalar_bar(
        &mut self,
        visible: bool,
        vmin: f32,
        vmax: f32,
        log_scale: bool,
        cpts: &[[f32; 3]],
        title: &str,
    ) {
        self.scalar_bar_visible = visible;
        if !visible {
            self.scalar_bar_line_count = 0;
            self.scalar_bar_labels.clear();
            return;
        }

        // Scalar bar geometry: a vertical gradient strip in NDC space.
        // The bar occupies the top-right corner; exact pixel sizes are computed
        // from the current viewport dimensions.
        let (w, h) = (self.width as f32, self.height as f32);
        // Bar dimensions & position in pixels
        let bar_w = 16.0_f32;
        let bar_h = (h * 0.45).min(220.0).max(60.0);
        let margin_r = 52.0_f32;  // from right edge
        let margin_t = 32.0_f32;  // from top edge
        let bar_x1 = w - margin_r - bar_w;  // left edge in pixels
        let bar_x2 = w - margin_r;           // right edge
        let bar_y1 = margin_t;               // top edge
        let bar_y2 = margin_t + bar_h;       // bottom edge

        // Convert pixel coords to NDC [-1, 1]
        let to_ndc = |px: f32, py: f32| -> [f32; 3] {
            [(px / w) * 2.0 - 1.0, 1.0 - (py / h) * 2.0, 0.0]
        };

        // Gradient: N horizontal line pairs, each colored by the colormap.
        const GRAD_STEPS: usize = 64;
        let mut verts: Vec<LineVertex> = Vec::with_capacity(GRAD_STEPS * 2);
        for i in 0..GRAD_STEPS {
            let t_top = i as f32 / GRAD_STEPS as f32;
            let t_bot = (i + 1) as f32 / GRAD_STEPS as f32;
            let t_mid = (t_top + t_bot) * 0.5;
            // t=0 → vmax (top), t=1 → vmin (bottom)
            let color = crate::colormap::sample(cpts, 1.0 - t_mid);
            let y_top = bar_y1 + t_top * bar_h;
            let y_bot = bar_y1 + t_bot * bar_h;
            // Draw top and bottom edges of this band (both same color → solid band)
            verts.push(LineVertex { position: to_ndc(bar_x1, y_top), color });
            verts.push(LineVertex { position: to_ndc(bar_x2, y_top), color });
            verts.push(LineVertex { position: to_ndc(bar_x1, y_bot), color });
            verts.push(LineVertex { position: to_ndc(bar_x2, y_bot), color });
        }
        // Thin white border around the bar
        let border = [0.7_f32, 0.7, 0.7];
        let corners = [
            to_ndc(bar_x1, bar_y1), to_ndc(bar_x2, bar_y1),
            to_ndc(bar_x2, bar_y1), to_ndc(bar_x2, bar_y2),
            to_ndc(bar_x2, bar_y2), to_ndc(bar_x1, bar_y2),
            to_ndc(bar_x1, bar_y2), to_ndc(bar_x1, bar_y1),
        ];
        for i in (0..corners.len()).step_by(2) {
            verts.push(LineVertex { position: corners[i],   color: border });
            verts.push(LineVertex { position: corners[i+1], color: border });
        }

        self.scalar_bar_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
        self.scalar_bar_line_count = verts.len() as u32;

        // Labels: title + tick values
        self.scalar_bar_labels.clear();
        let label_x = bar_x2 + 4.0;  // just right of bar

        let mut add_label = |text: String, px: f32, py: f32| {
            let mut buf = Buffer::new(&mut self.font_system, Metrics::new(11.0, 14.0));
            buf.set_size(&mut self.font_system, Some(80.0), Some(20.0));
            buf.set_text(
                &mut self.font_system,
                &text,
                Attrs::new().family(Family::SansSerif),
                Shaping::Basic,
            );
            buf.shape_until_scroll(&mut self.font_system, false);
            self.scalar_bar_labels.push(ScalarBarLabel { glyph_buf: buf, px, py });
        };

        // Tick labels: vmax at top, vmin at bottom, one or two intermediate
        let tick_count = 5_usize;
        for i in 0..=tick_count {
            let t = i as f32 / tick_count as f32;  // 0 = top (vmax), 1 = bottom (vmin)
            let val = if log_scale {
                let lmin = vmin.max(1e-10).ln();
                let lmax = vmax.max(1e-10).ln();
                (lmin + t * (lmax - lmin)).exp()
            } else {
                vmax + t * (vmin - vmax)
            };
            let py = bar_y1 + t * bar_h - 5.0;
            add_label(crate::grid::format_tick_pub(val), label_x, py);
        }
        if !title.is_empty() {
            // Title above the bar
            add_label(title.to_string(), bar_x1, bar_y1 - 16.0);
        }
    }

    // ── Data upload / actor management ───────────────────────────────────────

    /// Replace the entire scene with a single point cloud.
    pub fn set_points(&mut self, instances: &[PointInstance], positions: Vec<[f32; 3]>, count: u32, data_min: Vec3, data_max: Vec3) {
        self.actors.clear();
        if count == 0 { return; }
        self._add_actor_buf(instances, positions, count, data_min, data_max);
    }

    fn _add_actor_buf(&mut self, instances: &[PointInstance], positions: Vec<[f32; 3]>, count: u32, data_min: Vec3, data_max: Vec3) -> u32 {
        let id = self.next_actor_id;
        self.next_actor_id += 1;
        let mut actor = Actor {
            id,
            buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            positions,
            count,
            visible: true,
            data_min,
            data_max,
        };
        actor.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(instances));
        self.actors.push(actor);
        id
    }

    /// Add a new point cloud actor and return its ID.
    pub fn add_actor(&mut self, instances: &[PointInstance], positions: Vec<[f32; 3]>, count: u32, data_min: Vec3, data_max: Vec3) -> u32 {
        self._add_actor_buf(instances, positions, count, data_min, data_max)
    }

    /// Replace data for an existing actor in-place. Returns false if not found.
    pub fn update_actor_data(&mut self, id: u32, instances: &[PointInstance], positions: Vec<[f32; 3]>, count: u32, data_min: Vec3, data_max: Vec3) -> bool {
        if let Some(a) = self.actors.iter_mut().find(|a| a.id == id) {
            a.count = count;
            a.data_min = data_min;
            a.data_max = data_max;
            a.positions = positions;
            if count > 0 {
                a.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(instances));
            }
            true
        } else {
            false
        }
    }

    /// Remove an actor by ID. Returns false if not found.
    pub fn remove_actor(&mut self, id: u32) -> bool {
        if let Some(pos) = self.actors.iter().position(|a| a.id == id) {
            self.actors.remove(pos);
            true
        } else {
            false
        }
    }

    /// Show or hide an actor. Returns false if not found.
    pub fn set_actor_visibility(&mut self, id: u32, visible: bool) -> bool {
        if let Some(a) = self.actors.iter_mut().find(|a| a.id == id) {
            a.visible = visible;
            true
        } else {
            false
        }
    }

    /// Remove all actors.
    pub fn clear_actors(&mut self) {
        self.actors.clear();
    }

    /// Union of visible point actor and line overlay bounds. None when the scene is empty.
    pub fn actor_union_bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut bmin = Vec3::splat(f32::INFINITY);
        let mut bmax = Vec3::splat(f32::NEG_INFINITY);
        let mut any = false;
        for a in &self.actors {
            if !a.visible { continue; }
            bmin = bmin.min(a.data_min);
            bmax = bmax.max(a.data_max);
            any = true;
        }
        for la in &self.line_actors {
            if !la.visible || la.vertex_count == 0 { continue; }
            bmin = bmin.min(la.data_min);
            bmax = bmax.max(la.data_max);
            any = true;
        }
        if any { Some((bmin, bmax)) } else { None }
    }

    // ── Picking ───────────────────────────────────────────────────────────────

    /// Return `(actor_id, point_index, world_pos)` for the point closest to
    /// the given screen position. `None` when the scene is empty.
    pub fn pick_point(&self, screen_x: f32, screen_y: f32) -> Option<(u32, u32, [f32; 3])> {
        let (w, h) = (self.width as f32, self.height as f32);
        let vp = self.camera.view_proj();
        let mut best_dist_sq = f32::MAX;
        let mut best: Option<(u32, u32, [f32; 3])> = None;

        for actor in &self.actors {
            if !actor.visible { continue; }
            for (i, &pos) in actor.positions.iter().enumerate() {
                let clip = vp * Vec3::from(pos).extend(1.0);
                if clip.w <= 0.0 { continue; }
                let ndc = clip.truncate() / clip.w;
                if ndc.x.abs() > 1.05 || ndc.y.abs() > 1.05 { continue; }
                let sx = (ndc.x + 1.0) * 0.5 * w;
                let sy = (1.0 - ndc.y) * 0.5 * h;
                let d_sq = (sx - screen_x).powi(2) + (sy - screen_y).powi(2);
                if d_sq < best_dist_sq {
                    best_dist_sq = d_sq;
                    best = Some((actor.id, i as u32, pos));
                }
            }
        }
        best
    }

    /// Return `(actor_id, point_index)` for all visible points whose screen
    /// projection falls inside the given screen-space rectangle.
    pub fn pick_rectangle(&self, x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<(u32, u32)> {
        let (w, h) = (self.width as f32, self.height as f32);
        let vp = self.camera.view_proj();
        // Convert screen rect to NDC ranges (flip Y: screen-Y increases downward)
        let ndc_x_min = (x0.min(x1) / w) * 2.0 - 1.0;
        let ndc_x_max = (x0.max(x1) / w) * 2.0 - 1.0;
        let ndc_y_min = 1.0 - (y0.max(y1) / h) * 2.0;
        let ndc_y_max = 1.0 - (y0.min(y1) / h) * 2.0;

        let mut result = Vec::new();
        for actor in &self.actors {
            if !actor.visible { continue; }
            for (i, &pos) in actor.positions.iter().enumerate() {
                let clip = vp * Vec3::from(pos).extend(1.0);
                if clip.w <= 0.0 { continue; }
                let ndc = clip.truncate() / clip.w;
                if ndc.x >= ndc_x_min && ndc.x <= ndc_x_max
                    && ndc.y >= ndc_y_min && ndc.y <= ndc_y_max
                {
                    result.push((actor.id, i as u32));
                }
            }
        }
        result
    }

    // ── Selection rectangle overlay ───────────────────────────────────────────

    /// Draw an in-progress selection rectangle (screen coords, pixels).
    pub fn set_selection_rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        let (w, h) = (self.width as f32, self.height as f32);
        let to_ndc = |px: f32, py: f32| -> [f32; 3] {
            [(px / w) * 2.0 - 1.0, 1.0 - (py / h) * 2.0, 0.0]
        };
        let tl = to_ndc(x0, y0);
        let tr = to_ndc(x1, y0);
        let br = to_ndc(x1, y1);
        let bl = to_ndc(x0, y1);
        let col = [0.4_f32, 0.8, 1.0];
        let verts: [LineVertex; 8] = [
            LineVertex { position: tl, color: col },
            LineVertex { position: tr, color: col },
            LineVertex { position: tr, color: col },
            LineVertex { position: br, color: col },
            LineVertex { position: br, color: col },
            LineVertex { position: bl, color: col },
            LineVertex { position: bl, color: col },
            LineVertex { position: tl, color: col },
        ];
        self.sel_rect_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
        self.sel_rect_visible = true;
    }

    pub fn clear_selection_rect(&mut self) {
        self.sel_rect_visible = false;
    }

    // ── Line overlay actors ───────────────────────────────────────────────────

    pub fn add_line_actor(&mut self, vertices: &[LineVertex]) -> u32 {
        let id = self.next_line_actor_id;
        self.next_line_actor_id += 1;
        let (data_min, data_max) = line_vertex_bounds(vertices);
        let mut actor = LineActor {
            id,
            buf: GrowableBuffer::new(wgpu::BufferUsages::VERTEX),
            vertex_count: vertices.len() as u32,
            visible: true,
            data_min,
            data_max,
        };
        if !vertices.is_empty() {
            actor.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(vertices));
        }
        self.line_actors.push(actor);
        id
    }

    pub fn update_line_actor_data(&mut self, id: u32, vertices: &[LineVertex]) -> bool {
        if let Some(a) = self.line_actors.iter_mut().find(|a| a.id == id) {
            a.vertex_count = vertices.len() as u32;
            let (data_min, data_max) = line_vertex_bounds(vertices);
            a.data_min = data_min;
            a.data_max = data_max;
            if !vertices.is_empty() {
                a.buf.upload(&self.device, &self.queue, bytemuck::cast_slice(vertices));
            }
            true
        } else {
            false
        }
    }

    pub fn remove_line_actor(&mut self, id: u32) -> bool {
        if let Some(pos) = self.line_actors.iter().position(|a| a.id == id) {
            self.line_actors.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn set_line_actor_visibility(&mut self, id: u32, visible: bool) -> bool {
        if let Some(a) = self.line_actors.iter_mut().find(|a| a.id == id) {
            a.visible = visible;
            true
        } else {
            false
        }
    }

    pub fn clear_line_actors(&mut self) {
        self.line_actors.clear();
    }

    // ── Rendering modes ───────────────────────────────────────────────────────

    /// Set the point rendering style: 0 = circle (soft), 1 = square, 2 = gaussian.
    pub fn set_point_style(&mut self, style: u32) {
        self.point_style = style.min(2);
    }

    /// Set the LOD divisor. When > 1 each actor draws only `count / lod_factor`
    /// instances, giving fast interaction at the cost of apparent density.
    pub fn set_lod_factor(&mut self, factor: u32) {
        self.lod_factor = factor.max(1);
    }

    // ── Visual appearance ─────────────────────────────────────────────────────

    pub fn set_grid_visible(&mut self, visible: bool) {
        self.grid_visible = visible;
    }

    pub fn set_background_color(&mut self, r: f64, g: f64, b: f64) {
        self.bg_color = [r, g, b, 1.0];
    }

    pub fn set_axis_labels(&mut self, x: String, y: String, z: String) {
        self.axis_label_texts = [x, y, z];
        if let (Some(dmin), Some(dmax), Some(nmin), Some(nmax)) = (
            self.last_data_min, self.last_data_max,
            self.last_grid_min, self.last_grid_max,
        ) {
            self.last_grid_min = None;
            self.last_grid_max = None;
            self.set_grid(dmin, dmax, nmin, nmax);
        }
    }

    // ── Orientation axes ──────────────────────────────────────────────────────

    pub fn set_orientation_axes_visible(&mut self, visible: bool) {
        self.axes_visible = visible;
    }

    fn update_axes_buf(&mut self) {
        if !self.axes_visible { return; }
        // Corner center in NDC (bottom-left, accounting for wgpu Y-up NDC).
        let (cx, cy) = (-0.82_f32, -0.82_f32);
        let scale = 0.13_f32;
        let vm = self.camera.view_matrix();
        // World-axis directions in camera space (X=right, Y=up in NDC).
        let axes: [([f32; 3], [f32; 3]); 3] = [
            ([1., 0., 0.], [0.95, 0.30, 0.30]),  // X — red
            ([0., 1., 0.], [0.30, 0.90, 0.30]),  // Y — green
            ([0., 0., 1.], [0.40, 0.60, 1.00]),  // Z — blue
        ];
        let mut verts = [LineVertex { position: [0.; 3], color: [0.; 3] }; 6];
        for (i, (world_axis, color)) in axes.iter().enumerate() {
            // transform_vector3 applies only the rotation part of the view matrix.
            let d = vm.transform_vector3(Vec3::from(*world_axis));
            verts[i * 2]     = LineVertex { position: [cx, cy, 0.], color: *color };
            verts[i * 2 + 1] = LineVertex { position: [cx + d.x * scale, cy + d.y * scale, 0.], color: *color };
        }
        self.axes_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&verts));
    }

    pub fn clear_grid(&mut self) {
        self.line_count = 0;
        self.cached_labels.clear();
        self.last_grid_min = None;
        self.last_grid_max = None;
        self.last_data_min = None;
        self.last_data_max = None;
    }

    pub fn set_tick_override(&mut self, x: Option<usize>, y: Option<usize>, z: Option<usize>) {
        self.tick_override = [x, y, z];
        // Rebuild immediately if we already have grid bounds from a prior set_grid call.
        if let (Some(dmin), Some(dmax), Some(nmin), Some(nmax)) = (
            self.last_data_min, self.last_data_max,
            self.last_grid_min, self.last_grid_max,
        ) {
            self.last_grid_min = None; // force rebuild path in set_grid
            self.last_grid_max = None;
            self.set_grid(dmin, dmax, nmin, nmax);
        }
    }

    pub fn set_grid(&mut self, data_min: Vec3, data_max: Vec3, nice_min: Vec3, nice_max: Vec3) {
        // Skip rebuild when rounded bounds and tick overrides are unchanged.
        if self.last_grid_min == Some(nice_min) && self.last_grid_max == Some(nice_max) {
            return;
        }
        self.last_grid_min = Some(nice_min);
        self.last_grid_max = Some(nice_max);
        self.last_data_min = Some(data_min);
        self.last_data_max = Some(data_max);

        let geo = build_grid(data_min, data_max, nice_min, nice_max, self.tick_override);

        self.line_buf.upload(&self.device, &self.queue, bytemuck::cast_slice(&geo.vertices));
        self.line_count = geo.vertices.len() as u32;

        // Pre-shape all label text — only happens when grid changes, not per-frame
        self.cached_labels.clear();
        for anchor in geo.labels {
            let mut buf = Buffer::new(&mut self.font_system, Metrics::new(11.0, 14.0));
            buf.set_size(&mut self.font_system, Some(120.0), Some(20.0));
            buf.set_text(
                &mut self.font_system,
                &anchor.text,
                Attrs::new().family(Family::SansSerif),
                Shaping::Basic,
            );
            buf.shape_until_scroll(&mut self.font_system, false);
            self.cached_labels.push(CachedLabel {
                glyph_buf: buf,
                world_pos: anchor.world_pos,
                tick_pos: anchor.tick_pos,
                is_axis_title: false,
            });
        }

        // Axis title labels — one per axis, at the midpoint of each outer edge.
        let axis_world_pos = [
            Vec3::new((nice_min.x + nice_max.x) * 0.5, nice_min.y, nice_min.z),
            Vec3::new(nice_min.x, (nice_min.y + nice_max.y) * 0.5, nice_min.z),
            Vec3::new(nice_min.x, nice_min.y, (nice_min.z + nice_max.z) * 0.5),
        ];
        let axis_texts = self.axis_label_texts.clone();
        for (text, world_pos) in axis_texts.iter().zip(axis_world_pos.iter()) {
            if text.is_empty() { continue; }
            let mut buf = Buffer::new(&mut self.font_system, Metrics::new(14.0, 18.0));
            buf.set_size(&mut self.font_system, Some(200.0), Some(24.0));
            buf.set_text(
                &mut self.font_system,
                text,
                Attrs::new().family(Family::SansSerif),
                Shaping::Basic,
            );
            buf.shape_until_scroll(&mut self.font_system, false);
            self.cached_labels.push(CachedLabel {
                glyph_buf: buf,
                world_pos: *world_pos,
                tick_pos: *world_pos,
                is_axis_title: true,
            });
        }
    }

    pub fn fit_camera(&mut self, center: Vec3, radius: f32) {
        let aspect = self.width as f32 / self.height.max(1) as f32;
        self.camera = Camera::fit(center, radius, aspect);
        self.fit_center = center;
        self.fit_radius = radius;
    }

    // ── Resize ────────────────────────────────────────────────────────────────

    pub fn resize(&mut self, width: u32, height: u32) {
        let (w, h) = (width.max(1), height.max(1));
        // Skip if nothing changed
        if w == self.width && h == self.height {
            return;
        }
        self.width = w;
        self.height = h;
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
        let (dt, dv) = make_depth_texture(&self.device, w, h);
        self.depth_texture = dt;
        self.depth_view = dv;
        self.camera.aspect = w as f32 / h as f32;
    }

    // ── Camera controls ───────────────────────────────────────────────────────

    pub fn mouse_drag(&mut self, dx: f32, dy: f32, button: u8) {
        match button {
            1 => self.camera.orbit(glam::Vec2::new(dx, dy)),
            2 => self.camera.pan(glam::Vec2::new(dx, dy)),
            _ => {}
        }
    }

    pub fn scroll(&mut self, delta: f32) {
        self.camera.zoom(delta);
    }

    /// Resets to the last fitted view (center + radius from the most recent `fit_camera` call).
    pub fn reset_camera(&mut self) {
        let parallel = self.camera.parallel;
        let aspect = self.width as f32 / self.height.max(1) as f32;
        self.camera = Camera::fit(self.fit_center, self.fit_radius, aspect);
        self.camera.parallel = parallel;
    }

    pub fn get_camera_state(&self) -> CameraState {
        self.camera.state()
    }

    pub fn set_camera_state(&mut self, state: CameraState) {
        self.camera.apply_state(state);
    }

    pub fn set_parallel_projection(&mut self, on: bool) {
        self.camera.parallel = on;
    }

    /// Reorient camera to a preset view direction, preserving target and distance.
    /// `yaw` and `pitch` are the spherical angles for the desired look direction.
    pub fn set_view_direction(&mut self, yaw: f32, pitch: f32) {
        self.camera.yaw = yaw;
        self.camera.pitch = pitch.clamp(-1.55, 1.55);
    }

    /// Fit the camera to explicit world-space bounds [min_x,min_y,min_z, max_x,max_y,max_z].
    pub fn fit_to_bounds(&mut self, bounds: [f32; 6]) {
        let bmin = glam::Vec3::from_slice(&bounds[0..3]);
        let bmax = glam::Vec3::from_slice(&bounds[3..6]);
        let center = (bmin + bmax) * 0.5;
        let radius = (bmax - bmin).length() * 0.5;
        self.fit_camera(center, radius.max(1e-6));
    }

    // ── Export ────────────────────────────────────────────────────────────────

    /// Render the current scene to an offscreen texture and return raw RGBA bytes.
    ///
    /// Returns `(width, height, rgba_bytes)`. Bytes are always RGBA regardless of
    /// the internal surface format (BGRA is swapped before returning).
    pub fn screenshot(&mut self) -> Result<(u32, u32, Vec<u8>), Box<dyn std::error::Error + Send + Sync>> {
        let (w, h) = (self.width, self.height);

        // Offscreen color texture — same format as surface so all pipelines match.
        let offscreen_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screenshot_color"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = offscreen_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Fresh depth texture (same size).
        let (_depth_tex, depth_view) = make_depth_texture(&self.device, w, h);

        // Update uniforms and text, exactly as in render().
        let view_proj = self.camera.view_proj();
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            screen_size: [w as f32, h as f32],
            style: self.point_style,
            _pad: 0.0,
        };
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        self.viewport.update(&self.queue, Resolution { width: w, height: h });
        self.prepare_text_labels(view_proj);
        self.update_axes_buf();

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("screenshot") }
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("screenshot_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.bg_color[0], g: self.bg_color[1],
                            b: self.bg_color[2], a: self.bg_color[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.line_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            if self.grid_visible {
                if let Some(slice) = self.line_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..self.line_count, 0..1);
                }
            }
            for la in &self.line_actors {
                if la.visible && la.vertex_count > 0 {
                    if let Some(slice) = la.buf.slice() {
                        pass.set_vertex_buffer(0, slice);
                        pass.draw(0..la.vertex_count, 0..1);
                    }
                }
            }
            pass.set_pipeline(&self.point_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            for actor in &self.actors {
                if actor.visible && actor.count > 0 {
                    if let Some(slice) = actor.buf.slice() {
                        pass.set_vertex_buffer(0, slice);
                        pass.draw(0..6, 0..actor.count);
                    }
                }
            }
            pass.set_pipeline(&self.line_pipeline);
            pass.set_bind_group(0, &self.overlay_bind_group, &[]);
            if self.scalar_bar_visible && self.scalar_bar_line_count > 0 {
                if let Some(slice) = self.scalar_bar_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..self.scalar_bar_line_count, 0..1);
                }
            }
            // Selection rect intentionally excluded from screenshots.
            if self.axes_visible {
                pass.set_pipeline(&self.line_pipeline_nodepth);
                pass.set_bind_group(0, &self.overlay_bind_group, &[]);
                if let Some(slice) = self.axes_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..6, 0..1);
                }
            }
            self.text_renderer.render(&self.text_atlas, &self.viewport, &mut pass).ok();
        }

        // Readback buffer — rows must be aligned to COPY_BYTES_PER_ROW_ALIGNMENT.
        let bytes_per_px = 4u32;
        let unpadded_row = w * bytes_per_px;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) & !(align - 1);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot_readback"),
            size: (padded_row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &offscreen_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Wait for GPU and read bytes back to CPU.
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let raw = readback.slice(..).get_mapped_range();
        let is_bgra = matches!(
            self.surface_format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        let mut pixels = Vec::with_capacity((w * h * bytes_per_px) as usize);
        for row in 0..h as usize {
            let start = row * padded_row as usize;
            let row_bytes = &raw[start..start + (w * bytes_per_px) as usize];
            if is_bgra {
                for px in row_bytes.chunks_exact(4) {
                    pixels.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
                }
            } else {
                pixels.extend_from_slice(row_bytes);
            }
        }
        drop(raw);
        readback.unmap();
        Ok((w, h, pixels))
    }

    // ── Render ────────────────────────────────────────────────────────────────

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let view_proj = self.camera.view_proj();
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            screen_size: [self.width as f32, self.height as f32],
            style: self.point_style,
            _pad: 0.0,
        };
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        self.viewport.update(&self.queue, Resolution { width: self.width, height: self.height });

        self.prepare_text_labels(view_proj);
        self.update_axes_buf();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.bg_color[0], g: self.bg_color[1],
                            b: self.bg_color[2], a: self.bg_color[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.line_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            if self.grid_visible {
                if let Some(slice) = self.line_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..self.line_count, 0..1);
                }
            }
            // User-defined line overlay actors (depth-tested)
            for la in &self.line_actors {
                if la.visible && la.vertex_count > 0 {
                    if let Some(slice) = la.buf.slice() {
                        pass.set_vertex_buffer(0, slice);
                        pass.draw(0..la.vertex_count, 0..1);
                    }
                }
            }

            pass.set_pipeline(&self.point_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            let lod = self.lod_factor.max(1);
            for actor in &self.actors {
                if actor.visible && actor.count > 0 {
                    if let Some(slice) = actor.buf.slice() {
                        pass.set_vertex_buffer(0, slice);
                        pass.draw(0..6, 0..(actor.count / lod).max(1));
                    }
                }
            }

            // Scalar bar: screen-space overlay drawn with identity view_proj.
            pass.set_pipeline(&self.line_pipeline);
            pass.set_bind_group(0, &self.overlay_bind_group, &[]);
            if self.scalar_bar_visible && self.scalar_bar_line_count > 0 {
                if let Some(slice) = self.scalar_bar_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..self.scalar_bar_line_count, 0..1);
                }
            }
            // Selection rectangle overlay (same pipeline, identity view_proj).
            if self.sel_rect_visible {
                if let Some(slice) = self.sel_rect_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..8, 0..1);
                }
            }
            // Orientation axes: 3 colored axis lines in the bottom-left corner.
            if self.axes_visible {
                pass.set_pipeline(&self.line_pipeline_nodepth);
                pass.set_bind_group(0, &self.overlay_bind_group, &[]);
                if let Some(slice) = self.axes_buf.slice() {
                    pass.set_vertex_buffer(0, slice);
                    pass.draw(0..6, 0..1);
                }
            }

            self.text_renderer
                .render(&self.text_atlas, &self.viewport, &mut pass)
                .ok();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Trim glyph atlas every 120 frames — infrequent enough to not defeat caching
        self.atlas_trim_counter += 1;
        if self.atlas_trim_counter >= 120 {
            self.text_atlas.trim();
            self.atlas_trim_counter = 0;
        }

        Ok(())
    }

    fn prepare_text_labels(&mut self, vp: glam::Mat4) {
        // No early-return on empty: we must always call prepare() so glyphon can
        // flush stale vertices from the previous frame (e.g. after clear_grid).

        // Build TextArea list by projecting pre-shaped buffers to current screen positions
        let (w, h) = (self.width, self.height);

        // Minimum screen-space gap (pixels) between a tick mark and its label.
        // When the push direction collapses into the depth axis (e.g. orthographic
        // aligned views), the gap is near-zero and the label is suppressed instead
        // of piling up on the grid.
        const MIN_PUSH_PX: f32 = 16.0;

        let mut text_areas: Vec<TextArea> = Vec::with_capacity(self.cached_labels.len());
        for label in &self.cached_labels {
            let clip = vp * label.world_pos.extend(1.0);
            if clip.w <= 0.0 { continue; }
            let ndc = clip.truncate() / clip.w;
            if ndc.x < -1.1 || ndc.x > 1.1 || ndc.y < -1.1 || ndc.y > 1.1 { continue; }

            let mut sx = (ndc.x + 1.0) * 0.5 * w as f32;
            let mut sy = (1.0 - ndc.y) * 0.5 * h as f32;

            if label.is_axis_title {
                // Axis titles: push 24px away from the grid center (screen mid).
                let cx = w as f32 * 0.5;
                let cy = h as f32 * 0.5;
                let dx = sx - cx;
                let dy = sy - cy;
                let len = (dx * dx + dy * dy).sqrt().max(1.0);
                sx += dx / len * 24.0;
                sy += dy / len * 24.0;
            } else {
                // Tick labels: push away from their tick mark; suppress when depth-aligned.
                let tick_clip = vp * label.tick_pos.extend(1.0);
                if tick_clip.w > 0.0 {
                    let tndc = tick_clip.truncate() / tick_clip.w;
                    let tx = (tndc.x + 1.0) * 0.5 * w as f32;
                    let ty = (1.0 - tndc.y) * 0.5 * h as f32;
                    let push = glam::Vec2::new(sx - tx, sy - ty);
                    let push_len = push.length();
                    if push_len < 1.0 {
                        continue;
                    }
                    if push_len < MIN_PUSH_PX {
                        let n = push / push_len;
                        sx = tx + n.x * MIN_PUSH_PX;
                        sy = ty + n.y * MIN_PUSH_PX;
                    }
                }
            }

            text_areas.push(TextArea {
                buffer: &label.glyph_buf,
                left: sx,
                top: sy,
                scale: 1.0,
                bounds: TextBounds::default(),
                default_color: if label.is_axis_title {
                    Color::rgb(220, 220, 240)
                } else {
                    Color::rgb(200, 200, 200)
                },
                custom_glyphs: &[],
            });
        }

        // Scalar bar text labels (screen-space, pixel positions already known).
        for lbl in &self.scalar_bar_labels {
            text_areas.push(TextArea {
                buffer: &lbl.glyph_buf,
                left: lbl.px,
                top: lbl.py,
                scale: 1.0,
                bounds: TextBounds::default(),
                default_color: Color::rgb(200, 200, 200),
                custom_glyphs: &[],
            });
        }

        // Always call prepare — even with an empty list, this clears any
        // glyph vertices from the previous frame, preventing stale text when
        // all labels project off-screen.
        let _ = self.text_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.font_system,
            &mut self.text_atlas,
            &self.viewport,
            text_areas,
            &mut self.swash_cache,
        );
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn line_vertex_bounds(verts: &[LineVertex]) -> (Vec3, Vec3) {
    let mut bmin = Vec3::splat(f32::INFINITY);
    let mut bmax = Vec3::splat(f32::NEG_INFINITY);
    for v in verts {
        let p = Vec3::from(v.position);
        bmin = bmin.min(p);
        bmax = bmax.max(p);
    }
    if verts.is_empty() { (Vec3::ZERO, Vec3::ZERO) } else { (bmin, bmax) }
}

fn make_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn build_point_pipeline(
    device: &wgpu::Device,
    uniform_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    wgsl: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("points_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("point_layout"),
        bind_group_layouts: &[uniform_layout],
        push_constant_ranges: &[],
    });
    let stride = std::mem::size_of::<PointInstance>() as u64;
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("point_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                    wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32   },
                    wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Float32x3 },
                    wgpu::VertexAttribute { offset: 28, shader_location: 3, format: wgpu::VertexFormat::Float32   },
                ],
            }],
        },
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

fn build_line_pipeline(
    device: &wgpu::Device,
    uniform_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    wgsl: &str,
    depth_test: bool,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lines_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("line_layout"),
        bind_group_layouts: &[uniform_layout],
        push_constant_ranges: &[],
    });
    let stride = std::mem::size_of::<LineVertex>() as u64;
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("line_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                    wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
                ],
            }],
        },
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: if depth_test { wgpu::CompareFunction::Less } else { wgpu::CompareFunction::Always },
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

