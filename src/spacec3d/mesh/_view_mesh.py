import gzip
import http.server
import json
import re
import socketserver
import threading
import time
import webbrowser
from os import PathLike
from pathlib import Path

import neuroglancer
import numpy as np

from ._volume_query import box_volume, segment_volumes

_QUERY_LAYER = "query_box"
_SEG_LAYER = "segmentation"


def view_precomputed_mesh(
    mesh_dir: str | PathLike[str],
    address: str = "127.0.0.1",
    data_port: int = 9010,
    segments: list[int] | None = None,
    query_volume: str | PathLike[str] | None = None,
    enable_query: bool = True,
) -> None:
    """Loads the mesh into neuroglancer.

    query_volume: path to the query_volume.npz written by create_mesh. If None,
        <mesh_dir>/query_volume.npz is used when present. When available, interactive
        volume queries are enabled:
          - hover a mesh and press 'c'  -> that segment's enclosed volume
          - draw a box in the 'query_box' layer and press 'b' -> foreground volume in it
        Results print to the console and to the viewer's status bar.
    enable_query: set False to skip all volume-query setup (annotation layer,
        keybindings) and view exactly like the original mesh-only viewer.
    """
    mesh_dir = Path(mesh_dir).resolve()
    if not mesh_dir.exists():
        raise SystemExit(f"Not found: {mesh_dir}")

    segments = segments or detect_segments(mesh_dir)
    if not segments:
        raise SystemExit(f"No mesh segments found in {mesh_dir}.")
    print(f"pre-selecting segments: {segments}", flush=True)

    labels, voxel_size = (None, None)
    if enable_query:
        labels, voxel_size = _load_query_volume(query_volume, mesh_dir)

    data_url = serve_precomputed(mesh_dir, address, data_port)
    print(f"serving {data_url}", flush=True)

    neuroglancer.set_server_bind_address(address)
    viewer = neuroglancer.Viewer()

    layer_kwargs = {
        "source": f"precomputed://{data_url}",
        "segments": segments,
        "segment_default_color": "#ffffff",  # make sure all segments have the same color by default
    }

    with viewer.txn() as s:
        s.layers[_SEG_LAYER] = neuroglancer.SegmentationLayer(**layer_kwargs)
        s.show_default_annotations = False
        s.show_slices = False
        if labels is not None:
            # xy-3d: one xy cross-section (to draw query boxes) beside the 3d view.
            # The volume is a thin z-slab, so xz/yz panels would be useless slivers.
            s.layout = "xy-3d"
            s.cross_section_background_color = "#000000"  # match the 3d panel
            # center the crosshair on the volume so the slice lands on tissue
            nz, ny, nx = labels.shape
            s.position = [nx / 2, ny / 2, nz / 2]  # x, y, z voxel coordinates
            # cross_section_scale is PHYSICAL METERS PER PIXEL (stored internally as
            # voxels/pixel = value / voxel_scale). To fit the tissue, divide its
            # physical extent by a nominal panel width. Use the precomputed resolution,
            # not the npz voxel_size (which may be 1 and unrelated to the display scale).
            res_x, res_y, res_z = _resolution_m(mesh_dir)
            extent_m = max(nx * res_x, ny * res_y, nz * res_z)
            s.cross_section_scale = extent_m / 800.0
        else:
            s.layout = "3d"

    viewer_url = viewer.get_viewer_url()
    print(f"\nNeuroglancer URL:\n  {viewer_url}\n", flush=True)

    webbrowser.open(viewer_url)

    # Add the query layer only after the client has resolved the segmentation
    # layer's coordinate space, so the box layer shares it (no second space, and
    # neuroglancer's auto-framing of the mesh is preserved).
    if labels is not None:
        _enable_volume_queries(viewer, labels, voxel_size)

    input("Press Enter to stop.\n")


def _load_query_volume(query_volume, mesh_dir: Path):
    """Load (labels, voxel_size) from the npz, or (None, None) if unavailable."""
    if query_volume is None:
        candidate = mesh_dir / "query_volume.npz"
        if not candidate.exists():
            return None, None
        query_volume = candidate
    data = np.load(query_volume)
    voxel_size = tuple(float(v) for v in data["voxel_size"])
    return data["labels"], voxel_size


def _resolution_m(mesh_dir: Path) -> tuple[float, float, float]:
    """Voxel size in meters (x, y, z) from the precomputed info; defaults to 1 um.

    This is the resolution neuroglancer uses for its coordinate space, which can
    differ from the npz voxel_size (e.g. voxel_size=(1,1,1) but info says 1000 nm).
    """
    try:
        info = json.loads((mesh_dir / "info").read_text())
        res_nm = info["scales"][0]["resolution"]  # nanometers, [x, y, z]
        return tuple(float(r) * 1e-9 for r in res_nm)
    except (OSError, KeyError, json.JSONDecodeError, IndexError):
        return (1e-6, 1e-6, 1e-6)


def _wait_for_dimensions(viewer, timeout: float = 20.0):
    """Block until the client resolves the global coordinate space (>=3 dims), or
    return None on timeout. Populated only after the browser connects to the viewer."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        dims = viewer.state.dimensions
        if dims is not None and len(dims.names) >= 3:
            return dims
        time.sleep(0.25)
    return None


def _enable_volume_queries(viewer, labels: np.ndarray, voxel_size) -> None:
    """Add the box annotation layer (sharing the resolved coordinate space) and
    register the query keybindings. Segment-pick works even if the box layer can't."""
    dims = _wait_for_dimensions(viewer)
    if dims is None:
        print(
            "volume queries: coordinate space not ready; box-draw ('b') disabled, "
            "segment-pick ('c') still available.",
            flush=True,
        )
    else:
        def _add_query_layer(s):
            s.layers[_QUERY_LAYER] = neuroglancer.LocalAnnotationLayer(
                dimensions=dims,
                annotation_color="#ffff00",
            )

        # retry_txn: the client is live and updates state concurrently, so a plain
        # txn races (ConcurrentModificationError); retry re-applies on conflict.
        viewer.retry_txn(_add_query_layer)
    _setup_volume_queries(viewer, labels, voxel_size, box_enabled=dims is not None)


def _setup_volume_queries(
    viewer, labels: np.ndarray, voxel_size, box_enabled: bool = True
) -> None:
    """Register the segment-pick and box-draw volume query actions / keybindings."""
    seg_vols = segment_volumes(labels, voxel_size)
    voxel_vol = float(np.prod(voxel_size))

    # ground-truth references for sanity-checking query results (main thread -> notebook)
    total_fg = sum(seg_vols.values()) / voxel_vol if voxel_vol else 0
    total_vox = labels.size
    nz, ny, nx = labels.shape
    print(
        f"label volume {nx}x{ny}x{nz} (x,y,z): {len(seg_vols):,} segments, "
        f"{total_fg:,.0f} foreground voxels = {100.0 * total_fg / total_vox:.2f}% of "
        f"{total_vox:,}. A whole-volume box should report ~{total_fg:,.0f} voxels.",
        flush=True,
    )

    def on_segment_query(s):
        seg = _picked_segment(s, _SEG_LAYER)
        if seg is None:
            _report(viewer, "hover a mesh, then press 'c' for its volume")
            return
        vol = seg_vols.get(seg)
        if vol is None:
            _report(viewer, f"segment {seg}: not in label volume")
            return
        _report(viewer, f"segment {seg} volume: {vol:,.1f}  ({seg_vols[seg] / voxel_vol:,.0f} voxels)")

    def on_box_query(s):
        state = viewer.state
        dim_names = list(state.dimensions.names) if state.dimensions else []
        boxes = [
            a
            for a in state.layers[_QUERY_LAYER].annotations
            if getattr(a, "point_a", None) is not None
            and getattr(a, "point_b", None) is not None
        ]
        if not boxes:
            _report(viewer, f"draw a box in '{_QUERY_LAYER}', then press 'b'")
            return
        box = boxes[-1]
        idx_a = _anno_to_indices(box.point_a, dim_names)
        idx_b = _anno_to_indices(box.point_b, dim_names)
        vol, count, capacity = box_volume(labels, voxel_size, idx_a, idx_b)
        fill = 100.0 * count / capacity if capacity else 0.0
        # logged so coordinate alignment can be sanity-checked on the first run
        print(
            f"[box] point_a={np.asarray(box.point_a)} point_b={np.asarray(box.point_b)} "
            f"dims={dim_names} -> idx_a={idx_a} idx_b={idx_b}",
            flush=True,
        )
        _report(
            viewer,
            f"box volume: {vol:,.1f}  ({count:,} of {capacity:,} box voxels, {fill:.1f}% fill)",
        )

    viewer.actions.add("query-segment-volume", on_segment_query)
    if box_enabled:
        viewer.actions.add("query-box-volume", on_box_query)

    def _bind(s):
        s.input_event_bindings.viewer["keyc"] = "query-segment-volume"
        if box_enabled:
            s.input_event_bindings.viewer["keyb"] = "query-box-volume"

    viewer.config_state.retry_txn(_bind, lock=True)

    msg = "volume queries enabled:  'c' = segment under cursor"
    if box_enabled:
        msg += f"   'b' = box drawn in '{_QUERY_LAYER}'"
    print(msg, flush=True)


def _report(viewer, message: str) -> None:
    """Show a query result in the console and the viewer status bar."""
    print(message, flush=True)

    def _set_status(s):
        s.status_messages["volume"] = message

    # runs on a keypress while the client is live, so retry on concurrent updates
    viewer.config_state.retry_txn(_set_status, lock=True)


def _picked_segment(action_state, layer_name: str) -> int | None:
    """Segment id under the cursor in layer_name, or None."""
    try:
        sel = action_state.selected_values[layer_name]
    except (KeyError, TypeError):
        return None
    val = getattr(sel, "value", None)
    if val is None:
        return None
    key = getattr(val, "key", None)  # SegmentIdMapEntry
    if key is not None:
        return int(key)
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _anno_to_indices(point, dim_names) -> tuple[float, float, float]:
    """Annotation coords (coordinate-space values) -> (z, y, x) voxel indices.

    The annotation layer shares the source voxel grid, so a coordinate value equals
    a voxel index. Reorder by dimension name; fall back to reversing (x,y,z)->(z,y,x).
    """
    point = np.asarray(point, dtype=float)
    names = [str(n).lower() for n in dim_names]
    if {"x", "y", "z"} <= set(names):
        coord = {n: point[i] for i, n in enumerate(names)}
        return (coord["z"], coord["y"], coord["x"])
    return tuple(point[:3][::-1])


class _RangeCORSHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with CORS headers and HTTP 206 Range support."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range")
        super().end_headers()

    def guess_type(self, path):
        # Serve .gz files as raw bytes; Neuroglancer decompresses them itself.
        if str(path).endswith(".gz"):
            return "application/octet-stream"
        return super().guess_type(path)

    def log_message(self, format, *args):
        pass  # suppress per-request noise

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        range_header = self.headers.get("Range")
        if not range_header:
            super().do_GET()
            return

        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404, "File not found")
            return

        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if not m:
            self.send_error(416, "Requested Range Not Satisfiable")
            return

        file_size = path.stat().st_size
        start = int(m.group(1))
        end = min(int(m.group(2)) if m.group(2) else file_size - 1, file_size - 1)
        if start > end:
            self.send_error(416, "Requested Range Not Satisfiable")
            return

        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()

        with open(path, "rb") as f:
            f.seek(start)
            self.wfile.write(f.read(length))


def serve_precomputed(path: Path, address: str, port: int) -> str:
    """Start the HTTP server and return its url."""
    def handler(*args, **kwargs):
        return _RangeCORSHandler(*args, directory=str(path.parent), **kwargs)

    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer((address, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://{address}:{port}/{path.name}"


def detect_segments(precomputed_path: Path) -> list[int] | None:
    """Read segment IDs from all shard labels files in the mesh directory."""
    try:
        info = json.loads((precomputed_path / "info").read_text())
        mesh_dir = precomputed_path / info["mesh"]
    except (KeyError, json.JSONDecodeError):
        return None

    labels_files = sorted(mesh_dir.glob("*.labels.gz"))
    if not labels_files:
        return None

    all_segments: list[int] = []
    for lf in labels_files:
        with gzip.open(lf, "rb") as f:
            all_segments.extend(int(s) for s in json.loads(f.read()))

    return sorted(set(all_segments))
