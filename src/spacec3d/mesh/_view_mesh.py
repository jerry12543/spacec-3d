import gzip
import http.server
import json
import re
import socketserver
import threading
import webbrowser
from os import PathLike
from pathlib import Path

import neuroglancer


def view_precomputed_mesh(
    mesh_dir: str | PathLike[str],
    address: str = "127.0.0.1",
    data_port: int = 9010,
    segments: list[int] | None = None,
) -> None:
    """Loads the mesh into neuroglancer."""
    mesh_dir = Path(mesh_dir).resolve()
    if not mesh_dir.exists():
        raise SystemExit(f"Not found: {mesh_dir}")

    segments = segments or detect_segments(mesh_dir)
    if not segments:
        raise SystemExit(f"No mesh segments found in {mesh_dir}.")
    print(f"pre-selecting segments: {segments}", flush=True)

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
        s.layers["segmentation"] = neuroglancer.SegmentationLayer(**layer_kwargs)
        s.layout = "3d"
        s.show_default_annotations = False
        s.show_slices = False

    viewer_url = viewer.get_viewer_url()
    print(f"\nNeuroglancer URL:\n  {viewer_url}\n", flush=True)

    webbrowser.open(viewer_url)

    input("Press Enter to stop.\n")


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
