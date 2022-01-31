import hashlib

from flask import Flask
from flask import abort
from flask import make_response
from flask import render_template
from flask import url_for

from tiffslide.deepzoom import MinimalComputeAperioDZGenerator

app = Flask(__name__)

# note: this is not good practice, but it'll do for the sake of simplicity
_dzgen = None


@app.route("/")
def index():
    # noinspection PyProtectedMember
    fn = _dzgen._openfile.path  # just so we don't need to worry about caching
    su = url_for("dzi", image_fn=hashlib.sha1(fn.encode()).hexdigest())
    return render_template("index.html", filename=fn, slide_url=su)


@app.route("/<image_fn>.dzi")
def dzi(image_fn):
    resp = make_response(_dzgen.get_dzi())
    resp.mimetype = "application/xml"
    return resp


@app.route("/<image_fn>_files/<int:level>/<int:col>_<int:row>.jpeg")
def tile(image_fn, level, col, row):
    try:
        tile_bytes = _dzgen.get_tile(level, col, row)
    except (KeyError, IndexError):
        return abort(404)

    resp = make_response(tile_bytes)
    resp.mimetype = "image/jpeg"
    return resp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="server port")
    parser.add_argument("--bind", type=str, default="127.0.0.1", help="server address")
    parser.add_argument("svs_file", help="svs file")
    args = parser.parse_args()

    _dzgen = MinimalComputeAperioDZGenerator(args.svs_file)

    app.run(host=args.bind, port=args.port)
