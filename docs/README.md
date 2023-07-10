# TiffSlide Benchmarks

To run the benchmarks first install the dev environment

```shell
git clone https://github.com/bayer-science-for-a-better-life/tiffslide.git
cd tiffslide
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pip install -r docs/requirements.bench.txt
```

Then download the files to a local test folder

```shell
mkdir -p tiffslide_benchmark_files
cd tiffslide_benchmark_files

mkdir -p Aperio       && curl -L https://data.cytomine.coop/open/openslide/aperio-svs/CMU-2.svs             -o Aperio/CMU-2.svs
mkdir -p Generic-TIFF && curl -L https://data.cytomine.coop/open/openslide/generic-tiff/CMU-1.tiff          -o Generic-TIFF/CMU-2.svs
mkdir -p Hamamatsu    && curl -L https://data.cytomine.coop/open/openslide/hamamatsu-ndpi/OS-3.ndpi         -o Hamamatsu/OS-3.ndpi
mkdir -p Ventana      && curl -L https://data.cytomine.coop/open/openslide/ventana-bif/OS-2.bif             -o Ventana/OS-2.bif

mkdir -p Leica        && curl -L https://openslide.cs.cmu.edu/download/openslide-testdata/Leica/Leica-2.scn -o Leica/Leica-2.scn

mkdir -p Aperio-JP2K  && aws s3 cp --no-sign-request s3://tcga-2-open/2aa283f3-732c-4879-8d37-1fec3ccf5bdc/TCGA-05-4395-01Z-00-DX1.20205276-ca16-46b2-914a-fe5e576a5cf9.svs Aperio-JP2K/TCGA-05-4395-01Z-00-DX1.20205276-ca16-46b2-914a-fe5e576a5cf9.svs

pwd
```

Then generate the benchmarks by running:

```
OPENSLIDE_TESTDATA_DIR=/path/to/tiffslide_benchmark_files/ python docs/generate_benchmark_plots.py
```

This updates the two images in `./docs/images/`

To change the files the benchmark is run on, change the dictionary in [tiffslide/tests/test_benchmark.py](tiffslide/tests/test_benchmark.py)

Note: the benchmark script generates a file `.` that needs to be deleted to rerun the benchmarks.
