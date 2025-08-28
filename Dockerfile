FROM ghcr.io/osgeo/gdal:ubuntu-small-3.11.3

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV GDAL_DISABLE_READDIR_ON_OPEN=TRUE
ENV GDAL_CACHEMAX=500
ENV GDAL_MAX_DATASET_POOL_SIZE=4096
ENV GDAL_SWATH_SIZE=100000000
ENV GTIFF_DIRECT_IO=YES
ENV GTIFF_VIRTUAL_MEM_IO=YES
ENV VSI_CACHE=True
ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3/dist-packages

RUN apt-get update && apt-get install -y \
  python3-pip \
  wget

# Set the working directory to /app
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install . --break-system-packages

ENTRYPOINT ["deforest"]
CMD ["--help"]
