FROM registry.gitlab.com/satelligence/dprof/dprof-img-process:2.3.0-4-g8fc1c86

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

RUN pip3 install --upgrade pip

# Set the working directory to /app
WORKDIR /app

# Copy the application's requirements.txt and run pip to install all
# dependencies. Remove build artifact.
ADD requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt
RUN rm /app/requirements.txt
