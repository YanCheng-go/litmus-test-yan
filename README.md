# Litmus test

_A decisively indicative test._

Our main product we call Smart Forest and Commodity Analytics (SFCA). It uses
satellite imagery to inform our clients about the status and risks in their areas. The biggest component of this service are alerts on where (illegal) deforestation is happening.

The objective for you will be to use some of the components of our stack to create a basic _deforestation detector_, using satellite data to identify disappearing forest.

### Getting started
To get you up-to-speed we created a docker container.

Start by installing Docker on [Windows](https://docs.docker.com/docker-for-windows/install/)
, or for Linux Ubuntu from the repositories with apt-get: `sudo apt-get install docker.io`

Clone this repo:

`git clone https://gitlab.com/satelligence/litmus-test.git`

Change dir to litmus-test

`cd litmus-test`

Pull docker image:

`docker pull docker.io/satelligence/litmus-test`

NOTE for unix users: if you do not have permissions to connect to the docker daemon, add yourself to the docker group: 

`sudo usermod -a -G docker $USER`

You might have to log out and back in again!

Run docker with a binding of the current workdir at `/app`:

Linux: `docker run -v $(pwd):/app -t -i satelligence/litmus-test /bin/bash`

Windows Powershell: `docker run -v ${pwd}:/app/ -t -i satelligence/litmus-test /bin/bash`

Windows cmd: `docker run -v %cd%:/app/ -t -i satelligence/litmus-test /bin/bash`

Download the sample data from within the docker, using wget:

```bash
mkdir data
cd data
wget https://storage.googleapis.com/s11-litmustest/Peninsular_Malaysia_{2016..2017}_{1..2}_Landsat8.tif
```

### The input Data
The data you just downloaded are 4 Landsat-8 half year image composites, covering a small area on the Malaysia Peninsular. Landsat-8 is the eighth NASA satellite of the landsat series. It measures reflectance in the so called optical part of the electromagnetic spectrum. Besides the familiar blue, green and red, it measures near infrared, shortwave infrared and thermal infrared.

2016_1 rgb image looks like this:

![pm_2016_1_rgb](https://storage.googleapis.com/s11-litmustest/Screenshot%20from%202018-02-01%2014-00-16.png)

The landsat bands are particularly suited for detecting vegetation characteristics. A forest for the human eye is mostly green and no forest is often brownish or grey. For landsat a more effective measure of forest is the relative reflectance in the shortwave infrared compared to near infrared. Using Landsat's `1.5 * swir / nir` will give high values for bare areas and low values for forested areas.

### A simple start
In the `litmus_test.py` file, you will find an example function for reading a tif into a Numpy array.

Run the script from within the docker:

`python3 litmus_test.py`

It should give some lines with information about the first image as output.
When you open the script in an editor you will also see some extra comments.

### The deforestation detector
Your task is to use the numpy arrays you can now create to identify **which** pixels in the sample data got deforested and **when** this has happened. The approach for doing this we leave up to you. We encourage you to be creative in your solution, but we are most interested in the implementation, contrary to the solution itself (use the swir/nir ratio mentioned above!). Besides testing your learning ability or proficiency with the libraries, we will look at readability, maintainability and scalability of your implementation. A unit test will be greatly appreciated. Feel free to take litmus-test.py as a starting point or create your own python script.

We ask you to spend around 6 hours on your implementation and the remaining 2 on describing the choices you made and what you would do next to make your work better. Please submit your work in a merge (pull) request on https://gitlab.com/satelligence/litmus-test before saturday 2300 hours.

Best of luck!
