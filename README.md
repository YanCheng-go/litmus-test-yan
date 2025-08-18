# Litmus test

_A decisively indicative test._

Our main product we call Smart Forest and Commodity Analytics (SFCA). It uses
satellite imagery to inform our clients about the status and risks in their areas. The biggest component of this service are alerts on where (illegal) deforestation is happening.

The objective for you will be to use some of the components of our stack to create a basic _deforestation detector_, using satellite data to identify disappearing forest.

### Getting started


Clone this repo using [`git clone`](https://www.atlassian.com/git/tutorials/setting-up-a-repository/git-clone). 
You can find the blue clone button in the top right of the repo page. You can either use the https option and sign in 
with your username and password, or you can use the ssh option, but you'd need to set up
[ssh keys](https://docs.gitlab.com/ee/user/ssh.html) first

After you have cloned the repo you can enter the directory and start going! You have 2 options:

    1. Use Docker as explained below
    2. Use your own preferred way of developing using e.g. a virtual env. Check the requirements.txt for necessary libs

#### Docker
If you want to use the docker option, first  install Docker on [Windows](https://docs.docker.com/docker-for-windows/install/)
, or for Linux Ubuntu from the repositories with apt-get: `sudo apt-get install docker.io`
 
After installing Docker, build the image locally:

`docker build -t litmustest . `

NOTE for unix users: if you do not have permissions to connect to the docker daemon, add yourself to the docker group: 

`sudo usermod -a -G docker $USER`

You might have to log out and back in again!

Run docker with a binding of the current workdir at `/app`:

Linux: `docker run -v $(pwd):/app -t -i litmustest /bin/bash`

Windows Powershell: `docker run -v ${pwd}:/app/ -t -i litmustest /bin/bash`

Windows cmd: `docker run -v %cd%:/app/ -t -i litmustest /bin/bash`


### The input Data
You can retrieve the input data from within the docker container or inside your own terminal using wget. wget comes 
preinstalled in the docker image, but if you are developing locally you can either install wget yourself or get all 
3 images by yourself using the url in the code section below. The url starts with `https://storage..`

```bash
mkdir data
cd data
wget https://storage.googleapis.com/s11-litmustest/Peninsular_Malaysia_{2016..2017}_{1..2}_Landsat8.tif
```

The data you just downloaded are 3 Landsat-8 half year image composites, covering a small area on the Malaysia Peninsular. 
Landsat-8 is the eighth NASA satellite of the landsat series. It measures reflectance in the so called optical part of 
the electromagnetic spectrum. Besides the familiar blue, green and red, it measures near infrared, shortwave infrared 
and thermal infrared.

2016_1 rgb image looks like this:

![pm_2016_1_rgb](https://storage.googleapis.com/s11-litmustest/Screenshot%20from%202018-02-01%2014-00-16.png)

The landsat bands are particularly suited for detecting vegetation characteristics. A forest for the human eye 
is mostly green and no forest is often brownish or grey. For landsat a more effective measure of forest is the 
relative reflectance in the shortwave infrared compared to near infrared. Using Landsat's `1.5 * swir / nir` 
will give high values for bare areas and low values for forested areas.

### A simple start
In the `litmus_test.py` file, you will find an example function for reading a tif into a Numpy array.

Run the script from within the docker:

`python3 litmus_test.py`

It should give some lines with information about the first image as output.
When you open the script in an editor you will also see some extra comments.

### The deforestation detector
Your task is to use the numpy arrays you can now create to identify **which** pixels in the sample data got deforested and **when** this has happened. The approach for doing this we leave up to you. We encourage you to be creative in your solution, but we are most interested in the implementation, contrary to the solution itself (use the swir/nir ratio mentioned above!). Besides testing your learning ability or proficiency with the libraries, we will look at readability, maintainability and scalability of your implementation. A unit test will be greatly appreciated. Feel free to take litmus-test.py as a starting point or create your own python script.

We ask you to spend around 6 hours on your implementation and the remaining 2 on describing the choices you made and what you would do next to make your work better. Please submit your work in a merge (pull) request.

Best of luck!
