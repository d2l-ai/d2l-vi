<<<<<<< HEAD:chapter_appendix/aws.md
# Using AWS to Run Code
=======
# Using AWS Instances
:label:`sec_aws`
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

If your local machine has limited computing resources, you can use cloud computing services to obtain more powerful computing resources and use them to run the deep learning code in this document. In this section, we will show you how to apply for instances and use Jupyter Notebook to run code on AWS (Amazon's cloud computing service). The example here includes two steps:

1. Apply for a K80 GPU "p2.xlarge" instance.
2. Install CUDA and the corresponding MXNet GPU version.

The process to apply for other instance types and install other MXNet versions is basically the same as that described here.


<<<<<<< HEAD:chapter_appendix/aws.md
## Apply for an Account and Log In
=======
## Registering Account and Logging In
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

First, we need to register an account at https://aws.amazon.com/. It usually requires a credit card.

After logging into your AWS account, click "EC2" (marked by the red box in Figure 12.8) to go to the EC2 panel.

![ Log into your AWS account. ](../img/aws.png)


## Creating and Running an EC2 Instance

<<<<<<< HEAD:chapter_appendix/aws.md
Figure 12.9 shows the EC2 panel. In the area marked by the red box in Figure 12.9, select a nearby data center to reduce latency. If you are located in China you can select a nearby Asia Pacific region, such as Asia Pacific (Seoul). Please note that some data centers may not have GPU instances. Click the "Launch Instance" button marked by the red box in Figure 12.8 to launch your instance.
=======
:numref:`fig_ec2` shows the EC2 panel with sensitive account information greyed out.
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

![ EC2 panel. ](../img/ec2.png)

<<<<<<< HEAD:chapter_appendix/aws.md

The row at the top of Figure 12.10 shows the seven steps in the instance configuration process. In the first step "1. Choose AMI", choose Ubuntu 16.04 for the operating system.

![ Choose an operating system. ](../img/os.png)
=======
### Presetting Location
Select a nearby data center to reduce latency, *e.g.,* "Oregon". (marked by the red box in the top-right of :numref:`fig_ec2`) If you are located in China
you can select a nearby Asia Pacific region, such as Seoul or Tokyo. Please note
that some data centers may not have GPU instances.

### Increasing Limits
Before choosing an instance, check if there are quantity
restrictions by clicking the "Limits" label in the bar on the left as shown in
:numref:`fig_ec2`. :numref:`fig_limits` shows an example of such a
limitation. The account currently cannot open "p2.xlarge" instance per region. If
you need to open one or more instances, click on the "Request limit increase" link to
apply for a higher instance quota. Generally, it takes one business day to
process an application.

![ Instance quantity restrictions. ](../img/limits.png)
:width:`700px`
:label:`fig_limits`


### Launching Instance
Next, click the "Launch Instance" button marked by the red box in :numref:`fig_ec2` to launch your instance.

We begin by selecting a suitable AMI (AWS Machine Image). Enter "Ubuntu" in the search box (marked by the red box in :numref:`fig_ubuntu`):


![ Choose an operating system. ](../img/ubuntu_new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2 provides many different instance configurations to choose from. This can sometimes feel overwhelming to a beginner. Here's a table of suitable machines:

| Name | GPU         | Notes                         |
|------|-------------|-------------------------------|
| g2   | Grid K520   | ancient                       |
| p2   | Kepler K80  | old but often cheap as spot   |
| g3   | Maxwell M60 | good trade-off                |
| p3   | Volta V100  | high performance for FP16     |
| g4   | Turing T4   | inference optimized FP16/INT8 |

All the above servers come in multiple flavors indicating the number of GPUs used. For example, a p2.xlarge has 1 GPU and a p2.16xlarge has 16 GPUs and more memory. For more details see *e.g.,* the [AWS EC2 documentation](https://aws.amazon.com/ec2/instance-types/) or a [summary page](https://www.ec2instances.info). For the purpose of illustration, a p2.xlarge will suffice (marked in red box of :numref:`fig_p2x`).
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

EC2 provides many different instance configurations to choose from. As shown in Figure 12.11, In "Step 2: Choose an Instance Type”，choose a "p2.xlarge" instance with K80 GPU. We can also choose instances with multiple GPUs such as "p2.16xlarge". If you want to compare machine configurations and fees of different instances, you may refer to https://www.ec2instances.info/.

![ Choose an instance. ](../img/p2x.png)

<<<<<<< HEAD:chapter_appendix/aws.md
Before choosing an instance, we suggest you check if there are quantity restrictions by clicking the "Limits" label in the bar on the, as left shown in Figure 12.9. As shown in Figure 12.12, this account can only open one "p2.xlarge" instance per region. If you need to open more instances, click on the "Request limit increase" link to apply for a higher instance quota. Generally, it takes one business day to process an application.

![ Instance quantity restrictions. ](../img/limits.png)

In this example, we keep the default configurations for the steps "3. Configure Instance", "5. Add Tags", and "6. Configure Security Group". Tap on "4. Add Storage" and increase the default hard disk size to 40 GB. Note that you will need about 4 GB to install CUDA.
=======
So far, we have finished the first two of seven steps for launching an EC2 instance, as shown on the top of :numref:`fig_disk`. In this example, we keep the default configurations for the steps "3. Configure Instance", "5. Add Tags", and "6. Configure Security Group". Tap on "4. Add Storage" and increase the default hard disk size to 64 GB (marked in red box of :numref:`fig_disk`). Note that CUDA by itself already takes up 4GB.
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

![ Modify instance hard disk size. ](../img/disk.png)


<<<<<<< HEAD:chapter_appendix/aws.md
Finally, go to "7. Review" and click "Launch" to launch the configured instance. The system will now prompt you to select the key pair used to access the instance. If you do not have a key pair, select "Create a new key pair" in the first drop-down menu in Figure 12.14 to generate a key pair. Subsequently, you can select "Choose an existing key pair" for this menu and then select the previously generated key pair. Click "Launch Instances" to launch the created instance.
=======
![ Select a key pair. ](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

![ Select a key pair. ](../img/keypair.png)

Click the instance ID shown in Figure 12.15 to view the status of this instance.

<<<<<<< HEAD:chapter_appendix/aws.md
![C lick the instance ID. ](../img/launching.png)

As shown in Figure 12.16, after the instance state turns green, right-click the instance and select "Connect" to view the instance access method. For example, enter the following in the command line:
=======

### Connecting Instance

As shown in :numref:`fig_connect`, after the instance state turns green, right-click the instance and select `Connect` to view the instance access method.

![ View instance access and startup method. ](../img/connect.png)
:width:`700px`
:label:`fig_connect`

If this is a new key, it must not be publicly viewable for SSH to work. Go to the folder where you store `D2L_key.pem` (*e.g.,* Downloads folder) and make the key to be not publicly viewable.
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

```bash
cd /Downloads  ## if D2L_key.pem is stored in Downloads folder
chmod 400 D2L_key.pem
```


<<<<<<< HEAD:chapter_appendix/aws.md
![ View instance access and startup method. ](../img/connect.png)


## Install CUDA

If you log into a GPU instance, you need to download and install CUDA. First, update and install the package needed for compilation.

```
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```

NVIDIA releases a major version of CUDA every year. Here we download the latest CUDA 9.0 when the book is written. Visit the official website of NVIDIA (https://developer.nvidia.com/cuda-90-download-archive) to obtain the download link of CUDA 9.0, as shown in Figure 12.17.

![Find the CUDA 9.0 download address. ](../img/cuda.png)


After finding the download address, download and install CUDA 9.0. For example:
=======
![ View instance access and startup method. ](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`


Now, copy the ssh command in the lower red box of :numref:`fig_chmod` and paste onto the command line:

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```


When the command line prompts "Are you sure you want to continue connecting (yes/no)", enter "yes" and press Enter to log into the instance.

Your server is ready now.


## Installing CUDA

Before installing CUDA, be sure to update the instance with the latest drivers.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```


Here we download CUDA 10.1. Visit NVIDIA's official repository at (https://developer.nvidia.com/cuda-downloads) to find the download link of CUDA 10.1 as shown in :numref:`fig_cuda`.

![Find the CUDA 10.1 download address. ](../img/cuda101.png)
:width:`500px`
:label:`fig_cuda`
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

Copy the instructions and paste them into the terminal to install
CUDA 10.1.

```bash
## Paste the copied link from CUDA website
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
<<<<<<< HEAD:chapter_appendix/aws.md
# The download link and file name are subject to change, so always use those
# from the NVIDIA website
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
sudo sh cuda_9.0.176_384.81_linux-run
```
=======
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md


After installing the program, run the following command to view the GPUs.

```bash
nvidia-smi
```
<<<<<<< HEAD:chapter_appendix/aws.md
Do you accept the previously read EULA?
accept/decline/quit: accept
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81?
(y)es/(n)o/(q)uit: y
Do you want to install the OpenGL libraries?
(y)es/(n)o/(q)uit [ default is yes ]: y
Do you want to run nvidia-xconfig?
This will ... vendors.
(y)es/(n)o/(q)uit [ default is no ]: n
Install the CUDA 9.0 Toolkit?
(y)es/(n)o/(q)uit: y
Enter Toolkit Location
 [ default is /usr/local/cuda-9.0 ]:
Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: y
Install the CUDA 9.0 Samples?
(y)es/(n)o/(q)uit: n
=======


Finally, add CUDA to the library path to help other libraries find it.

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> ~/.bashrc
```


## Installing MXNet and Downloading the D2L Notebooks

First, to simplify the installation, you need to install [Miniconda](https://conda.io/en/latest/miniconda.html) for Linux. The download link and file name are subject to changes, so please go the Miniconda website and click "Copy Link Address" as shown in :numref:`fig_miniconda`.

![ Download Miniconda. ](../img/miniconda.png)
:width:`700px`
:label:`fig_miniconda`

```bash
# The link and file name are subject to changes
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md
```


After Miniconda installation, run the following command to activate CUDA and conda.

```bash
~/miniconda3/bin/conda init
source ~/.bashrc
```


<<<<<<< HEAD:chapter_appendix/aws.md
```
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64" >> ~/.bashrc
```

## Acquire the Code for this Book and Install MXNet GPU Version

We have introduced the way to obtaining code of the book and setting up the running environment in Section ["Getting started with Gluon"](../chapter_prerequisite/install.md). First, install Miniconda of the Linux version (website: https://conda.io/miniconda.html), such as
=======
Next, download the code for this book.

```bash
sudo apt-get install unzip
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-0.7.0.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```


Then create the conda `d2l` environment and enter `y` to proceed with the installation.
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

```bash
conda create --name d2l -y
```


After creating the `d2l` environment, activate it and install `pip`.

```bash
conda activate d2l
conda install python=3.7 pip -y
```

<<<<<<< HEAD:chapter_appendix/aws.md
After installation, run `source ~/.bashrc` once to activate CUDA and Conda. Next, download the code for this book and install and activate the Conda environment.
=======
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

Finally, install MXNet and the `d2l` package. The postfix `cu101` means that this is the CUDA 10.1 variant. For different versions, say only CUDA 10.0, you would want to choose `cu100` instead.

```bash
pip install mxnet-cu101==1.6.0b20191122
pip install d2l==0.11.0
```
<<<<<<< HEAD:chapter_appendix/aws.md
mkdir d2l-en && cd d2l-en
curl https://www.d2l.ai/d2l-en-1.0.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
conda env create -f environment.yml
source activate gluon
```
=======

>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

The MXNet CPU version is installed in the environment by default. Now, you must replace it with the MXNet GPU version. As the CUDA version is 9.0, install `mxnet-cu90`. Generally speaking, if your CUDA version is x.y, the corresponding MXNET version is `mxnet-cuxy`.

```
<<<<<<< HEAD:chapter_appendix/aws.md
pip uninstall mxnet
# X.Y.Z should be replaced with the version number depended on by the book
pip install mxnet-cu90==X.Y.Z
```

## Run Jupyter Notebook

Now, you can run Jupyter Notebook:

```
jupyter notebook
```

Figure 12.18 shows the possible output after you run Jupyter Notebook. The last row is the URL for port 8888.
=======
$ python
>>> from mxnet import np, npx
>>> np.zeros((1024, 1024), ctx=npx.gpu())
```


## Running Jupyter

To run Jupyter remotely you need to use SSH port forwarding. After all, the server in the cloud does not have a monitor or keyboard. For this, log into your server from your desktop (or laptop) as follows.

```
# This command must be run in the local command line
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
conda activate d2l
jupyter notebook
```


:numref:`fig_jupyter` shows the possible output after you run Jupyter Notebook. The last row is the URL for port 8888.
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

![ Output after running Jupyter Notebook. The last row is the URL for port 8888. ](../img/jupyter.png)

Because the instance you created does not expose port 8888, you can launch SSH in the local command line and map the instance to the local port 8889.

```
# This command must be run in the local command line
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
```

Finally, copy the URL shown in the last line of the Jupyter Notebook output in Figure 12.18 to your local browser and change 8888 to 8889. Press Enter to use Jupyter Notebook to run the instance code from your local browser.

## Close Unused Instances

As cloud services are billed by use duration, you will generally want to close instances you no longer use.

<<<<<<< HEAD:chapter_appendix/aws.md
If you plan on restarting the instance after a short time, right-click on the example shown in Figure 12.16 and select "Instance State" $\rightarrow$ "Stop" to stop the instance. When you want to use it again, select "Instance State" $\rightarrow$ "Start" to restart the instance. In this situation, the restarted instance will retain the information stored on its hard disk before it was stopped (for example, you do not have to reinstall CUDA and other runtime environments). However, stopped instances will still be billed a small amount for the hard disk space retained.

If you do not plan to use the instance again for a long time, right-click on the example in Figure 12.16 and select "Image" $\rightarrow$ "Create" to create an image of the instance. Then, select "Instance State" $\rightarrow$ "Terminate" to terminate the instance (it will no longer be billed for hard disk space). The next time you want to use this instance, you can follow the steps for creating and running an EC2 instance described in this section to create an instance based on the saved image. The only difference is that, in "1. Choose AMI" shown in Figure 12.10, you must use the "My AMIs" option on the left to select your saved image. The created instance will retain the information stored on the image hard disk. For example, you will not have to reinstall CUDA and other runtime environments.
=======
As cloud services are billed by the time of use, you should close instances that are not being used. Note that there are alternatives: "Stopping" an instance means that you will be able to start it again. This is akin to switching off the power for your regular server. However, stopped instances will still be billed a small amount for the hard disk space retained. "Terminate" deletes all data associated with it. This includes the disk, hence you cannot start it again. Only do this if you know that you will not need it in the future.

If you want to use the instance as a template for many more instances,
right-click on the example in Figure 14.16 :numref:`fig_connect` and select "Image" $\rightarrow$
"Create" to create an image of the instance. Once this is complete, select
"Instance State" $\rightarrow$ "Terminate" to terminate the instance. The next
time you want to use this instance, you can follow the steps for creating and
running an EC2 instance described in this section to create an instance based on
the saved image. The only difference is that, in "1. Choose AMI" shown in
:numref:`fig_ubuntu`, you must use the "My AMIs" option on the left to select your saved
image. The created instance will retain the information stored on the image hard
disk. For example, you will not have to reinstall CUDA and other runtime
environments.
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/aws.md

## Summary

* You can use cloud computing services to obtain more powerful computing resources and use them to run the deep learning code in this document.

## Exercise

* The cloud offers convenience, but it does not come cheap. Research the prices of cloud services and find ways to reduce overhead.

## [Discussions](https://discuss.mxnet.io/t/2399)

![](../img/qr_aws.svg)
