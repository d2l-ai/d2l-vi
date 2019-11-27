# Selecting Servers and GPUs
:label:`sec_buy_gpu`

Deep learning training generally requires large volumes of computing resources. Currently, GPUs are the most common computation acceleration hardware used for deep learning. Compared with CPUs, GPUs are cheaper and provide more intensive computing. On the one hand, GPUs can deliver the same compute power at a tenth of the price of CPUs. On the other hand, a single sever can generally support 8 or 16 GPUs. Therefore, the GPU quantity can be viewed as a standard to measure the deep learning compute power of a server.

<<<<<<< HEAD:chapter_appendix/buy-gpu.md
## Selecting a GPU
=======
## Selecting Servers
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/buy-gpu.md

At present, AMD and NVIDIA are the two main manufacturers of dedicated GPUs. NVIDIA was the first to enter the deep learning field and provides better support for deep learning frameworks. Therefore, most buyers choose NVIDIA GPUs.

<<<<<<< HEAD:chapter_appendix/buy-gpu.md
NVIDIA provides two types of GPUs, targeting individual uses (such as the GTX series) and enterprise users (such as the Tesla series). The two types of GPUs provide comparable compute power. However, the enterprise user GPUs generally use passive heat dissipation and add a memory check function. Therefore, these GPUs are more suitable for data centers and usually cost ten times more than individual user GPUs.
=======
1. **Power Supply**. GPUs use significant amounts of power. Budget with up to 350W per device (check for the *peak demand* of the graphics card rather than typical demand, since efficient code can use lots of energy. If your power supply is not up to the demand you will find that your system becomes unstable. 
1. **Chassis Size**. GPUs are large and the auxiliary power connectors often need extra space. Also, large chassis are easier to cool.
1. **GPU Cooling**. If you have large numbers of GPUs you might want to invest in water cooling. Also, aim for *reference designs* even if they have fewer fans, since they are thin enough to allow for air intake between the devices. If you buy a multi-fan GPU it might be too thick to get enough air when installing multiple GPUs and you will run into thermal throttling.
1. **PCIe Slots**. Moving data to and from the GPU (and exchanging it between GPUs) requires lots of bandwidth. We recommend PCIe 3.0 slots with 16 lanes. If you mount multiple GPUs, be sure to carefully read the motherboard description to ensure that 16x bandwidth is still available when multiple GPUs are used at the same time and that you are getting PCIe 3.0 as opposed to PCIe 2.0 for the additional slots. Some motherboards downgrade to 8x or even 4x bandwidth with multiple GPUs installed. This is partly due to the number of PCIe lanes that the CPU offers. 
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/buy-gpu.md

If you are a large company with 100 or more servers, you should consider the NVIDIA Tesla series for enterprise users. If you are a lab or small to mid-size company with 10 to 100 servers, you should consider the NVIDIA DGX series if your budget is sufficient. Otherwise, you can consider more cost-effective servers, such as Supermicro, and then purchase and install GTX series GPUs.

<<<<<<< HEAD:chapter_appendix/buy-gpu.md
NVIDIA generally releases a new GPU version every one or two years, such as the GTX 1000 series released in 2017. Each series offers several different models that provide different performance levels.

GPU performance is primarily a combination of the following three parameters:
=======
* **Beginner**. Buy a low end GPU with low power consumption (cheap gaming GPUs suitable for deep learning use 150-200W). If you are lucky your current computer will support it.
* **1 GPU**. A low-end CPU with 4 cores will be plenty sufficient and most motherboards suffice. Aim for at least 32GB DRAM and invest into an SSD for local data access. A power supply with 600W should be sufficient. Buy a GPU with lots of fans. 
* **2 GPUs**. A low-end CPU with 4-6 cores will suffice. Aim for 64GB DRAM and invest into an SSD. You will need in the order of 1000W for two high-end GPUs. In terms of mainboards, make sure that they have *two* PCIe 3.0 x16 slots. If you can, get a mainboard that has two free spaces (60mm spacing) between the PCIe 3.0 x16 slots for extra air. In this case, buy two GPUs with lots of fans.
* **4 GPUs**. Make sure that you buy a CPU with relatively fast single-thread speed (i.e., high clock frequency). You will probably need a CPU with a larger number of PCIe lanes, such as an AMD Threadripper. You will likely need relatively expensive mainboards to get 4 PCIe 3.0 x16 slots since they probably need a PLX to multiplex the PCIe lanes. Buy GPUs with reference design that are narrow and let air in between the GPUs. You need a 1600-2000W power supply and the outlet in your office might not support that. This server will probably run *loud and hot*. You do not want it under your desk. 128GB of DRAM is recommended. Get an SSD (1-2TB NVMe) for local storage and a bunch of hard disks in RAID configuration to store your data.
* **8 GPUs**. You need to buy a dedicated multi-GPU server chassis with multiple redundant power supplies (e.g., 2+1 for 1600W per power supply). This will require dual socket server CPUs, 256GB ECC DRAM, a fast network card (10GbE recommended), and you will need to check whether the servers support the *physical form factor* of the GPUs. Airflow and wiring placement differ significantly between consumer and server GPUs (e.g., RTX 2080 vs. Tesla V100). This means that you might not be able to install the consumer GPU in a server due to insufficient clearance for the power cable or lack of a suitable wiring harness (as one of the coauthors painfully discovered). 

## Selecting GPUs
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/buy-gpu.md

1. Compute power: Generally we look for 32-bit floating-point compute power. 16-bit floating point training is also entering the mainstream. If you are only interested in prediction, you can also use 8-bit integer.
2. Memory size: As your models become larger or the batches used during training grow bigger, you will need more GPU memory.
3. Memory bandwidth: You can only get the most out of your compute power when you have sufficient memory bandwidth.

<<<<<<< HEAD:chapter_appendix/buy-gpu.md
For most users, it is enough to look at compute power. The GPU memory should be no less than 4 GB. However, if the GPU must simultaneously display graphical interfaces, we recommend a memory size of at least 6 GB. There is generally not much variation in memory bandwidth, with few options to choose from.
=======
NVIDIA provides two types of GPUs, targeting individual users (e.g., via the GTX and RTX series) and enterprise users (via its Tesla series). The two types of GPUs provide comparable compute power. However, the enterprise user GPUs generally use (passive) forced cooling, more memory, and ECC (error correcting) memory. These GPUs are more suitable for data centers and usually cost ten times more than consumer GPUs.
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/buy-gpu.md

Figure 12.19 compares the 32-bit floating-point compute power and price of the various GTX 900 and 1000 series models. The prices are the suggested prices found on Wikipedia.

![Floating-point compute power and price comparison. ](../img/gtx.png)

From Figure 12.19, we can see two things:

1. Within each series, price and performance are roughly proportional. However, the newer models offer better cost effectiveness, as can be seen by comparing the 980 Ti and 1080 Ti.
2. The performance to cost ratio of the GTX 1000 series is about two times greater than the 900 series.

<<<<<<< HEAD:chapter_appendix/buy-gpu.md
If we look at the earlier GTX series, we will observe a similar pattern. Therefore, we recommend you buy the latest GPU model in your budget.

=======
:numref:`fig_flopsvsprice` compares the 32-bit floating-point compute power and price of the various GTX 900, GTX 1000 and RTX 2000 series models. The prices are the suggested prices found on Wikipedia.

![Floating-point compute power and price comparison. ](../img/flopsvsprice.svg)
:label:`fig_flopsvsprice`
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/buy-gpu.md

## Machine Configuration

Generally, GPUs are primarily used for deep learning training. Therefore, you do not have to purchase high-end CPUs. When deciding on machine configurations, you can find a mid to high-end configuration based on recommendations on the Internet. However, given the power consumption, heat dissipation performance, and size of GPUs, you need to consider three additional factors in machine configurations.

<<<<<<< HEAD:chapter_appendix/buy-gpu.md
1. Chassis size: GPUs are relatively large, so you should look for a large chassis with a built-in fan.
2. Power source: When purchasing GPUs, you must check the power consumption, as they can range from 50 W to 300 W. When choosing a power source, you must ensure it provides sufficient power and will not overload the data center power supply.
3. Motherboard PCIe card slot: We recommend PCIe 3.0 16x to ensure sufficient bandwidth between the GPU and main memory. If you mount multiple GPUs, be sure to carefully read the motherboard description to ensure that 16x bandwidth is still available when multiple GPUs are used at the same time. Be aware that some motherboards downgrade to 8x or even 4x bandwidth when 4 GPUs are mounted.
=======
![Floating-point compute power and energy consumption. ](../img/wattvsprice.svg)
:label:`fig_wattvsprice`


:numref:`fig_wattvsprice` shows how energy consumption scales mostly linearly with the amount of computation. Second, later generations are more efficient. This seems to be contradicted by the graph corresponding to the RTX 2000 series. However, this is a consequence of the TensorCores which draw disproportionately much energy. 
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_appendix_tools/buy-gpu.md


## Summary

* You should purchase the latest GPU model that you can afford.
* When deciding on machine configurations, you must consider GPU power consumption, heat dissipation, and size.

## Exercise

* You can browse the discussions about machine configurations in the forum for this section.

## [Discussions](https://discuss.mxnet.io/t/2400)

![](../img/qr_buy-gpu.svg)
