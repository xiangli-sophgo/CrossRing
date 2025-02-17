# Gem5 SPC.k2k bandwidth analyzer

## Legend Explanation
 
The primary colors (red, green, blue) represent DDR, L2M, SLC0 respectively;
the complementary colors (cyan, magenta, yellow) represent GDMA, SDMA, CDMA.

## Argument Description

***'-t'***: path to the Flit_Info.txt. (`m5out/Flit_Info.txt` by default).

***'-d'***: directory to save results (`m5out/perf_results` by default).

***'-cg'***: the group of cores, each core is a group by default.

***'-bn'***: number of bars, used to adjust the time steps.

***'--all'***: all cores as one group for bandwidth stacking.

***'--stack'***: stack the bandwidth of all IPs for each group.

## How to Use

**Eg.1**: all cores as one group for bandwidth stacking.
```shell
python3 tpu_perf_txt.py -t path/to/txt --all
```

**Eg.2**: stack the bandwidth of all IPs for all cores.
```shell
python3 tpu_perf_txt.py -t path/to/txt --all --stack
```

**Eg.3**: each core as one group by default.
```shell
python3 tpu_perf_txt.py -t path/to/txt
```

**Eg.4**: stack the bandwidth of all IPs for each core.
```shell
python3 tpu_perf_txt.py -t path/to/txt --stack
```

## How to Run

```shell
bash run_bw.sh 
```
or
```shell
bash run_bw.sh path/to/Flit_Info.txt
```
