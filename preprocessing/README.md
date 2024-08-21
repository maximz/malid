# Preprocessing commands

## Overview

`bcr.sh` and `tcr.sh` contain the preprocessing commands applied to our internal dataset. These Python 2 scripts were written by Krish Roskin.

Overview:

1. Paired-end reads are merged with FLASH (Fast Length Adjustment of SHort reads) v1.2.11.
2. Samples are demultiplexed by matching barcodes to the sample reads, and the barcodes and primers are trimmed off.
3. V, D, and J gene segments and junctional bases are annotated with IgBlast v1.3.0. The IgBlast reference data is provided [here](../igblast/)
4. For each person (across all of their samples, if there are multiple), CDR3 nucleotide sequences are grouped into clones using single linkage clustering.

Steps 3 and 4 are also implemented in the primary Mal-ID Python 3 codebase for use on AIRR and Adaptive input data.

## Installation

```bash
# Install FLASH from https://sourceforge.net/projects/flashpage/
wget https://cfhcable.dl.sourceforge.net/project/flashpage/FLASH-1.2.11.tar.gz
tar -zxvf FLASH-1.2.11.tar.gz
cd FLASH-1.2.11/
make
./flash --version # FLASH v1.2.11
sudo mv flash /usr/bin/flash
sudo chown root:root /usr/bin/flash
sudo chmod 755 /usr/bin/flash
cd ../
rm -r FLASH-1.2.11/

# Install Python 2
sudo apt-get update
sudo apt-get install python2-minimal
python2 -V # Python 2.7.18

# Install pip2
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
sudo python2 get-pip.py
pip2 --version # pip 20.3.4 from /usr/local/lib/python2.7/dist-packages/pip (python 2.7)

# Note that now we have pip2 and pip (pointing to pip2), but no pip3. This can make things confusing with our default systemwide Python 3 installation.
# So let's get a pip3:
sudo apt-get install python3-pip
# And change pip to point to pip3, not pip2:
sudo python3 -m pip install --upgrade --force pip
pip --version # pip 21.3.1 from /home/maxim/.local/lib/python3.8/site-packages/pip (python 3.8)
pip3 --version # pip 21.3.1 from /home/maxim/.local/lib/python3.8/site-packages/pip (python 3.8)

# System dependencies for Python packages
sudo apt-get install python2-dev -y  # for Python.h
sudo apt-get install libxml2-dev libxslt-dev -y # for lxml. automatically switches to libxslt1-dev
sudo apt-get install libgmp3-dev -y # gmpy
sudo apt-get install libcups2-dev -y # pycups
sudo apt-get install graphviz graphviz-dev pkg-config -y # pygraphviz - definitely need pkg-config
sudo apt-get install libfreetype6-dev -y # for matplotlib
sudo apt-get install libpq-dev -y # for psycopg2
sudo apt-get install libhdf5-dev -y # for tables
sudo apt-get install libcurl4-openssl-dev libssl-dev -y # for pycurl

# Python 2 environment installation
sudo python2 -m easy_install distribute
sudo pip2 install --upgrade distribute
sudo pip2 install -r py27_requirements.txt
sudo pip2 install -r py27_requirements_dependent_on_others.txt

# Create a way to activate Python 2
# The pipeline scripts use `#!/usr/bin/env python`, which doesn't respect aliases.
# So we need to put python2 in the PATH under the name "python", while avoiding setting python=python2 systemwide (which would break Python 3).
mkdir ~/boydlab/pyenv
ln -s /usr/bin/python2 ~/boydlab/pyenv/python
echo 'export PATH="$HOME/boydlab/pyenv:$PATH"' > ~/boydlab/pyenv/activate
```

Whenever running these scripts, first activate the Python 2 environment by running:

```bash
# Run every time
source ~/boydlab/pyenv/activate

# Result:
which python # /home/maxim/boydlab/pyenv/python
```
