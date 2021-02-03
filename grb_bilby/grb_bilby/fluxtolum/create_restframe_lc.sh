#!/usr/bin/bash

. ~/ciao-4.11/bin/ciao.sh

CIAO=~/ciao-4.11
if [ -d "~/ciao-4.11/contrib/lib/" ] ; then
    DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/ciao-4.11/contrib/lib/
fi
DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/ciao-4.11/lib:~/ciao-4.11/ots/lib
export DYLD_LIBRARY_PATH

#alias sherpa_on='export PATH=/Users/Antonia/sherpa/bin:$PATH'

#PYTHONPATH=$PYTHONPATH:/Users/Antonia/Downloads/ciao/ciao-4.6/lib/python2.7/site-packages:/Users/Antonia/Downloads/ciao/ciao-4.6/ots/lib/python2.7/site-packages:/Users/Antonia/Downloads/ciao/ciao-4.6/contrib/lib/python2.7/site-packages
#PYTHONPATH=/Users/Antonia/Downloads/ciao/ciao-4.6/lib/python2.7/site-packages:$PYTHONPATH
#export PYTHONPATH

python create_restframe_lc.py
