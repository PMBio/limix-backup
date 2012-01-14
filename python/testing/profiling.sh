export CPUPROFILE=/tmp/prof.out
python ./test_lmm.py
pprof --dot ./../_gpmix.so /tmp/prof.out > ./prof.dot 
dot -oprof.png -Tpng prof.dot
