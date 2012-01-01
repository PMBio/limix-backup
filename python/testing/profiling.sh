export CPUPROFILE=/tmp/prof.out
python ./gp_test.py
pprof --dot ./../_gpmix.so /tmp/prof.out > ./prof.dot 
dot -oprof.png -Tpng prof.dot