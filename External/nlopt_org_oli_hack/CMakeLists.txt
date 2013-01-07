
#set source directory for eigen:
SET(SOURCES util/timer.c
                      util/stop.c
		      util/mt19937ar.c
		      util/qsort_r.c
		      util/redblack.c
		      util/rescale.c
		      util/sobolseq.c
		      api/general.c
		      api/options.c
		      api/optimize.c
		      api/deprecated.c
		      slsqp/slsqp.c
		      luksan/mssubs.c
		      luksan/plip.c
		      luksan/plis.c
		      luksan/pnet.c
		      luksan/pssubs.c
		      mma/mma.c
		      cdirect/cdirect.c
		      neldermead/nldrmd.c
		      neldermead/sbplx.c
		      isres/isres.c
		      cobyla/cobyla.c
		      crs/crs.c
		      auglag/auglag.c
		      newuoa/newuoa.c 
		      mlsl/mlsl.c
		      direct/direct_wrap.c
		      direct/DIRect.c
		      direct/DIRserial.c
		      direct/DIRsubrout.c
		      bobyqa/bobyqa.c
		      praxis/praxis.c
)

include_directories(./)
include_directories(./api)
include_directories(./util)
include_directories(./praxis)
include_directories(./mma)
include_directories(./direct)
include_directories(./cdirect)
include_directories(./crs)
include_directories(./mlsl)
include_directories(./luksan)
include_directories(./cobyla)
include_directories(./newuoa)
include_directories(./neldermead)
include_directories(./auglag)
include_directories(./bobyqa)
include_directories(./isres)
include_directories(./slsqp)
ADD_LIBRARY(nlopt STATIC ${SOURCES})


