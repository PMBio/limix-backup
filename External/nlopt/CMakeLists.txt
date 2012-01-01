
#set source directory for eigen:
SET(SOURCES util/timer.c
                      util/stop.c
		      api/general.c
		      api/options.c
		      api/optimize.c
		      api/deprecated.c
		      slsqp/slsqp.c
		      luksan/mssubs.c
		      luksan/plip.c
		      luksan/plis.c
		      luksan/pnet.c
		      mma/mma.c
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


