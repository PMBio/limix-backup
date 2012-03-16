#find R related stuff
find_package(R REQUIRED)


if (BUILD_R_PACKAGE)
  find_package(SWIG REQUIRED)

  INCLUDE(${SWIG_USE_FILE})

  include_directories(./../)

  set(CMAKE_SWIG_FLAGS "")

  #add google profiling code if enable:
  if (GOOGLE_PERF)
  SET(SWIGLINKLIBS profiler)
  endif (GOOGLE_PERF)


  #Oli: changed to R_INCLUDEDIR
  include_directories(${R_INCLUDEDIR})
  include_directories(${R_INCLUDEDIR}/x86_64)
  include_directories(${R_INCLUDEDIR}/i386)

  set_source_files_properties(peer.i PROPERTIES CPLUSPLUS ON)
  set_source_files_properties(peer.i PROPERTIES SWIG_FLAGS "-includeall")
  set_source_files_properties(peer.i PROPERTIES SWIG_FLAGS "-cpperraswarn")
  swig_add_module(gpmixR r gpmix.i)
  swig_link_libraries(gpmixR gpmixlib nlopt ${LIBR_SO})
   
  IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")	 
    # Mac OS X specific code
    # Without this flag it does not find some of the libraries
    SET(CMAKE_MODULE_LINKER_FLAGS "-undefined dynamic_lookup")
  ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")


#  ADD_CUSTOM_COMMAND(
#    TARGET peer POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_directory
#       ${PEER_SOURCE_DIR}/R/peer 
#       ${PEER_SOURCE_DIR}/build/R/peer)

#  ADD_CUSTOM_COMMAND(
#    TARGET peer POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy
#       ${PEER_SOURCE_DIR}/build/R/peerR_wrap.cxx
#       ${PEER_SOURCE_DIR}/build/R/peer/src/peerR_wrap.cpp)

#  ADD_CUSTOM_COMMAND(
#    TARGET peer POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy
#       ${PEER_SOURCE_DIR}/build/R/peer.R
#       ${PEER_SOURCE_DIR}/build/R/peer/R/peer.R)

#  if (BUILD_UNIVERSAL)
#    ADD_CUSTOM_COMMAND(
#      TARGET peer POST_BUILD
#     COMMAND R CMD INSTALL --build ${PEER_SOURCE_DIR}/build/R/peer
#      COMMENT "Building and installing the peer R package")
#  else (BUILD_UNIVERSAL)
#    ADD_CUSTOM_COMMAND(
#     TARGET peer POST_BUILD
#      COMMAND R CMD INSTALL --no-multiarch --build ${PEER_SOURCE_DIR}/build/R/peer
#      COMMENT "Building and installing the peer R package")
#  endif (BUILD_UNIVERSAL)

endif(BUILD_R_PACKAGE)