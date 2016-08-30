CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( main )

FIND_PATH(IGRAPH_INCLUDE_DIR igraph/igraph.h
	PATHS
	$ENV{OSSIM_HOME}
	NO_DEFAULT_PATH
	PATH_SUFFIXES include.h
)

FIND_PATH(IGRAPH_INCLUDE_DIR igraph/igraph.h
	PATHS
	/home/covise/covise/extern_libs/include/igraph
	/usr/local/include/igraph
	/usr/local/include/<TAB>/igraph
	/usr/include/igraph
	/sw/include/igraph # Fink
	/opt/local/include/igraph # DarwinPorts
	/opt/csw/include/igraph # Blastwave
	/opt/include/igraph
	PATH_SUFFIXES include.h
)

FIND_LIBRARY( IGRAPH_LIBRARY 
	NAMES igraph
	PATHS $ENV{IGRAPH_HOME}
	NO_DEFAULT_PATH
	PATH_SUFFIXES lib64 lib dylib
)

FIND_LIBRARY( IGRAPH_LIBRARY 
	NAMES igraph
	PATHS
	/home/covise/covise/extern_libs
    /usr/local/Cellar/igraph/<TAB> #brew
    /usr/local
    /usr
    /sw
    /opt/local
    /opt/csw
    /opt
	/usr/freeware
	PATH_SUFFIXES lib64 lib dylib
)

SET( IGRAPH_FOUND "NO" )
IF( IGRAPH_LIBRARY AND IGRAPH_INCLUDE_DIR )
	SET( IGRAPH_FOUND "YES" )
	INCLUDE_DIRECTORIES( ${IGRAPH_INCLUDE_DIR} )
	LIST( APPEND ThirdParty_LIBS ${IGRAPH_LIBRARY} )
ELSE()
	MESSAGE (SEND_ERROR "igraph library not found" )
ENDIF( IGRAPH_LIBRARY AND IGRAPH_INCLUDE_DIR )

FIND_PACKAGE( OpenCV REQUIRED )
IF( OpenCV_FOUND )
	LIST( APPEND ThirdParty_LIBS ${OpenCV_LIBS} )
	INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
ELSE()
	MESSAGE (SEND_ERROR "OpenCV library not found" )
ENDIF( OpenCV_FOUND )

SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

ADD_EXECUTABLE( image_graph main.cpp )

TARGET_LINK_LIBRARIES( image_graph ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES( image_graph ${IGRAPH_LIBRARY} )
